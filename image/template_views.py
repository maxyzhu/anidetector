"""Server-rendered results page + annotated-image endpoint.

Two views:
  - results_page: an HTML gallery of processed images (Django template)
  - annotated_image: streams a supervision-annotated PNG for one image
"""

from __future__ import annotations

from django.db.models import Prefetch
from django.http import Http404, HttpResponse
from django.shortcuts import render

from core.models import Deployment, Image, SpeciesClassification
from core.visualize import annotate_image

# Prefetch each detection's SpeciesNet classifications, newest first, so the
# template's `d.classifications.all|first` is the latest one without extra queries.
_species_pf = Prefetch(
    "detections__classifications",
    queryset=SpeciesClassification.objects.filter(
        source=SpeciesClassification.Source.SPECIESNET
    ).order_by("-created_at"),
)


def results_page(request):
    """GET /results/ — gallery of processed images with filters.

    Query params (all optional, combine freely):
      ?show=all              include blank frames too (default hides them)
      ?category=animal       images having a detection of that MegaDetector category
      ?deployment=<id>       images from that deployment
      ?species=<label>       images whose SpeciesNet classification == label
      ?min_species_conf=0.5  images with a SpeciesNet result >= this confidence
      ?after=YYYY-MM-DD      captured on/after this date
      ?before=YYYY-MM-DD     captured on/before this date
    """
    SPECIESNET = SpeciesClassification.Source.SPECIESNET
    # Video representative frames are Images too; this is the photo gallery.
    qs = Image.objects.filter(
        status=Image.Status.PROCESSED, source=Image.Source.INGEST
    )

    if request.GET.get("show") != "all":
        # Default view hides blanks — the whole point of the blank filter.
        qs = qs.filter(is_blank=False)

    category = request.GET.get("category")
    if category:
        qs = qs.filter(detections__category=category)

    deployment = request.GET.get("deployment")
    if deployment:
        qs = qs.filter(deployment_id=deployment)

    species = request.GET.get("species")
    if species:
        qs = qs.filter(
            detections__classifications__source=SPECIESNET,
            detections__classifications__category=species,
        )

    min_conf = request.GET.get("min_species_conf")
    if min_conf:
        try:
            qs = qs.filter(
                detections__classifications__source=SPECIESNET,
                detections__classifications__confidence__gte=float(min_conf),
            )
        except ValueError:
            pass  # ignore a non-numeric value rather than 500

    after = request.GET.get("after")
    if after:
        qs = qs.filter(captured_at__date__gte=after)
    before = request.GET.get("before")
    if before:
        qs = qs.filter(captured_at__date__lte=before)

    # Joins across detections/classifications can duplicate image rows.
    qs = qs.select_related("deployment").distinct().prefetch_related(_species_pf).order_by("id")

    # Dropdown option lists for the filter form.
    deployments = Deployment.objects.filter(images__isnull=False).distinct().order_by("camera_id")
    species_list = (
        SpeciesClassification.objects.filter(source=SPECIESNET)
        .values_list("category", flat=True)
        .distinct()
        .order_by("category")
    )

    return render(
        request,
        "image/results.html",
        {"images": qs, "deployments": deployments,
         "species_list": species_list, "f": request.GET},
    )


def annotated_image(request, pk: int):
    """GET /results/<pk>/annotated.png — supervision-drawn boxes as PNG."""
    try:
        image = Image.objects.prefetch_related(_species_pf).get(pk=pk)
    except Image.DoesNotExist:
        raise Http404("No such image")
    detections = list(image.detections.all())
    try:
        png = annotate_image(image.path, detections, labels=species_labels(detections))
    except FileNotFoundError:
        raise Http404("Source image file missing on disk")
    return HttpResponse(png, content_type="image/png")