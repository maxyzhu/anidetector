"""Server-rendered results page + annotated-image endpoint.

Two views:
  - results_page: an HTML gallery of processed images (Django template)
  - annotated_image: streams a supervision-annotated PNG for one image
"""

from __future__ import annotations

from dataclasses import dataclass

from django.db.models import Prefetch
from django.http import Http404, HttpResponse
from django.shortcuts import render

from core.models import Deployment, Image, SpeciesClassification
from core.taxonomy import parse_taxon_path
from core.visualize import annotate_image, species_labels

# Prefetch each detection's SpeciesNet classifications, newest first, so only
# the latest is read and without an extra query per detection.
_species_pf = Prefetch(
    "detections__classifications",
    queryset=SpeciesClassification.objects.filter(
        source=SpeciesClassification.Source.SPECIESNET
    ).order_by("-created_at"),
)


def _common_name(label):
    """The stored label is SpeciesNet's whole taxonomy path. The API keeps it; a
    page has room for the common name. Human annotations are free text and do not
    parse, so they fall through unchanged."""
    if not label:
        return None
    taxon = parse_taxon_path(label)
    return taxon.common_name if taxon else label


@dataclass
class Box:
    """One detection as the gallery shows it: the MegaDetector category, plus
    the newest species result already reduced to what fits on a pill."""
    category: str
    confidence: float
    species: str | None
    species_label: str | None      # the whole taxonomy path, for the tooltip
    species_confidence: float | None


@dataclass
class ImageCard:
    image: Image
    boxes: list


def _cards(images):
    """Resolve the species labels here rather than in the template, so the
    template holds no parsing and the raw label is still available for `title`."""
    cards = []
    for image in images:
        boxes = []
        for detection in image.detections.all():
            newest = next(iter(detection.classifications.all()), None)
            boxes.append(Box(
                category=detection.category,
                confidence=detection.confidence,
                species=_common_name(newest.category) if newest else None,
                species_label=newest.category if newest else None,
                species_confidence=newest.confidence if newest else None,
            ))
        cards.append(ImageCard(image=image, boxes=boxes))
    return cards


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
    # The option value stays the stored label — that is what the filter matches
    # on — while the text shown is the common name. Sorted by that rather than by
    # the label, whose leading field is a uuid.
    species_list = sorted(
        (
            {"value": label, "name": _common_name(label)}
            for label in SpeciesClassification.objects.filter(source=SPECIESNET)
            .values_list("category", flat=True)
            .distinct()
        ),
        key=lambda option: option["name"],
    )

    return render(
        request,
        "image/results.html",
        {"cards": _cards(qs), "deployments": deployments,
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