"""Server-rendered results page + annotated-image endpoint.

Two views:
  - results_page: an HTML gallery of processed images (Django template)
  - annotated_image: streams a supervision-annotated PNG for one image
"""

from __future__ import annotations

from django.db.models import Prefetch
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, render

from .models import Event, EventSpecies, Image, SpeciesClassification
from .visualize import annotate_image

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
      ?site=<camera_site>    images from that camera site
      ?species=<label>       images whose SpeciesNet classification == label
      ?min_species_conf=0.5  images with a SpeciesNet result >= this confidence
      ?after=YYYY-MM-DD      captured on/after this date
      ?before=YYYY-MM-DD     captured on/before this date
    """
    SPECIESNET = SpeciesClassification.Source.SPECIESNET
    qs = Image.objects.filter(status=Image.Status.PROCESSED)

    if request.GET.get("show") != "all":
        # Default view hides blanks — the whole point of the blank filter.
        qs = qs.filter(is_blank=False)

    category = request.GET.get("category")
    if category:
        qs = qs.filter(detections__category=category)

    site = request.GET.get("site")
    if site:
        qs = qs.filter(camera_site=site)

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
    qs = qs.distinct().prefetch_related(_species_pf).order_by("id")

    # Dropdown option lists for the filter form.
    sites = (
        Image.objects.exclude(camera_site="")
        .values_list("camera_site", flat=True)
        .distinct()
        .order_by("camera_site")
    )
    species_list = (
        SpeciesClassification.objects.filter(source=SPECIESNET)
        .values_list("category", flat=True)
        .distinct()
        .order_by("category")
    )

    return render(
        request,
        "detections/results.html",
        {"images": qs, "sites": sites, "species_list": species_list, "f": request.GET},
    )


def events_page(request):
    """GET /events/ — gallery of events with their aggregated species."""
    # Species ordered by confidence so `species.all|first` is the top one (its
    # representative image becomes the card thumbnail).
    species_pf = Prefetch(
        "species",
        queryset=EventSpecies.objects.select_related("representative_image").order_by(
            "-confidence"
        ),
    )
    events = Event.objects.prefetch_related(species_pf).order_by("-start_time")
    return render(request, "detections/events.html", {"events": events})


def event_detail(request, pk: int):
    """GET /events/<pk>/ — one event: its species rollup + every image in it."""
    species_pf = Prefetch(
        "species",
        queryset=EventSpecies.objects.select_related("representative_image").order_by(
            "-confidence"
        ),
    )
    images_pf = Prefetch(
        "images",
        queryset=Image.objects.prefetch_related(_species_pf).order_by("captured_at", "id"),
    )
    event = get_object_or_404(
        Event.objects.prefetch_related(species_pf, images_pf), pk=pk
    )
    return render(request, "detections/event_detail.html", {"event": event})


def annotated_image(request, pk: int):
    """GET /results/<pk>/annotated.png — supervision-drawn boxes as PNG."""
    try:
        image = Image.objects.prefetch_related("detections").get(pk=pk)
    except Image.DoesNotExist:
        raise Http404("No such image")
    try:
        png = annotate_image(image.path, image.detections.all())
    except FileNotFoundError:
        raise Http404("Source image file missing on disk")
    return HttpResponse(png, content_type="image/png")