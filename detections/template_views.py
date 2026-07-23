"""Server-rendered results page + annotated-image endpoint.

Two views:
  - results_page: an HTML gallery of processed images (Django template)
  - annotated_image: streams a supervision-annotated PNG for one image
"""

from __future__ import annotations

from django.http import Http404, HttpResponse
from django.shortcuts import render

from .models import Image
from .visualize import annotate_image


def results_page(request):
    """GET /results/ — minimal gallery of processed, non-blank images.

    Query params:
      ?show=all      include blank frames too (default hides them)
      ?category=...  restrict to images having a detection of that category
    """
    qs = Image.objects.filter(status=Image.Status.PROCESSED)
    if request.GET.get("show") != "all":
        # Default view hides blanks — the whole point of the blank filter.
        qs = qs.filter(is_blank=False)
    category = request.GET.get("category")
    if category:
        qs = qs.filter(detections__category=category).distinct()

    qs = qs.prefetch_related("detections").order_by("id")
    return render(request, "detections/results.html", {"images": qs})


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