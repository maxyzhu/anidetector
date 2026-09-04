from django.db.models import Prefetch
from rest_framework import generics

from core.models import Image, SpeciesClassification
from .serializers import ImageSerializer


_species_pf = Prefetch(
    "detections__classifications",
    queryset=SpeciesClassification.objects
        .filter(source=SpeciesClassification.Source.SPECIESNET)
        .order_by("-created_at"),
)

# Video representative frames share the Image table; these endpoints are photos.
_ingested = Image.objects.filter(source=Image.Source.INGEST)


class ImageListView(generics.ListAPIView):
    """GET /api/images/ - list all images and their detections"""
    # Ordered because pagination over an unordered queryset can repeat or drop
    # rows between pages.
    queryset = _ingested.prefetch_related(_species_pf).order_by("id")
    serializer_class = ImageSerializer


class ImageFilteredListView(generics.ListAPIView):
    """GET /api/images/filtered/ - list all non-blank images and their detections"""
    queryset = (
        _ingested.prefetch_related(_species_pf)
        .filter(is_blank=False).order_by("id")
    )
    serializer_class = ImageSerializer


class ImageDetailView(generics.RetrieveAPIView):
    """GET /api/images/<int:pk>/ - get an image and its detections"""
    queryset = _ingested.prefetch_related(_species_pf)
    serializer_class = ImageSerializer