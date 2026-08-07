from django.db.models import Prefetch
from rest_framework import generics

from .models import Image, Event, EventSpecies, SpeciesClassification
from .serializers import ImageSerializer, EventSerializer


_species_pf = Prefetch(
    "detections__classifications",
    queryset=SpeciesClassification.objects
        .filter(source=SpeciesClassification.Source.SPECIESNET)
        .order_by("-created_at"),
)

_event_species_pf = Prefetch(
    "species",
    queryset=EventSpecies.objects.select_related("representative_image"),
)

class ImageListView(generics.ListAPIView):
    """GET /api/images/ - list all images and their detections"""
    queryset = Image.objects.prefetch_related(_species_pf).all()
    serializer_class = ImageSerializer


class ImageFilteredListView(generics.ListAPIView):
    """GET /api/images/filtered/ - list all non-blank images and their detections"""
    queryset = Image.objects.prefetch_related(_species_pf).filter(is_blank=False)
    serializer_class = ImageSerializer


class ImageDetailView(generics.RetrieveAPIView):
    """GET /api/images/<int:pk>/ - get an image and its detections"""
    queryset = Image.objects.prefetch_related(_species_pf).all()
    serializer_class = ImageSerializer


class EventListView(generics.ListAPIView):
    """GET /api/events/ - list all events and their species"""
    queryset = Event.objects.prefetch_related(_event_species_pf).all()
    serializer_class = EventSerializer


class EventDetailView(generics.RetrieveAPIView):
    """GET /api/events/<int:pk>/ - get an event and its species"""
    queryset = Event.objects.prefetch_related(_event_species_pf).all()
    serializer_class = EventSerializer