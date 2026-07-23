from rest_framework import generics

from .models import Image
from .serializers import ImageSerializer


class ImageListView(generics.ListAPIView):
    """GET /api/images/ - list all images and their detections"""
    queryset = Image.objects.prefetch_related("detections").all()
    serializer_class = ImageSerializer


class ImageFilteredListView(generics.ListAPIView):
    """GET /api/images/filtered/ - list all non-blank images and their detections"""
    queryset = Image.objects.prefetch_related("detections").filter(is_blank=False)
    serializer_class = ImageSerializer


class ImageDetailView(generics.RetrieveAPIView):
    """GET /api/images/<int:pk>/ - get an image and its detections"""
    queryset = Image.objects.prefetch_related("detections").all()
    serializer_class = ImageSerializer