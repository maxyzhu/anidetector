from django.contrib import admin
from .models import Image, Detection


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ["id", "path", "uploaded_at", "status", "is_blank"]
    list_filter = ["status", "is_blank"]


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ["id", "image", "category", "confidence"]
    list_filter = ["category"]