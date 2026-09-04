from django.contrib import admin

from core.models import Deployment, Detection, Image, Media


@admin.register(Deployment)
class DeploymentAdmin(admin.ModelAdmin):
    list_display = ("camera_id", "location", "country", "modality", "start_ts")
    search_fields = ("camera_id", "location")


@admin.register(Media)
class MediaAdmin(admin.ModelAdmin):
    list_display = ("path", "kind", "status", "duration", "fps", "deployment")
    list_filter = ("kind", "status")
    search_fields = ("path",)
    readonly_fields = ("width", "height", "fps", "duration", "gop_size")


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ("id", "path", "uploaded_at", "status", "is_blank", "deployment")
    list_filter = ("status", "is_blank")
    search_fields = ("path",)


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ("id", "image", "category", "confidence")
    list_filter = ("category", "status")