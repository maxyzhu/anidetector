from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

from image import template_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("image.urls")),
    path("api/video/", include("video.urls")),
    # Template views
    path("results/", template_views.results_page, name="results"),
    path("results/<int:pk>/annotated.png", template_views.annotated_image, name="annotated-image"),
]

# Track representative frames are written under MEDIA_ROOT; in production a real
# web server serves them.
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
