from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

from config import views
from image import template_views
from video import template_views as video_template_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("image.urls")),
    path("api/video/", include("video.urls")),
    # Template views
    path("", views.home_page, name="home"),
    path("results/", template_views.results_page, name="results"),
    path("results/<int:pk>/annotated.png", template_views.annotated_image, name="annotated-image"),
    path("video/", video_template_views.tracks_page, name="video-tracks"),
    path("video/tracks/<int:pk>/", video_template_views.track_detail_page,
         name="video-track-detail"),
]

# Track representative frames are written under MEDIA_ROOT; in production a real
# web server serves them.
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
