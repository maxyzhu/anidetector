from django.contrib import admin
from django.urls import path, include

from detections import template_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("detections.urls")),
    # Template views
    path("results/", template_views.results_page, name="results"),
    path("results/<int:pk>/annotated.png", template_views.annotated_image, name="annotated-image"),
]