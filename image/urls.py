from django.urls import path

from image import views

urlpatterns: list[path] = [
    path("images/", views.ImageListView.as_view(), name="image-list"),
    path("images/filtered/", views.ImageFilteredListView.as_view(), name="image-filtered-list"),
    path("images/<int:pk>/", views.ImageDetailView.as_view(), name="image-detail"),
]
