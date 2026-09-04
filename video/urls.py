from django.urls import path

from video import views

urlpatterns = [
    path("tracks/<int:pk>/", views.TrackDetailView.as_view(), name="track-detail"),
    path(
        "tracks/<int:pk>/behaviour/<str:behaviour>.jpg",
        views.BehaviourImageView.as_view(), name="track-behaviour-image",
    ),
]