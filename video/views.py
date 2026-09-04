from dataclasses import asdict
from pathlib import Path

from django.conf import settings
from django.http import Http404, HttpResponse
from rest_framework.generics import get_object_or_404
from rest_framework.response import Response
from rest_framework.views import APIView

from video.models import Track
from video.services import (
    Thresholds, labels_by_track, behaviour_images, track_timeline,
)

_DEFAULTS = {
    "displacement_enter_threshold": 0.5, "displacement_exit_threshold": 0.25,
    "deformation_enter_threshold": 0.5, "deformation_exit_threshold": 0.25,
    "min_duration": 0.0, "max_gap": 0.0,
}


def _thresholds_from(request):
    """Thresholds are query parameters, matching the activity_report command:
    they are what you are asking, not a property of the track."""
    return Thresholds(**{
        name: float(request.query_params.get(name, default))
        for name, default in _DEFAULTS.items()
    })


def _representative_url(request, track):
    """The representative frame is written under MEDIA_ROOT; Image.path is absolute."""
    if track.rep_detection_id is None:
        return None
    relative = Path(track.rep_detection.image.path).relative_to(settings.MEDIA_ROOT)
    return request.build_absolute_uri(f"{settings.MEDIA_URL}{relative}")


class TrackDetailView(APIView):
    def get(self, request, pk):
        track = get_object_or_404(
            Track.objects.select_related("media", "rep_detection__image"), pk=pk
        )
        timeline = track_timeline(track, _thresholds_from(request))
        return Response({
            "track_id": track.id,
            "species_label": labels_by_track([track])[track.id],
            "start_ts": track.start_ts,
            "end_ts": track.end_ts,
            "representative_image": _representative_url(request, track),
            "timeline": asdict(timeline),
            "behaviours": [
                {
                    "active": active,
                    "image": request.build_absolute_uri(
                        f"/api/video/tracks/{track.id}/behaviour/"
                        f"{'active' if active else 'rest'}.jpg"
                        f"?{request.query_params.urlencode()}"
                    ),
                }
                for active in (True, False)
            ],
        })


class BehaviourImageView(APIView):
    def get(self, request, pk, behaviour):
        if behaviour not in ("active", "rest"):
            raise Http404
        track = get_object_or_404(Track.objects.select_related("media"), pk=pk)
        payload = behaviour_images(track, _thresholds_from(request))[behaviour == "active"]
        if payload is None:
            raise Http404
        return HttpResponse(payload, content_type="image/jpeg")