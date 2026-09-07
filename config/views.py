"""The landing page: what is registered, and what to run next.

Lives in config rather than in an app because it is the one page that spans both
workflows — it counts video tracks and links to the stills gallery — and config
is already the composition root that knows about both. Putting it in core would
mean core referencing video's related_name, which tests/test_boundries.py cannot
catch and which is the first step to core knowing about everything.
"""

from pathlib import Path

from django.db.models import Count
from django.shortcuts import render

from core.models import Deployment, Detection, Image, Media


def _video_rows():
    videos = (
        Media.objects.filter(kind=Media.Kind.VIDEO)
        .select_related("deployment")
        .annotate(track_count=Count("tracks"))
        .order_by("-uploaded_at", "path")
    )
    return [
        {
            "name": Path(media.path).name,
            "folder": str(Path(media.path).parent),
            "status": media.status,
            "duration": media.duration,
            "tracks": media.track_count,
            "deployment": media.deployment,
            "error": media.error,
        }
        for media in videos
    ]


def home_page(request):
    """GET / — status, the commands to run next, and the registered videos."""
    deployment = Deployment.objects.order_by("id").first()
    by_status = dict(
        Media.objects.filter(kind=Media.Kind.VIDEO)
        .values("status")
        .annotate(total=Count("id"))
        .values_list("status", "total")
    )
    return render(
        request,
        "home.html",
        {
            "rows": _video_rows(),
            "by_status": by_status,
            "deployment": deployment,
            # The commands are copy-pasteable only if the id in them is real.
            "deployment_id": deployment.id if deployment else "<id>",
            "counts": {
                "deployments": Deployment.objects.count(),
                "videos": sum(by_status.values()),
                "images": Image.objects.filter(
                    source=Image.Source.INGEST).count(),
                "classified": Detection.objects.filter(
                    status=Detection.Status.PROCESSED).count(),
            },
        },
    )
