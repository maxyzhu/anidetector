"""
Thin wrappers around `services.process_media()` as Celery Tasks.
"""

from __future__ import annotations

from celery import shared_task

from core.models import Media
from video import services


@shared_task
def process_media_task(media_id, device=None):
    media = Media.objects.get(pk=media_id)
    last = None
    for last in services.process_media(media, device=device):
        pass
    if last is None:
        return None
    return {"media_id": media.id, "frames":last.frames, "tracks": last.tracks}


@shared_task
def enqueue_pending_video_media(device=None, limit=None):
    """Fan out 1 task per pending video media."""
    ids = services.pending_video_media().values_list('id', flat=True)
    if limit:
        ids = ids[:limit]
    ids = list(ids)
    for media_id in ids:
        process_media_task.delay(media_id, device=device)

    return len(ids)
