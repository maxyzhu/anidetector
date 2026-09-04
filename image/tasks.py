"""Celery entry points."""

from __future__ import annotations

from celery import shared_task

from image import services


@shared_task
def classify_pending_task(batch_size=None, device=None, limit=None):
    return services.classify_pending(batch_size=batch_size, device=device, limit=limit)
