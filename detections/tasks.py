"""Celery entry points.

Thin wrappers, nothing else. The orchestration lives in detections/services.py so
the identical code runs from a management command with no broker in sight — which
is what makes it testable, and what lets the video queue reuse the same shape
later without importing anything from here.
"""

from __future__ import annotations

from celery import shared_task

from detections import services


@shared_task
def classify_pending_task(batch_size=None, device=None, limit=None):
    return services.classify_pending(batch_size=batch_size, device=device, limit=limit)
