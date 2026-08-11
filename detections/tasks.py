"""
Species classification stage (SpeciesNet).

The core logic is a plain callable (classify_pending) so it can run synchronously
from a management command or be enqueued via the Celery task wrapper below — the
body is identical either way, which keeps it easy to test without a broker.

Option A: classify EVERY qualifying animal box (not just the top box per image),
so multiple animals in one frame each get a result. The model call itself lives in
inference/classifier.py; what stays here is the ORM state machine — claiming
batches, writing rows, marking failures.
"""

from __future__ import annotations

import logging
import time

from celery import shared_task
from django.conf import settings
from django.db import transaction

from detections.models import Detection, SpeciesClassification
from inference import get_classifier

logger = logging.getLogger(__name__)


def _claim_batch(batch_size):
    """Atomically claim up to batch_size pending animal detections.

    Marks the rows PROCESSING inside a transaction so concurrent Celery workers
    (skip_locked) each grab a *different* batch and never double-classify the same
    box. of=("self",) locks only the detection rows, not the joined image rows.
    """
    with transaction.atomic():
        batch = list(
            Detection.objects.filter(
                status=Detection.Status.PENDING,
                category=Detection.Category.ANIMAL,
                confidence__gte=settings.SPECIES_CONF_THRESHOLD,
                image__is_blank=False,
            )
            .select_related("image")
            .select_for_update(skip_locked=True, of=("self",))
            .order_by("id")[:batch_size]
        )
        if batch:
            Detection.objects.filter(pk__in=[d.pk for d in batch]).update(
                status=Detection.Status.PROCESSING
            )
    return batch


def classify_pending(batch_size=None, device=None, limit=None):
    """Drain pending animal detections batch by batch. Returns (processed, failed)."""
    from PIL import Image as PILImage

    batch_size = batch_size or settings.SPECIES_BATCH_SIZE
    padding = settings.CROP_PADDING
    model_version = settings.SPECIESNET_MODEL

    # The model name is read from settings here, not inside inference/, so the
    # inference package stays importable without Django.
    clf = get_classifier(model_version, device=device)
    processed = failed = 0
    remaining = limit  # None == unlimited

    while remaining is None or remaining > 0:
        size = batch_size if remaining is None else min(batch_size, remaining)
        batch = _claim_batch(size)
        if not batch:
            break

        filepaths, imgs = [], []
        pil_cache: dict[str, object] = {}  # reuse one PIL open per image in the batch
        t0 = time.monotonic()

        for det in batch:
            path = det.image.path
            try:
                img = pil_cache.get(path)
                if img is None:
                    img = PILImage.open(path).convert("RGB")
                    pil_cache[path] = img
                bbox = (det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2)
                imgs.append(clf.preprocess(img, bbox, padding))
            except Exception as e:
                logger.warning("preprocess failed for det#%s: %s", det.pk, e)
                imgs.append(None)
            filepaths.append(path)

        batch_votes = clf.predict_batch(filepaths, imgs)

        new_rows, done_ids, fail_ids = [], [], []
        for det, votes in zip(batch, batch_votes):
            # No votes == the model returned nothing usable for this crop.
            if votes:
                new_rows.append(
                    SpeciesClassification(
                        detection=det,
                        source=SpeciesClassification.Source.SPECIESNET,
                        category=votes[0].label,
                        confidence=votes[0].score,
                        top_k=[
                            {"category": v.label, "confidence": v.score} for v in votes
                        ],
                        model_version=model_version,
                    )
                )
                done_ids.append(det.pk)
            else:
                fail_ids.append(det.pk)

        SpeciesClassification.objects.bulk_create(new_rows)
        Detection.objects.filter(pk__in=done_ids).update(status=Detection.Status.PROCESSED)
        Detection.objects.filter(pk__in=fail_ids).update(status=Detection.Status.FAILED)

        processed += len(done_ids)
        failed += len(fail_ids)
        if remaining is not None:
            remaining -= len(batch)

        dt = time.monotonic() - t0
        per = dt / len(batch) if batch else 0
        logger.info(
            "batch=%d: %.2fs (%.3fs/box) ok=%d fail=%d",
            len(batch), dt, per, len(done_ids), len(fail_ids),
        )

    logger.info("classify_pending done: processed=%d failed=%d", processed, failed)
    return processed, failed


@shared_task
def classify_pending_task(batch_size=None, device=None, limit=None):
    """Celery entry point — thin wrapper so the logic above stays broker-agnostic."""
    return classify_pending(batch_size=batch_size, device=device, limit=limit)
