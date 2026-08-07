"""
Species classification stage (SpeciesNet).

Logic lives here as a plain callable so it runs synchronously today (Celery is
not wired up yet). When Celery/Redis land, wrap classify_pending() with
@shared_task and enqueue it — the body doesn't change.

Option A: classify EVERY qualifying animal box (not just the top box per image),
so multiple animals in one frame each get a result. This needs the low-level
SpeciesNetClassifier (preprocess a specific bbox + batch_predict), because the
high-level SpeciesNet.classify() only crops the highest-confidence box per image.
"""

from __future__ import annotations

import logging
import time

from django.conf import settings

from detections.models import Detection, SpeciesClassification
from detections.utils import bbox_to_xywh_padded

logger = logging.getLogger(__name__)

_classifier = None


def get_classifier(device=None):
    """Lazily load SpeciesNetClassifier once (weights load on first call)."""
    global _classifier
    if _classifier is None:
        from speciesnet.classifier import SpeciesNetClassifier

        _classifier = SpeciesNetClassifier(settings.SPECIESNET_MODEL, device=device)
    return _classifier


def _pending_qs():
    # NOTE: single-runner assumption. When Celery runs concurrent workers, claim
    # each batch with select_for_update(skip_locked=True) to avoid double work.
    return (
        Detection.objects.filter(
            status=Detection.Status.PENDING,
            category=Detection.Category.ANIMAL,
            confidence__gte=settings.SPECIES_CONF_THRESHOLD,
            image__is_blank=False,
        )
        .select_related("image")
        .order_by("id")
    )


def classify_pending(batch_size=None, device=None, limit=None):
    """Classify pending animal detections. Returns (processed, failed)."""
    from PIL import Image as PILImage
    from speciesnet.utils import BBox

    batch_size = batch_size or settings.SPECIES_BATCH_SIZE
    padding = settings.CROP_PADDING
    model_version = settings.SPECIESNET_MODEL

    qs = _pending_qs()
    dets = list(qs[:limit] if limit else qs)
    total = len(dets)
    if not total:
        logger.info("No pending detections to classify.")
        return 0, 0

    clf = get_classifier(device=device)
    processed = failed = 0

    for start in range(0, total, batch_size):
        chunk = dets[start : start + batch_size]
        filepaths, imgs = [], []
        pil_cache: dict[str, object] = {}  # reuse one PIL open per image in the chunk
        t0 = time.monotonic()

        for det in chunk:
            path = det.image.path
            try:
                img = pil_cache.get(path)
                if img is None:
                    img = PILImage.open(path).convert("RGB")
                    pil_cache[path] = img
                bbox = BBox(
                    *bbox_to_xywh_padded(
                        det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2, padding
                    )
                )
                imgs.append(clf.preprocess(img, bboxes=[bbox]))
            except Exception as e:
                logger.warning("preprocess failed for det#%s: %s", det.pk, e)
                imgs.append(None)
            filepaths.append(path)

        results = clf.batch_predict(filepaths, imgs)

        new_rows, done_ids, fail_ids = [], [], []
        for det, res in zip(chunk, results):
            clf_out = (res or {}).get("classifications") or {}
            classes = clf_out.get("classes") or []
            scores = clf_out.get("scores") or []
            if classes and scores:
                new_rows.append(
                    SpeciesClassification(
                        detection=det,
                        source=SpeciesClassification.Source.SPECIESNET,
                        category=classes[0],
                        confidence=float(scores[0]),
                        top_k=[
                            {"category": c, "confidence": float(s)}
                            for c, s in zip(classes, scores)
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
        dt = time.monotonic() - t0
        per = dt / len(chunk) if chunk else 0
        logger.info(
            "chunk %d-%d: %.2fs (%.3fs/box) ok=%d fail=%d",
            start, start + len(chunk), dt, per, len(done_ids), len(fail_ids),
        )

    logger.info("classify_pending done: processed=%d failed=%d", processed, failed)
    return processed, failed