"""Orchestration for the image workflow: scan, detect, classify.

This is where the pipeline actually lives. The management commands are argument
parsing and progress rendering around these functions, and the Celery tasks are
one-line wrappers around them, so the same code runs with or without a broker and
stays testable without either.

The split that matters is with `inference`: that package owns the models and
knows nothing about files or the database, while everything here is database
state — claiming work, writing rows, deciding what counts as done.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from PIL import Image as PILImage

from core.models import Detection, Image, SpeciesClassification
from inference import (
    ImageDirectorySource,
    get_classifier,
    get_detector,
    load_image_array,
)

logger = logging.getLogger(__name__)

# EXIF tag ids.
_EXIF_DATETIME = 306          # DateTime (top-level IFD)
_EXIF_OFFSET = 0x8769         # pointer to the Exif sub-IFD
_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_DATETIME_DIGITIZED = 36868


# --- ingest -----------------------------------------------------------------


class Phase(str, Enum):
    """Which half of an ingest run a progress event describes."""

    REGISTER = "register"
    DETECT = "detect"


@dataclass
class IngestProgress:
    """One observable step of an ingest run.

    ``phase`` says what the numbers mean: after REGISTER, ``total`` is how many
    image files were found and ``done`` how many of them were new; during DETECT
    they are the pending rows and how many have been through the detector.
    """

    phase: Phase
    done: int
    total: int
    failed: int
    elapsed: float

    @property
    def rate_per_minute(self):
        """Images per minute so far, or 0 before any time has passed."""
        return self.done / self.elapsed * 60 if self.elapsed > 0 else 0.0


def ingest_directory(folder, deployment, batch_size=8, device=None, conf=None,
                     retry_failed=False, limit=None):
    """Scan a folder, register new images, then run the detector over everything pending."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    # "auto" is resolved by inference when the model loads, not here.
    device = settings.TORCH_DEVICE if device is None else device
    conf = settings.DETECTION_CONFIDENCE_THRESHOLD if conf is None else conf

    started = time.monotonic()
    discovered, created = _register_new_images(folder, deployment, limit)
    yield IngestProgress(
        phase=Phase.REGISTER,
        done=created,
        total=discovered,
        failed=0,
        elapsed=time.monotonic() - started,
    )

    yield from _detect_pending(batch_size, device, conf, retry_failed)


def _register_new_images(folder, deployment, limit=None):
    """Create a pending Image row for every file not already known.

    Returns (discovered, created). ``limit`` caps the number of new rows, which
    keeps a profiling run to a known size.
    """
    discovered = created = 0
    for path in ImageDirectorySource(folder).paths():
        discovered += 1
        resolved = str(path.resolve())
        if Image.objects.filter(path=resolved).exists():
            continue
        Image.objects.create(
            deployment=deployment,
            path=resolved,
            captured_at=read_captured_at(path),
        )
        created += 1
        if limit and created >= limit:
            break
    return discovered, created


def _detect_pending(batch_size, device, conf, retry_failed):
    """Run the detector over pending image rows, yielding IngestProgress per batch.

    Walks by ascending id rather than by slice offset. Slicing the queryset would
    be wrong: rows leave it as they are marked processed, so a fixed offset skips
    the rows that shifted down behind it. It also terminates when --retry-failed
    keeps re-selecting a row that fails every time.
    """
    statuses = [Image.Status.PENDING]
    if retry_failed:
        statuses.append(Image.Status.FAILED)
    pending_qs = Image.objects.filter(status__in=statuses).order_by("id")

    total = pending_qs.count()
    if total == 0:
        return

    detector = get_detector(device=device)
    processed = failed = 0
    last_id = 0
    started = time.monotonic()

    while True:
        rows = list(pending_qs.filter(id__gt=last_id)[:batch_size])
        if not rows:
            break
        last_id = rows[-1].id

        arrays, loaded, unreadable = _load_batch(rows)
        failed += unreadable

        if arrays:
            try:
                results = detector.detect_batch(arrays, conf_threshold=conf)
            except Exception as exc:
                logger.warning("inference failed for a batch of %d: %s", len(arrays), exc)
                _mark_failed(loaded, f"inference: {exc}")
                failed += len(loaded)
            else:
                _persist_detections(loaded, results)
                processed += len(loaded)

        yield IngestProgress(
            phase=Phase.DETECT,
            done=processed,
            total=total,
            failed=failed,
            elapsed=time.monotonic() - started,
        )


def _load_batch(rows):
    """Read each row's pixels. Returns (arrays, rows that loaded, failure count).

    A file that will not open is marked FAILED immediately: it is a property of
    that one row, so it should not take the rest of the batch down with it.
    """
    arrays, loaded, failed = [], [], 0
    for row in rows:
        try:
            array, width, height = load_image_array(row.path)
        except Exception as exc:
            row.status = Image.Status.FAILED
            row.error = f"load: {exc}"
            row.save(update_fields=["status", "error"])
            failed += 1
            continue
        row.width, row.height = width, height
        arrays.append(array)
        loaded.append(row)
    return arrays, loaded, failed


def _persist_detections(rows, results):
    """Write one batch's boxes and flip its rows to processed, in one transaction."""
    to_create = []
    for row, detections in zip(rows, results):
        row.is_blank = len(detections) == 0
        row.status = Image.Status.PROCESSED
        to_create.extend(
            Detection(
                image=row,
                category=d.category,
                confidence=d.confidence,
                bbox_x1=d.x1,
                bbox_y1=d.y1,
                bbox_x2=d.x2,
                bbox_y2=d.y2,
            )
            for d in detections
        )

    with transaction.atomic():
        Image.objects.bulk_update(rows, ["width", "height", "status", "is_blank"])
        if to_create:
            Detection.objects.bulk_create(to_create)


def _mark_failed(rows, error):
    for row in rows:
        row.status = Image.Status.FAILED
        row.error = error
    Image.objects.bulk_update(rows, ["status", "error"])


def read_captured_at(path):
    """Read EXIF capture time as an aware datetime, or None if unavailable.

    Prefers DateTimeOriginal (when the shutter fired) from the Exif sub-IFD, then
    DateTimeDigitized, then the top-level DateTime. EXIF stores naive local time
    ("YYYY:MM:DD HH:MM:SS"); we attach the project's TIME_ZONE.

    The two except clauses are narrow on purpose. A blanket `except Exception`
    used to wrap this whole function, and it hid a missing import for long enough
    that captured_at was silently None for every image ever ingested.
    """
    try:
        with PILImage.open(path) as img:
            exif = img.getexif()
            raw = exif.get(_EXIF_DATETIME)
            try:
                sub = exif.get_ifd(_EXIF_OFFSET)
            except Exception:
                sub = {}
            raw = (
                sub.get(_EXIF_DATETIME_ORIGINAL)
                or sub.get(_EXIF_DATETIME_DIGITIZED)
                or raw
            )
    except OSError as exc:
        logger.debug("could not read EXIF from %s: %s", path, exc)
        return None

    if not raw:
        return None
    try:
        naive = datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        # Cameras do write malformed timestamps; not a reason to reject the image.
        logger.debug("unparseable EXIF timestamp %r in %s", raw, path)
        return None
    return timezone.make_aware(naive)


# --- species classification -------------------------------------------------


def requeue_failed_detections():
    """Put failed and stuck detections back in the queue. Returns how many.

    PROCESSING rows are included because a worker that died mid-batch leaves them
    claimed with nothing left running to finish them.
    """
    return Detection.objects.filter(
        status__in=[Detection.Status.FAILED, Detection.Status.PROCESSING],
        category=Detection.Category.ANIMAL,
    ).update(status=Detection.Status.PENDING)


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
    """Drain pending animal detections batch by batch. Returns (processed, failed).

    Classifies EVERY qualifying animal box rather than one per image, so several
    animals in one frame each get a result.
    """
    batch_size = batch_size or settings.SPECIES_BATCH_SIZE
    padding = settings.CROP_PADDING
    model_version = settings.SPECIESNET_MODEL
    device = settings.TORCH_DEVICE if device is None else device

    # The model name is read from settings here, not inside inference/, so that
    # package stays importable without Django.
    clf = get_classifier(model_version, device=device)
    processed = failed = 0
    remaining = limit  # None == unlimited

    while remaining is None or remaining > 0:
        size = batch_size if remaining is None else min(batch_size, remaining)
        batch = _claim_batch(size)
        if not batch:
            break

        filepaths, crops = [], []
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
                crops.append(clf.preprocess(img, bbox, padding))
            except Exception as exc:
                logger.warning("preprocess failed for det#%s: %s", det.pk, exc)
                crops.append(None)
            filepaths.append(path)

        batch_votes = clf.predict_batch(filepaths, crops)

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

