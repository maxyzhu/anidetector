"""
python manage.py ingest <folder> 

- scan -> detect -> filter -> persist

Pipeline entry point:
1. Scan the folder for images
2. Detect objects in the images
3. Filter out low-confidence detections
4. Persist the detections to the database
"""

from __future__ import annotations

import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from detections.models import Detection, Image
from inference import IMAGE_SUFFIXES

# EXIF tags ids.
_EXIF_DATETIME = 306          # DateTime (top-level IFD)
_EXIF_OFFSET = 0x8769         # pointer to the Exif sub-IFD
_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_DATETIME_DIGITIZED = 36868

def _read_captured_at(path):
    """Read EXIF capture time as an aware datetime, or None if unavailable.

    Prefers DateTimeOriginal (when the shutter fired) from the Exif sub-IFD,
    then DateTimeDigitized, then the top-level DateTime. EXIF stores naive local
    time ("YYYY:MM:DD HH:MM:SS"); we attach the project's TIME_ZONE.
    """
    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as img:
            exif = img.getexif()
            raw = exif.get(_EXIF_DATETIME)
            try:
                sub = exif.get_ifd(_EXIF_OFFSET)
                raw = sub.get(_EXIF_DATETIME_ORIGINAL) or sub.get(
                    _EXIF_DATETIME_DIGITIZED
                ) or raw
            except Exception:
                pass
        if not raw:
            return None
        naive = datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
        return timezone.make_aware(naive)
    except Exception:
        # Missing/broken EXIF is not fatal — the image is still ingested.
        return None
        

class Command(BaseCommand):
    help = "Ingest a folder of images: detect, filter blanks, persist results."

    def add_arguments(self, parser):
        parser.add_argument(
            "folder", 
            type=str, 
            help="Folder of images to ingest.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Images per model inference call (default 8).",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            help="Torch device: 'cpu', 'mps', 'cuda' (default 'cpu').",
        )
        parser.add_argument(
            "--conf",
            type=float,
            default=settings.DETECTION_CONFIDENCE_THRESHOLD,
            help="Confidence threshold for detections.",
        )
        parser.add_argument(
            "--retry-failed",
            action="store_true",
            help="Retry failed images.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Register at most this many new images (handy for profiling).",
        )
    
    def handle(self, *args, **opts):
        folder = Path(opts["folder"])
        if not folder.is_dir():
            self.stderr.write(f"Not a directory: {folder}")
            return
        
        batch_size = opts["batch_size"]
        conf = opts["conf"]
        limit = opts["limit"]

        # Phase 1: register files as pending image rows.
        discovered = 0
        created = 0
        for file in sorted(folder.rglob("*")):
            if file.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            discovered += 1

            resolved = file.resolve()
            if Image.objects.filter(path=str(resolved)).exists():
                continue
            Image.objects.create(
                path=str(resolved),
                camera_site=file.parent.name,
                captured_at=_read_captured_at(file),
            )
            created += 1
            if limit and created >= limit:
                break  # cap registration so profiling touches only N images

        self.stdout.write(f"Discovered {discovered} image(s), created {created} image(s).")

        # Phase 2: process in batches from pending images.
        statuses = [Image.Status.PENDING]
        if opts["retry_failed"]:
            statuses.append(Image.Status.FAILED)
        pending_qs = Image.objects.filter(status__in=statuses).order_by("id")
        total = pending_qs.count()
        if total == 0:
            self.stdout.write("No pending images to process, Done.")
            return
        
        self.stdout.write(f"Processing {total} pending images in batches of {batch_size}...")

        from inference import get_detector, load_image_array
        # Cached singleton: reused across batches and across repeated runs in-process.
        detector = get_detector(device=opts["device"])

        processed = 0
        failed = 0
        started = time.monotonic()

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_rows = pending_qs[batch_start:batch_end]

            arrays: list = []
            valid_rows: list[Image] = []
            for row in batch_rows:
                try:
                    arr, w, h = load_image_array(row.path)
                    row.width, row.height = w, h
                    arrays.append(arr)
                    valid_rows.append(row)
                except Exception as e:
                    row.status = Image.Status.FAILED
                    row.error = f"load: {e}"
                    row.save(update_fields=["status", "error"])
                    failed += 1
            
            if not arrays:
                continue
            
            try:
                batch_results = detector.detect_batch(arrays, conf_threshold=conf)
            except Exception as e:
                failed += len(arrays)
                for row in valid_rows:
                    row.status = Image.Status.FAILED
                    row.error = f"inference: {e}"
                Image.objects.bulk_update(valid_rows, fields=["status", "error"])
                continue
            
            detections_to_create: list[Detection] = []
            for row, dets in zip(valid_rows, batch_results):
                row.is_blank = len(dets) == 0
                row.status = Image.Status.PROCESSED
                for d in dets:
                    detections_to_create.append(
                        Detection(
                            image=row,
                            category=d.category,
                            confidence=d.confidence,
                            bbox_x1=d.x1,
                            bbox_y1=d.y1,
                            bbox_x2=d.x2,
                            bbox_y2=d.y2,
                        )
                    )
            
            with transaction.atomic():
                Image.objects.bulk_update(
                    valid_rows, ["width", "height", "status", "is_blank"]
                )
                if detections_to_create:
                    Detection.objects.bulk_create(detections_to_create)
            

            processed += len(valid_rows)

            elapsed = time.monotonic() - started
            rate = processed / elapsed * 60 if elapsed > 0 else 0
            # progress bar
            pct = (processed / total * 100) if total > 0 else 0
            bar_length = 20
            filled_length = int(bar_length * processed // total) if total > 0 else 0
            bar = "=" * filled_length + "-" * (bar_length - filled_length)
            self.stdout.write(
                f"\r|{bar}| {pct:.1f}% {processed}/{total} images processed "
                f"({failed} failed) {rate:.1f} images/min\x1b[K",
                ending="\n" if processed == total else ""
            )
            self.stdout.flush()
            
        self.stdout.write(
            self.style.SUCCESS(
                f"Done. processed={processed} failed={failed} "
                f"in {time.monotonic() - started:.1f}s."
            )
        )