"""
python manage.py ingest <folder> --deployment <id>

- scan -> detect -> filter -> persist

A CLI shell: arguments in, progress out. The pipeline itself is
image.services.ingest_directory, so a Celery task can drive the identical
run and differ only in how it renders progress.
"""

from __future__ import annotations

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.models import Deployment
from image.services import Phase, ingest_directory

_BAR_WIDTH = 20


class Command(BaseCommand):
    help = "Ingest a folder of images: detect, filter blanks, persist results."

    def add_arguments(self, parser):
        parser.add_argument(
            "folder",
            type=str,
            help="Folder of images to ingest.",
        )
        parser.add_argument(
            "--deployment",
            type=int,
            required=True,
            help="Deployment id these images belong to.",
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
        try:
            deployment = Deployment.objects.get(pk=opts["deployment"])
        except Deployment.DoesNotExist:
            raise CommandError(
                f"No Deployment with id {opts['deployment']}. "
                f"Create one in the admin or the shell first."
            ) from None

        progress = ingest_directory(
            opts["folder"],
            deployment,
            batch_size=opts["batch_size"],
            device=opts["device"],
            conf=opts["conf"],
            retry_failed=opts["retry_failed"],
            limit=opts["limit"],
        )

        last_detect = None
        try:
            for event in progress:
                if event.phase is Phase.REGISTER:
                    self.stdout.write(
                        f"Discovered {event.total} image(s), "
                        f"created {event.done} image(s)."
                    )
                else:
                    self._write_bar(event)
                    last_detect = event
        except NotADirectoryError as exc:
            raise CommandError(f"Not a directory: {exc}") from None

        if last_detect is None:
            self.stdout.write("No pending images to process, Done.")
            return

        self.stdout.write("")  # close the in-place progress line
        self.stdout.write(
            self.style.SUCCESS(
                f"Done. processed={last_detect.done} failed={last_detect.failed} "
                f"in {last_detect.elapsed:.1f}s."
            )
        )

    def _write_bar(self, event):
        """Redraw the progress bar in place; the closing newline comes from handle()."""
        pct = (event.done / event.total * 100) if event.total else 0
        filled = int(_BAR_WIDTH * event.done // event.total) if event.total else 0
        bar = "=" * filled + "-" * (_BAR_WIDTH - filled)
        self.stdout.write(
            f"\r|{bar}| {pct:.1f}% {event.done}/{event.total} images processed "
            f"({event.failed} failed) {event.rate_per_minute:.1f} images/min\x1b[K",
            ending="",
        )
        self.stdout.flush()
