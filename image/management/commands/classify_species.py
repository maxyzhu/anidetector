"""python manage.py classify_species - run SpeciesNet on pending animal detections.

A CLI shell around image.services; --async hands the same work to Celery.
"""

from __future__ import annotations

import logging

from django.conf import settings
from django.core.management.base import BaseCommand

from image import services


class Command(BaseCommand):
    help = "Classify pending animal detections with SpeciesNet."

    def add_arguments(self, parser):
        parser.add_argument("--batch-size", type=int, default=settings.SPECIES_BATCH_SIZE)
        parser.add_argument(
            "--device", type=str, default=None,
            help="Torch device: 'cpu', 'mps', 'cuda' (default: auto-detect).",
        )
        parser.add_argument(
            "--limit", type=int, default=None,
            help="Max detections to process this run.",
        )
        parser.add_argument(
            "--retry-failed", action="store_true",
            help="Requeue failed + stuck-processing detections (e.g. after a worker crash).",
        )
        parser.add_argument("--async", action="store_true", dest="use_async",
            help="Run the task asynchronously using Celery.",
        )

    def handle(self, *args, **opts):
        logging.basicConfig(level=logging.INFO, format="%(message)s")

        if opts["retry_failed"]:
            requeued = services.requeue_failed_detections()
            self.stdout.write(f"Requeued {requeued} failed/stuck detection(s) to pending.")

        if opts["use_async"]:
            from image.tasks import classify_pending_task

            classify_pending_task.delay(
                batch_size=opts["batch_size"],
                device=opts["device"],
                limit=opts["limit"],
            )
            self.stdout.write("Enqueued to Celery.")
            return

        processed, failed = services.classify_pending(
            batch_size=opts["batch_size"],
            device=opts["device"],
            limit=opts["limit"],
        )
        self.stdout.write(
            self.style.SUCCESS(f"Done. processed={processed} failed={failed}.")
        )
