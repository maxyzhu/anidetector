"""
python manage.py cluster

- group images into **events** by capture time.

A CLI shell around detections.services.cluster_images.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from detections.services import cluster_images


class Command(BaseCommand):
    help = "Group images into events by camera_site + capture-time gap."

    def add_arguments(self, parser):
        parser.add_argument(
            "--rebuild",
            action="store_true",
            help="Delete all existing events and re-cluster every image."
        )

    def handle(self, *args, **opts):
        result = cluster_images(rebuild=opts["rebuild"])

        if result.removed:
            self.stdout.write(f"Removed {result.removed} existing event row(s).")
        self.stdout.write(
            self.style.SUCCESS(
                f"Done. events={result.events} "
                f"(skipped {result.skipped} image(s) without capture time)."
            )
        )
