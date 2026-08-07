"""
python manage.py cluster 

- group images into **events** by capture time.

Images at the same camera_site are split into events whenever the gap between
consecutive captures exceeds EVENT_GAP_SECONDS. Images without EXIF time are
skipped (they can't be temporally clustered).
"""

from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand

from detections.models import Event, Image


class Command(BaseCommand):
    help = "Group images into events by camera_site + capture-time gap."

    def add_arguments(self, parser):
        parser.add_argument(
            "--rebuild",
            action="store_true",
            help="Delete all existing events and re-cluster every image."
        )
    
    def handle(self, *args, **opts):
        gap = timedelta(seconds=setting.EVENT_GAP_SECONDS)

        if opts["rebuild"]:
            deleted, _ = Event.objects.all().delete()
            self.stdout.write(f"Removed {deleted} existing event row(s).")
        
        qs = Image.objects.filter(captured_at__isnull=False) # only process images with EXIF time
        if not opts["rebuild"]:
            qs = qs.filter(event__isnull=True) # only process images without an event
        qs = qs.order_by("camera_site", "captured_at")

        events = 0
        bucket: list[Image] = []
        site = None
        prev = None

        def flush():
            nonlocal events
            if not bucket:
                return
            event = Event.objects.create(
                camera_site=site,
                start_time=bucket[0].captured_at,
                end_time=bucket[-1].captured_at,
                image_count=len(bucket),
            )
            Image.objects.filter(pk__in=[img.pk for img in bucket]).update(event=event)
            events += 1
        
        for img in qs.iterator():
            new_group = (
                img.camera_site != site
                or prev is None
                or img.captured_at - prev.captured_at > gap # core logic for clustering
            )
            if new_group:
                flush()
                bucket, site = [], img.camera_site
            bucket.append(img)
            prev = img
        flush()

        skipped = Image.objects.filter(captured_at__isnull=True).count()
        self.stdout.write(
            self.style.SUCCESS(
                f"Done. events={events} (skipped {skipped} image(s) without capture time)."
            )
        )