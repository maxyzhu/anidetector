"""
python manage.py rollup 

- aggregate per-event species into EventSpecies.

For each event, take the latest SpeciesNet classification of each of its
detections, group by species, and write one EventSpecies per species:
  confidence      = max over the group (most confident frame)
  detection_count = number of frames supporting that species
  representative_image = image of that most-confident crop
No low-support filtering — every species that appears is kept.
"""

from django.conf import settings
from django.core.management.base import BaseCommand

from detections.models import Event, EventSpecies, SpeciesClassification


class Command(BaseCommand):
    help = "Rollup detections into events and species."

    def add_arguments(self, parser):
        parser.add_argument("--event", type=int, default=None, help="Rollup a single event id.")

    def handle(self, *args, **opts):
        events = Event.objects.all()
        if opts["event"] is not None:
            events = events.filter(id=opts["event"])

        total_events = 0
        total_species = 0
        skipped = 0

        for event in events.iterator():
            # Latest SpeciesNet classification per detection in this event.
            rows = (
                SpeciesClassification.objects.filter(
                    source=SpeciesClassification.Source.SPECIESNET,
                    detection__image__event=event,
                )
                .select_related("detection")
                .order_by("detection_id", "-created_at")
            )

            latest_seen: set[int] = set()
            agg: dict[str, dict] = {} # {confidence, count, rep_image_id}
            for c in rows:
                if c.detection_id in latest_seen:
                    continue
                latest_seen.add(c.detection_id)

                conf = c.confidence or 0.0
                bucket = agg.get(c.category)
                if bucket is None:
                    agg[c.category] = {
                        "conf": conf,
                        "count": 1,
                        "rep_image_id": c.detection.image_id,
                    }
                else:
                    bucket["count"] += 1
                    if conf > bucket["conf"]:
                        bucket["conf"] = conf
                        bucket["rep_image_id"] = c.detection.image_id
            
            if not agg:
                skipped += 1
                continue
            
            new_rows = [
                EventSpecies(
                    event=event,
                    category=cat,
                    confidence=v["conf"],
                    detection_count=v["count"],
                    representative_image_id=v["rep_image_id"],
                )
                for cat, v in agg.items()
            ]

            with transaction.atomic():
                EventSpecies.objects.filter(event=event).delete() # idempotent rebuild
                EventSpecies.objects.bulk_create(new_rows)

            total_events += 1
            total_species += len(new_rows)

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. events={total_events} species_rows={total_species} "
                f"(skipped {skipped} event(s) with no classifications)."
            )
        )