"""python manage.py activity_report - bouts and durations from stored signals.

Every threshold is an argument, not a setting: they are query parameters, and
storing them would imply the verdict is baked into the output.
"""

from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from video.models import Track
from video.services import Thresholds, activity_report, sensitivity_table


class Command(BaseCommand):
    help = "Report activity bouts for stored tracks."

    def add_arguments(self, parser):
        parser.add_argument("--media", type=int, default=None, help="Limit to one Media id.")
        # Defaults sit above the measured noise floor: a motionless animal reads
        # 0.12 body lengths/s on average and 0.30 at the 95th percentile, so an
        # enter threshold below that would call detector jitter movement.
        parser.add_argument("--displacement-enter", type=float, default=0.5)
        parser.add_argument("--displacement-exit", type=float, default=0.25)
        parser.add_argument("--deformation-enter", type=float, default=0.5)
        parser.add_argument("--deformation-exit", type=float, default=0.25)
        parser.add_argument("--min-duration", type=float, default=0.0)
        parser.add_argument("--max-gap", type=float, default=0.0)
        parser.add_argument("--json", dest="json_path", default=None)
        parser.add_argument(
            "--sensitivity", action="store_true",
            help="Sweep the threshold instead of reporting one setting.",
        )

    def handle(self, *args, **opts):
        tracks = Track.objects.all().order_by("start_ts")
        if opts["media"] is not None:
            tracks = tracks.filter(media_id=opts["media"])
        tracks = list(tracks)
        if not tracks:
            raise CommandError("No tracks to report on.")
        
        thresholds = Thresholds(
            displacement_enter_threshold=opts["displacement_enter"],
            displacement_exit_threshold=opts["displacement_exit"],
            deformation_enter_threshold=opts["deformation_enter"],
            deformation_exit_threshold=opts["deformation_exit"],
            min_duration=opts["min_duration"],
            max_gap=opts["max_gap"],
        )
        if opts["sensitivity"]:
            payload = self._sensitivity(tracks, thresholds)
        else:
            payload = self._report(tracks, thresholds)

        if opts["json_path"]:
            with open(opts["json_path"], "w") as handle:
                json.dump(payload, handle, indent=2)
            self.stdout.write(f"Wrote {opts['json_path']}")

    def _report(self, tracks, thresholds):
        report = activity_report(tracks, thresholds)
        self.stdout.write(
            f"{'track':>7}{'species':>18}{'span':>9}{'active':>9}{'rest':>9}{'bouts':>7}"
        )
        for item in report:
            self.stdout.write(
                f"{item.track_id:>7}{(item.species_label or '-'):>18}"
                f"{item.end_ts - item.start_ts:>8.1f}s"
                f"{item.budget.active_seconds:>8.1f}s"
                f"{item.budget.rest_seconds:>8.1f}s"
                f"{item.budget.bout_count:>7}"
            )
        return [
            {
                "track_id": item.track_id,
                "species_label": item.species_label,
                "start_ts": item.start_ts,
                "end_ts": item.end_ts,
                "active_seconds": item.budget.active_seconds,
                "rest_seconds": item.budget.rest_seconds,
                "bout_count": item.budget.bout_count,
                "bouts": [
                    {"start_ts": b.start_ts, "end_ts": b.end_ts, "active": b.active}
                    for b in item.bouts
                ],
            }
            for item in report
        ]

    def _sensitivity(self, tracks, thresholds):
        enter_thresholds = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
        min_durations = [0.0, 1.0, 2.0, 5.0]
        rows = sensitivity_table(tracks, thresholds, enter_thresholds, min_durations)

        self.stdout.write(
            f"{'enter':>7}{'min_dur':>9}{'bouts':>8}{'active':>10}{'active%':>9}"
        )
        for row in rows:
            fraction = "" if row.active_fraction is None else f"{row.active_fraction*100:.1f}%"
            self.stdout.write(
                f"{row.enter_threshold:>7.2f}{row.min_duration:>9.1f}"
                f"{row.bouts:>8}{row.active_seconds:>9.1f}s{fraction:>9}"
            )
        # The number that decides whether any of this is usable.
        by_threshold = {}
        for row in rows:
            if row.min_duration == 0.0:
                by_threshold[row.enter_threshold] = row.active_seconds
        if by_threshold:
            low, high = min(by_threshold.values()), max(by_threshold.values())
            spread = high / low if low else float("inf")
            self.stdout.write("")
            self.stdout.write(
                f"active time varies {spread:.1f}x across thresholds "
                f"{min(by_threshold):.2f}-{max(by_threshold):.2f}"
            )
        return [row.__dict__ | {"active_fraction": row.active_fraction} for row in rows]