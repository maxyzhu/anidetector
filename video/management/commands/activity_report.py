"""python manage.py activity_report - bouts and durations from stored signals.

Every threshold is an argument, not a setting: they are query parameters, and
storing them would imply the verdict is baked into the output.
"""

from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from video.models import Track
from video.services import (
    THRESHOLD_DEFAULTS, Thresholds, activity_report, sensitivity_table,
)


class Command(BaseCommand):
    help = "Report activity bouts for stored tracks."

    def add_arguments(self, parser):
        parser.add_argument("--media", type=int, default=None, help="Limit to one Media id.")
        # Defaults come from services so the command, the API and the page cannot
        # answer differently; the reasoning for the numbers lives there.
        parser.add_argument("--displacement-enter", type=float,
                            default=THRESHOLD_DEFAULTS["displacement_enter_threshold"])
        parser.add_argument("--displacement-exit", type=float,
                            default=THRESHOLD_DEFAULTS["displacement_exit_threshold"])
        parser.add_argument("--deformation-enter", type=float,
                            default=THRESHOLD_DEFAULTS["deformation_enter_threshold"])
        parser.add_argument("--deformation-exit", type=float,
                            default=THRESHOLD_DEFAULTS["deformation_exit_threshold"])
        parser.add_argument("--min-duration", type=float,
                            default=THRESHOLD_DEFAULTS["min_duration"])
        parser.add_argument("--max-gap", type=float,
                            default=THRESHOLD_DEFAULTS["max_gap"])
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
        percentiles = [50, 70, 80, 90, 95, 99]
        min_durations = [0.0, 1.0, 2.0, 5.0]
        rows = sensitivity_table(tracks, thresholds, percentiles, min_durations)

        # displ/deform are what the percentile resolved to on this data; without
        # them the table cannot be reproduced against another deployment.
        self.stdout.write(
            f"{'pct':>5}{'displ':>8}{'deform':>8}{'min_dur':>9}"
            f"{'bouts':>7}{'active':>10}{'active%':>9}"
        )
        for row in rows:
            fraction = "" if row.active_fraction is None else f"{row.active_fraction*100:.1f}%"
            self.stdout.write(
                f"{row.percentile:>4.0f}%{row.displacement_enter:>8.3f}"
                f"{row.deformation_enter:>8.3f}{row.min_duration:>9.1f}"
                f"{row.bouts:>7}{row.active_seconds:>9.1f}s{fraction:>9}"
            )
        self.stdout.write("")
        self.stdout.write(self._verdict(rows))
        return [row.__dict__ | {"active_fraction": row.active_fraction} for row in rows]

    def _verdict(self, rows):
        """The number that decides whether any of this is usable."""
        active_by_percentile = {
            row.percentile: row.active_seconds for row in rows if row.min_duration == 0.0
        }
        if not active_by_percentile:
            return "no rows to compare"

        low, high = min(active_by_percentile.values()), max(active_by_percentile.values())
        span = f"p{min(active_by_percentile):.0f}-p{max(active_by_percentile):.0f}"
        if low > 0:
            return f"active time varies {high / low:.1f}x across {span}"
        # Reporting an infinite spread says nothing; where it dies says where the
        # usable range ends.
        dies_at = min(p for p, seconds in active_by_percentile.items() if seconds == 0)
        return (
            f"active time reaches 0.0s at p{dies_at:.0f}, so {span} has no finite "
            f"spread; compare below that percentile"
        )