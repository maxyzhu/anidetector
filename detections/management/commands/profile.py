"""
python manage.py profile <dataset> - time each ingest-pipeline stage.

Runs the whole pipeline (ingest -> cluster -> classify -> rollup) against a
throwaway test database (auto-created, auto-destroyed) so it never touches your
dev/prod data. Model load times are measured separately from per-item throughput.

    uv run python manage.py profile ./example_images --limit 200 --device mps --json out.json

v1 is stage-level (one row per stage). Sub-stage spans (EXIF / inference / DB)
can be added later without changing this command's shape. Videos are ignored for
now (ingest only scans image files) — a future `video_ingest` stage slots in here.
"""

from __future__ import annotations

import datetime
import io
import json
import time
from pathlib import Path

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = "Profile each ingest-pipeline stage on a throwaway test database."

    def add_arguments(self, parser):
        parser.add_argument("dataset", type=str, help="Folder of images to profile against.")
        parser.add_argument(
            "--limit", type=int, default=None,
            help="Profile only the first N images (recommended for large datasets).",
        )
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument(
            "--device", type=str, default=None,
            help="Torch device: 'cpu', 'mps', 'cuda' (default: cpu for detector, auto for SpeciesNet).",
        )
        parser.add_argument(
            "--keep-db", action="store_true",
            help="Keep the test database afterward instead of destroying it.",
        )
        parser.add_argument(
            "--json", dest="json_path", type=str, default=None,
            help="Also write machine-readable results to this path.",
        )

    def handle(self, *args, **opts):
        if not Path(opts["dataset"]).is_dir():
            self.stderr.write(f"Not a directory: {opts['dataset']}")
            return

        old_name = connection.settings_dict["NAME"]
        self.stdout.write("Creating throwaway test database (runs migrations) ...")
        connection.creation.create_test_db(verbosity=0, autoclobber=True)
        try:
            rows, meta = self._profile(opts)
            self._print_table(rows, meta)
            if opts["json_path"]:
                self._write_json(rows, meta, opts["json_path"])
        finally:
            if opts["keep_db"]:
                self.stdout.write(f"Kept test DB: {connection.settings_dict['NAME']}")
            else:
                connection.creation.destroy_test_db(old_name, verbosity=0)

    def _profile(self, opts):
        from detections.inference import get_detector
        from detections.models import (
            Detection,
            Event,
            EventSpecies,
            Image,
            SpeciesClassification,
        )
        from detections.tasks import get_classifier

        dataset = opts["dataset"]
        device = opts["device"]
        md_device = device or "cpu"
        batch_size = opts["batch_size"]
        limit = opts["limit"]
        sink = io.StringIO()  # swallow the sub-commands' own progress output
        rows: list[dict] = []

        def stage(name, fn, items_fn=None, unit=None):
            t0 = time.monotonic()
            fn()
            dt = time.monotonic() - t0
            rows.append(
                {"stage": name, "seconds": dt,
                 "items": items_fn() if items_fn else None, "unit": unit}
            )

        # Model loads are measured on their own so they don't skew throughput.
        stage("MegaDetector load", lambda: get_detector(device=md_device))
        stage(
            "ingest",
            lambda: call_command("ingest", dataset, batch_size=batch_size,
                                 device=md_device, limit=limit, stdout=sink),
            items_fn=Image.objects.count, unit="img/s",
        )
        stage(
            "cluster",
            lambda: call_command("cluster", stdout=sink),
            items_fn=Event.objects.count,
        )
        stage("SpeciesNet load", lambda: get_classifier(device=device))
        stage(
            "classify",
            lambda: call_command("classify_species", batch_size=batch_size,
                                 device=device, stdout=sink),
            items_fn=SpeciesClassification.objects.count, unit="box/s",
        )
        stage(
            "rollup",
            lambda: call_command("rollup", stdout=sink),
            items_fn=EventSpecies.objects.count,
        )

        meta = {
            "dataset": dataset,
            "device": device or "auto",
            "batch_size": batch_size,
            "limit": limit,
            "model_version": settings.SPECIESNET_MODEL,
            "images": Image.objects.count(),
            "detections": Detection.objects.count(),
        }
        return rows, meta

    def _print_table(self, rows, meta):
        w = self.stdout.write
        w("")
        w(f"dataset={meta['dataset']}  device={meta['device']}  "
          f"batch={meta['batch_size']}  limit={meta['limit']}")
        w(f"model={meta['model_version']}  images={meta['images']}  "
          f"detections={meta['detections']}")
        line = "-" * 62
        w(line)
        w(f"{'stage':22}{'items':>8}{'time':>10}{'throughput':>16}")
        w(line)
        total = 0.0
        for r in rows:
            total += r["seconds"]
            items = "-" if r["items"] is None else str(r["items"])
            tput = ""
            if r["unit"] and r["items"] and r["seconds"] > 0:
                tput = f"{r['items'] / r['seconds']:.1f} {r['unit']}"
            w(f"{r['stage']:22}{items:>8}{r['seconds']:>9.2f}s{tput:>16}")
        w(line)
        w(f"{'total':22}{'':>8}{total:>9.2f}s")
        w("")

    def _write_json(self, rows, meta, path):
        payload = {
            "meta": {
                **meta,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            "stages": rows,
            "total_seconds": sum(r["seconds"] for r in rows),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        self.stdout.write(f"Wrote JSON: {path}")
