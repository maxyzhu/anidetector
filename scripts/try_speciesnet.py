#!/usr/bin/env python3
"""
try_speciesnet.py — Standalone check for the SpeciesNet classifier (no database).

Mirrors the target roadmap pipeline:
    image -> (MegaDetector detection) -> filter boxes by custom confidence
          -> SpeciesNet classifier -> print species

Two detection sources:
  A) --detections-json : feed external detections (simulates reading existing
     MegaDetector records from Postgres).
  B) no flag           : run SpeciesNet's built-in detector on the fly
     (quick smoke test).

Usage:
    uv run python scripts/try_speciesnet.py example_images/01.webp --threshold 0.5
    uv run python scripts/try_speciesnet.py img1.jpg img2.jpg -t 0.6 --country USA
"""

import argparse
import json
import sys

from speciesnet import DEFAULT_MODEL, SpeciesNet


def build_instances_dict(paths, country=None, admin1=None):
    """Wrap a list of image paths into SpeciesNet's instances_dict."""
    instances = []
    for p in paths:
        inst = {"filepath": p}
        if country:
            inst["country"] = country
        if admin1:
            inst["admin1_region"] = admin1
        instances.append(inst)
    return {"instances": instances}


def detections_from_megadetector_rows(rows):
    """
    Mapping example for production wiring (NOT called by this POC; shows the format).
    rows: Detection records queried from Postgres, each with at least
          filepath, bbox_xyxy (absolute pixels), img_w, img_h, confidence, category.
    Returns SpeciesNet's detections_dict (bbox normalized to [xmin, ymin, w, h]).
    """
    by_file = {}
    for r in rows:
        x1, y1, x2, y2 = r["bbox_xyxy"]
        w, h = r["img_w"], r["img_h"]
        det = {
            "category": str(r.get("category", "1")),   # "1" = animal
            "label": r.get("label", "animal"),
            "conf": float(r["confidence"]),
            "bbox": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
        }
        by_file.setdefault(r["filepath"], []).append(det)
    return {
        "predictions": [
            {"filepath": fp, "detections": dets} for fp, dets in by_file.items()
        ]
    }


def get_detections(model, instances_dict, detections_json):
    """Source A: read external json. Source B: run SpeciesNet's own detector live."""
    if detections_json:
        with open(detections_json) as f:
            return json.load(f)
    print(">> No --detections-json given; running SpeciesNet's built-in detector ...")
    return model.detect(instances_dict=instances_dict, progress_bars=True)


def filter_by_confidence(detections_dict, threshold):
    """Core step: drop boxes below the custom confidence (the 'filter MD results')."""
    kept, dropped = 0, 0
    out = []
    for pred in detections_dict.get("predictions", []):
        good = [d for d in pred.get("detections", []) if d.get("conf", 0) >= threshold]
        dropped += len(pred.get("detections", [])) - len(good)
        kept += len(good)
        out.append({"filepath": pred["filepath"], "detections": good})
    print(f">> threshold {threshold}: kept {kept} boxes, dropped {dropped} boxes")
    return {"predictions": out}


def print_results(predictions_dict):
    for pred in predictions_dict.get("predictions", []):
        print(f"\n=== {pred['filepath']} ===")
        clf = pred.get("classifications")
        if not clf or not clf.get("classes"):
            print("  (no classification)")
            continue
        for cls, score in zip(clf["classes"][:5], clf["scores"][:5]):
            print(f"  {score:0.3f}  {cls}")


def main():
    ap = argparse.ArgumentParser(description="SpeciesNet classifier wiring POC")
    ap.add_argument("images", nargs="+", help="image paths")
    ap.add_argument("-t", "--threshold", type=float, default=0.2,
                    help="custom detection confidence threshold; boxes below this "
                         "are not classified (default 0.2)")
    ap.add_argument("--detections-json", help="external MegaDetector detections json (optional)")
    ap.add_argument("--country", help="ISO country code (e.g. USA) for geo prior (optional)")
    ap.add_argument("--admin1", help="first-level admin region (e.g. CA), optional")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="model name (default: crop-based v4)")
    args = ap.parse_args()

    print(f">> Loading model: {args.model}")
    model = SpeciesNet(args.model, components="all")

    instances = build_instances_dict(args.images, args.country, args.admin1)
    detections = get_detections(model, instances, args.detections_json)
    detections = filter_by_confidence(detections, args.threshold)

    print(">> Running SpeciesNet classifier ...")
    predictions = model.classify(
        instances_dict=instances,
        detections_dict=detections,
        progress_bars=True,
    )
    print_results(predictions)


if __name__ == "__main__":
    main()