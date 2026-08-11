"""MegaDetector wrapper: image arrays in, ParsedDetection lists out.

Deliberately knows nothing about files, progress bars or the database. Walking a
directory and writing rows is the caller's job (detections/services.py after
Step 2), which is what lets the video pipeline reuse this untouched.
"""

from __future__ import annotations

import numpy as np

from inference.types import ParsedDetection

_CATEGORY_BY_ID = {
    0: "animal",
    1: "person",
    2: "vehicle",
}


class Detector:
    """Wraps one loaded MegaDetector variant. Use registry.get_detector() to cache it."""

    def __init__(self, device: str = "cpu", version: str | None = None):
        # Imported lazily so the variant/licence policy stays in one place and
        # importing this module does not drag in torch.
        from inference.registry import resolve_detector_variant

        self.version, loader, self.licence = resolve_detector_variant(version)
        self.device = device
        self._model = loader(version=self.version, pretrained=True, device=device)

    def detect_batch(self, images, conf_threshold):
        """Run inference on a batch and return per-image parsed detections.

        NOTE: this parses the CURRENTLY-observed PytorchWildlife return shape.
        Run scripts/try_models.py first and adjust _parse_one if your installed
        version differs — that's exactly why parsing is quarantined here.
        """
        raw_results = self._model.batch_image_detection(images)
        return [
            self._parse_one(raw, conf_threshold) for raw in _as_list(raw_results, len(images))
        ]

    def _parse_one(self, raw, conf_threshold):
        """Parse a single raw result into a ParsedDetection."""
        out: list[ParsedDetection] = []
        if not isinstance(raw, dict) or "detections" not in raw:
            raise ValueError(
                f"Unexpected shape from model result: {type(raw)}; "
                "inspect with scripts/try_models.py and update _parse_one."
            )

        dets = raw["detections"]
        norm_coords = raw.get("normalized_coords")
        confs = getattr(dets, "confidence", None)
        class_ids = getattr(dets, "class_id", None)
        if norm_coords is None or confs is None or class_ids is None:
            raise ValueError(
                "Missing normalized_coords/confidence/class_id; "
                "inspect with scripts/try_models.py and update _parse_one."
            )

        for box, conf, cid in zip(norm_coords, confs, class_ids):
            conf = float(conf)
            if conf < conf_threshold:
                continue
            category = _CATEGORY_BY_ID.get(int(cid))
            if category is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            out.append(ParsedDetection(category, conf, x1, y1, x2, y2))

        return out


def _as_list(x, n):
    """Ensure x is a list of length n.

    NOTE: some MegaDetector versions return a list,
          some a single object for a 1-image batch.
    """
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if n == 1:
        return [x]
    return list(x)
