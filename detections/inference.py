"""
Inference pipeline for the MegaDetector model.
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage

_CATEGORY_BY_ID = {
    0: "person",
    1: "vehicle",
    2: "animal",
}


@dataclass
class ParsedDetection:
    """One box normalized and translated."""

    category: str # "person", "vehicle", "animal"
    confidence: float
    # normalized and translated to 0-1
    x1: float
    y1: float
    x2: float
    y2: float


class Detector:
    """Loads the model in singlton pattern."""

    def __init__(self, device: str = "cpu", version: str="MDV6-yolov9-c"):
        from PytorchWildlife.models import detection as pw_detection

        self._model = pw_detection.MegaDetector(
            version=version, pretrained=True, device=device
        )
    
    def detect_batch(self, images, conf_threshold):
        """Run inference on a batch and return per-image parsed detections.

        NOTE: this parses the CURRENTLY-observed PytorchWildlife return shape.
        Run scripts/try_model.py first and adjust _parse_one if your installed
        version differs — that's exactly why parsing is quarantined here.
        """
        raw_results = self._model.batch_image_detection(images)
        return [
            self._parse_one(raw, conf_threshold) for raw in _as_list(raw_results, len(images))
        ]
    
    def _parse_one(self, raw, conf_threshold):
        """Parse a single raw result into a ParsedDetection."""
        if not isinstance(raw, dict) or "detections" not in raw:
            raise ValueError(
                f"Unexpected shape from model result: {type(raw)}; "
                "inspect with scripts/try_model.py and update _parse_one."
            )
        
        dets = raw["detections"]
        norm_coords = raw.get("normalized_coords")
        confs = getattr(dets, "confidence". None)
        class_ids = getattr(dets, "class_id". None)
        if norm_coords is None or confs is None or class_ids is None:
            raise ValueError(
                "Missing normalized_coords/confidence/class_id; "
                "inspect with scripts/try_model.py and update _parse_one."
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

def load_image_array(path):
    """Explicitly read one image to an RGB array, returning (array, width, height)."""
    with PILImage.open(path) as img:
        return np.array(img.convert("RGB")), img.width, img.height


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