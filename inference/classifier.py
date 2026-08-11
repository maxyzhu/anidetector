"""SpeciesNet wrapper: crops in, SpeciesVote lists out.

Uses the low-level SpeciesNetClassifier rather than the high-level
SpeciesNet.classify(), because the latter only crops the highest-confidence box
per image and we want every qualifying animal box classified.

As with the detector, no database and no file walking here: the caller opens the
images (and gets to cache them across a batch) and owns the result rows.
"""

from __future__ import annotations

from inference.types import SpeciesVote

# Map our detection categories to SpeciesNet's numeric detection categories.
# SpeciesNet uses "1"=animal, "2"=human, "3"=vehicle.
_SPECIESNET_CATEGORY = {
    "animal": ("1", "animal"),
    "person": ("2", "human"),
    "vehicle": ("3", "vehicle"),
}


class Classifier:
    """Wraps one loaded SpeciesNet model. Use registry.get_classifier() to cache it."""

    def __init__(self, model, device=None):
        from speciesnet.classifier import SpeciesNetClassifier

        self.model_version = model
        self.device = device
        self._clf = SpeciesNetClassifier(model, device=device)

    def preprocess(self, image, bbox_xyxy, padding=1.0):
        """Crop one detection out of an open PIL image, ready for predict_batch.

        Returns whatever tensor shape SpeciesNet wants — opaque to us, and only
        ever handed straight back to predict_batch.
        """
        from speciesnet.utils import BBox

        bbox = BBox(*bbox_to_xywh_padded(*bbox_xyxy, padding))
        return self._clf.preprocess(image, bboxes=[bbox])

    def predict_batch(self, filepaths, preprocessed):
        """Classify a prepared batch; returns one SpeciesVote list per input.

        Votes come back in the model's own order (most confident first). An input
        the model could not classify yields an empty list, so the caller can keep
        its own rows aligned with the batch and mark just those as failed.
        """
        results = self._clf.batch_predict(filepaths, preprocessed)
        return [_parse_votes(res) for res in results]


def _parse_votes(result):
    """Pull the classification list out of one SpeciesNet result dict."""
    classifications = (result or {}).get("classifications") or {}
    classes = classifications.get("classes") or []
    scores = classifications.get("scores") or []
    return [SpeciesVote(label=c, score=float(s)) for c, s in zip(classes, scores)]


def bbox_to_xywh_padded(x1, y1, x2, y2, padding=1.0):
    """Convert a normalized xyxy box to SpeciesNet's normalized [xmin, ymin, w, h].

    Enlarges the box around its center by ``padding`` (1.1 == +10%) so the
    animal's edges are not cropped off, then clamps the result to stay inside
    the [0, 1] image bounds.
    """
    w = x2 - x1
    h = y2 - y1
    new_w = w * padding
    new_h = h * padding
    new_x = x1 - (new_w - w) / 2.0
    new_y = y1 - (new_h - h) / 2.0

    # Clamp origin into the image, then shrink size so the box never spills out.
    new_x = min(max(new_x, 0.0), 1.0)
    new_y = min(max(new_y, 0.0), 1.0)
    new_w = min(new_w, 1.0 - new_x)
    new_h = min(new_h, 1.0 - new_y)
    return [new_x, new_y, new_w, new_h]


def build_instances_dict(paths, country=None, admin1=None):
    """Wrap image filepaths into SpeciesNet's instances_dict."""
    instances = []
    for p in paths:
        inst = {"filepath": str(p)}
        if country:
            inst["country"] = country
        if admin1:
            inst["admin1_region"] = admin1
        instances.append(inst)
    return {"instances": instances}


def build_detections_dict(detections, padding=1.0):
    """Group detections into SpeciesNet's detections_dict, keyed by filepath.

    ``detections`` is any iterable of objects exposing ``image.path``,
    ``category``, ``confidence`` and ``bbox_x1/y1/x2/y2`` (normalized xyxy) —
    duck-typed so this stays usable without importing the ORM.
    Boxes are converted to padded [xmin, ymin, w, h].
    """
    by_file: dict[str, list[dict]] = {}
    for det in detections:
        code, label = _SPECIESNET_CATEGORY.get(det.category, ("1", "animal"))
        box = bbox_to_xywh_padded(
            det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2, padding
        )
        by_file.setdefault(det.image.path, []).append(
            {
                "category": code,
                "label": label,
                "conf": float(det.confidence),
                "bbox": box,
            }
        )
    return {
        "predictions": [
            {"filepath": fp, "detections": dets} for fp, dets in by_file.items()
        ]
    }
