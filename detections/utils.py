"""Pure helpers for the SpeciesNet classification stage.

These functions have no Django/model imports so they are easy to unit-test in
isolation. They translate our stored detections into the dict shapes SpeciesNet
expects and handle bbox padding.
"""

from __future__ import annotations

# Map our Detection.Category values to SpeciesNet's numeric detection categories.
# SpeciesNet uses "1"=animal, "2"=human, "3"=vehicle.
_SPECIESNET_CATEGORY = {
    "animal": ("1", "animal"),
    "person": ("2", "human"),
    "vehicle": ("3", "vehicle"),
}


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
    """Group Detection rows into SpeciesNet's detections_dict, keyed by filepath.

    ``detections`` is any iterable of objects exposing ``image.path``,
    ``category``, ``confidence`` and ``bbox_x1/y1/x2/y2`` (normalized xyxy).
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
