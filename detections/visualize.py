"""Render detection boxes onto an image using the `supervision` library.
"""

from __future__ import annotations

import io

import numpy as np
import supervision as sv
from PIL import Image as PILImage

# Per-category colors (BGR-format; supervision takes a ColorPalette).
_CATEGORY_ORDER = ["animal", "person", "vehicle"]
_PALETTE = sv.ColorPalette(
    colors=[
        sv.Color(r=94, g=197, b=34),   # animal — green
        sv.Color(r=68, g=68, b=239),   # person — red
        sv.Color(r=246, g=130, b=59),  # vehicle — blue
    ]
)


def annotate_image(image_path: str, detections) -> bytes:
    """Draw boxes + labels for `detections` on the image at `image_path`.

    `detections` is any iterable of objects exposing category, confidence and
    normalized bbox fields (bbox_x1..bbox_y2) — i.e. our Detection model rows.
    Returns PNG bytes, ready to be written to an HttpResponse.

    WHY convert normalized coords back to pixels here: we stored boxes as
    [0,1] fractions (resolution-independent), but supervision draws in pixel
    space, so we scale by the actual loaded image size at render time.
    """
    pil = PILImage.open(image_path).convert("RGB")
    scene = np.array(pil)
    h, w = scene.shape[0], scene.shape[1]

    dets = list(detections)
    if not dets:
        # Nothing to draw (blank frame) — return the original image as PNG.
        return _to_png_bytes(scene)

    xyxy = np.array(
        [
            [d.bbox_x1 * w, d.bbox_y1 * h, d.bbox_x2 * w, d.bbox_y2 * h]
            for d in dets
        ],
        dtype=float,
    )
    class_ids = np.array([_CATEGORY_ORDER.index(d.category) for d in dets], dtype=int)
    confidences = np.array([d.confidence for d in dets], dtype=float)

    sv_detections = sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)
    
    labels = [f"{d.category} {d.confidence:.2f}" for d in dets]

    box_annotator = sv.BoxAnnotator(color=_PALETTE, color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(color=_PALETTE, color_lookup=sv.ColorLookup.CLASS)

    annotated = box_annotator.annotate(scene=scene.copy(), detections=sv_detections)
    annotated = label_annotator.annotate(
        scene=annotated, detections=sv_detections, labels=labels
    )
    return _to_png_bytes(annotated)


def _to_png_bytes(scene: np.ndarray) -> bytes:
    buf = io.BytesIO()
    PILImage.fromarray(scene).save(buf, format="PNG")
    return buf.getvalue()