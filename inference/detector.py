"""MegaDetector wrapper: image arrays in, ParsedDetection lists out.

Deliberately knows nothing about files, progress bars or the database. Walking a
directory and writing rows is the caller's job (image/services.py), which is what
lets the video pipeline reuse this untouched.
"""

from __future__ import annotations

from inference.types import ParsedDetection

_CATEGORY_BY_ID = {
    0: "animal",
    1: "person",
    2: "vehicle",
}


class Detector:
    """Wraps one loaded MegaDetector variant. Use registry.get_detector() to cache it.

    Every variant we support has the same awkward shape: batch_image_detection()
    accepts only a *directory path*, while single_image_detection() accepts an
    array but omits the normalized_coords the parser needs. So the batch entry
    point here is a loop, and normalization happens in _detect_one.
    """

    def __init__(self, device: str = "cpu", version: str | None = None):
        # Imported lazily so the variant/licence policy stays in one place and
        # importing this module does not drag in torch.
        from inference.registry import resolve_detector_variant

        self.version, variant, loader = resolve_detector_variant(version)
        self.licence = variant.licence
        self.device = device
        self._model = loader(version=self.version, pretrained=True, device=device)

    def raw_batch(self, images, conf_threshold):
        """The unparsed per-image result dicts.

        Public so golden_value/try_models.py can show exactly what _parse_one is fed —
        that diagnostic is the whole reason parsing is quarantined in one place.
        """
        return [self._detect_one(image, conf_threshold) for image in images]

    def detect_batch(self, images, conf_threshold):
        """Run inference on image arrays; one ParsedDetection list per image.

        NOTE: this parses the CURRENTLY-observed PytorchWildlife return shape.
        Run golden_value/try_models.py first and adjust _parse_one if your installed
        version differs — that's exactly why parsing is quarantined here.
        """
        return [
            self._parse_one(raw, conf_threshold)
            for raw in self.raw_batch(images, conf_threshold)
        ]

    def _detect_one(self, image, conf_threshold):
        """One array through the model, with boxes normalized to 0-1.

        We do the arithmetic ourselves rather than reusing upstream's, which also
        dodges a bug: rtdetr_apache_base.batch_image_detection divides x by the
        image *height* and y by the *width* (it indexes a [w, h] tensor as if it
        were [h, w]), silently wrong for any non-square image.

        The model applies det_conf_thres itself with a strict >, so a box sitting
        exactly on the threshold never reaches _parse_one.
        """
        raw = self._model.single_image_detection(image, det_conf_thres=conf_threshold)
        height, width = image.shape[:2]
        raw["normalized_coords"] = [
            [
                _clamp(x1 / width), _clamp(y1 / height),
                _clamp(x2 / width), _clamp(y2 / height),
            ]
            for x1, y1, x2, y2 in raw["detections"].xyxy
        ]
        return raw
    
    def detect_tiled(self, image, tile_threshold, whole_threshold=None,
                      grid=(3,2), overlap=0.25, nms_iou=0.45):
        """The whole frame plus overlapping tiles, merged by IoU.

        Two thresholds because they answer different questions. The tiles are
        where a still animal becomes scoreable at all, so they carry the gate
        that is allowed to start a track. The whole frame runs lower and only
        keeps an existing track alive between tiled passes.
        """
        whole_threshold = tile_threshold if whole_threshold is None else whole_threshold
        height, width = image.shape[:2]
        found = list(self.detect_batch([image], whole_threshold)[0])

        cols, rows = grid
        tile_width = int(width / cols * (1 + overlap))
        tile_height = int(height / rows * (1 + overlap))
        for row in range(rows):
            for col in range(cols):
                left = min(int(col * width / cols), width - tile_width)
                top = min(int(row * height / rows), height - tile_height)
                tile = image[top:top+tile_height, left:left+tile_width]
                actual_height, actual_width = tile.shape[:2]
                for detection in self.detect_batch([tile], tile_threshold)[0]:
                    found.append(ParsedDetection(
                        category=detection.category, 
                        confidence=detection.confidence,
                        x1=(left + detection.x1 * actual_width) / width,
                        y1=(top + detection.y1 * actual_height) / height,
                        x2=(left + detection.x2 * actual_width) / width,
                        y2=(top + detection.y2 * actual_height) / height,
                    ))
                
        return _merge(found, nms_iou)

    def _parse_one(self, raw, conf_threshold):
        """Parse a single raw result into a ParsedDetection."""
        out: list[ParsedDetection] = []
        if not isinstance(raw, dict) or "detections" not in raw:
            raise ValueError(
                f"Unexpected shape from model result: {type(raw)}; "
                "inspect with golden_value/try_models.py and update _parse_one."
            )

        dets = raw["detections"]
        norm_coords = raw.get("normalized_coords")
        confs = getattr(dets, "confidence", None)
        class_ids = getattr(dets, "class_id", None)
        if norm_coords is None or confs is None or class_ids is None:
            raise ValueError(
                "Missing normalized_coords/confidence/class_id; "
                "inspect with golden_value/try_models.py and update _parse_one."
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


def _clamp(value):
    """Keep a normalized coordinate inside the frame.

    RT-DETR does not clip its boxes to the image, so it emits values a fraction of
    a pixel past the edge (0.00027 over, measured on example_images). Everything
    downstream — crop padding, drawing — is entitled to assume 0..1.
    """
    return min(max(float(value), 0.0), 1.0)

def _iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    if overlap <= 0.0:
        return 0.0
    area = lambda box: max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    union = area(a) + area(b) - overlap
    return overlap / union if union > 0.0 else 0.0

def _merge(detections, iou_threshold):
    """Overlapping tiles see the same animal twice; keep the most confident."""
    kept = []
    for detection in sorted(detections, key=lambda d: -d.confidence):
        if all(
            detection.category != other.category or
            _iou(detection.bbox, other.bbox) < iou_threshold
            for other in kept
        ):
            kept.append(detection)
    return kept

