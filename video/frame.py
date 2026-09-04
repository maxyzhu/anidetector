"""Pick and extract the frame that represents a track or a behaviour.

Selection is a strategy rather than a scoring function: ranking by bbox area
needs no pixels, but ranking by sharpness has to decode candidates first, and a
scorer taking only a bbox could never be swapped for one.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from PIL import Image as PILImage

from video.decode import frames_at
from video.motion import crop


@dataclass(frozen=True)
class Candidate:
    ts: float
    bbox: tuple[float, float, float, float]


def _area(bbox):
    x1, y1, x2, y2 = bbox
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def select_by_area(candidates: list[Candidate], media, top_k=8):
    """Largest box wins: closest to the lens, so the friendliest crop for
    SpeciesNet. Touches no pixels, which is why it is the default."""
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: _area(candidate.bbox))


def select_by_area_and_sharpness(candidates: list[Candidate], media, top_k=8):
    """Prefilter on area, then re-rank the survivors on Laplacian variance.

    Two stages because sharpness needs pixels: ranking every sample this way
    would decode the whole track to choose one frame.
    """
    if not candidates:
        return None
    shortlist = sorted(candidates, key=lambda c: _area(c.bbox), reverse=True)[:top_k]
    by_ts = {candidate.ts: candidate for candidate in shortlist}

    best = None
    best_score = -1.0
    for frame in frames_at(media.path, list(by_ts), media.id):
        candidate = _nearest(by_ts, frame.timestamp)
        score = _sharpness(crop(frame.array, candidate.bbox)) * _area(candidate.bbox)
        if score > best_score:
            best, best_score = candidate, score
    return best or shortlist[0]


def _nearest(by_ts, timestamp):
    # frames_at lands on the first frame at or after the target, so match back.
    return by_ts[min(by_ts, key=lambda ts: abs(ts - timestamp))]


def _sharpness(patch):
    if patch.size == 0:
        return 0.0
    grey = np.asarray(PILImage.fromarray(patch).convert("L"), dtype=np.float32)
    # Laplacian variance: a blurred frame has little high-frequency energy.
    laplacian = (
        -4 * grey[1:-1, 1:-1]
        + grey[:-2, 1:-1] + grey[2:, 1:-1]
        + grey[1:-1, :-2] + grey[1:-1, 2:]
    )
    return float(laplacian.var()) if laplacian.size else 0.0


SELECTORS = {
    "area": select_by_area,
    "area_sharpness": select_by_area_and_sharpness,
}


def get_selector(name=None):
    name = name or settings.VIDEO_FRAME_SELECTOR
    try:
        return SELECTORS[name], name
    except KeyError:
        raise ImproperlyConfigured(f"unknown VIDEO_FRAME_SELECTOR {name!r}")


def candidates_from_signals(signals):
    return [
        Candidate(
            ts=signal.ts,
            bbox=(signal.bbox_x1, signal.bbox_y1, signal.bbox_x2, signal.bbox_y2),
        )
        for signal in signals
    ]


def _resize_to_width(array, max_width):
    image = PILImage.fromarray(array)
    if image.width <= max_width:
        return array
    height = round(image.height * max_width / image.width)
    return np.asarray(image.resize((max_width, height), PILImage.LANCZOS))


def encode_jpeg(array, max_width=None, quality=None):
    array = _resize_to_width(array, max_width or settings.VIDEO_REPRESENTATIVE_WIDTH)
    buffer = io.BytesIO()
    PILImage.fromarray(array).save(
        buffer, format="JPEG",
        quality=quality or settings.VIDEO_REPRESENTATIVE_QUALITY, optimize=True,
    )
    return buffer.getvalue()


def frame_at(media, timestamp):
    """One frame, or None past the end of the file."""
    return next(iter(frames_at(media.path, [timestamp], media.id)), None)


def _score_area(bbox, frame_array):
    return _area(bbox)


def _score_area_and_sharpness(bbox, frame_array):
    return _area(bbox) * _sharpness(crop(frame_array, bbox))


SCORERS = {"area": _score_area, "area_sharpness": _score_area_and_sharpness}


def get_scorer(name=None):
    """The online counterpart of get_selector, keyed by the same setting.

    A sharpness term is cheap here and expensive there: during the decode pass the
    pixels are already in hand, whereas picking from stored signals has to seek
    back into the file before it can look at any.
    """
    name = name or settings.VIDEO_FRAME_SELECTOR
    try:
        return SCORERS[name]
    except KeyError:
        raise ImproperlyConfigured(f"unknown VIDEO_FRAME_SELECTOR {name!r}")


class RepresentativePicker:
    """Best-so-far representative frame for one live track.

    Online because the frames stream past exactly once, which is what removes the
    second decode pass entirely. Downscales on improvement rather than keeping the
    source array: the pick only ever gets better, so what stays resident per live
    track is a display-sized frame instead of a 4K one.
    """

    def __init__(self, scorer=None, max_width=None):
        self.scorer = scorer or get_scorer()
        self.max_width = max_width or settings.VIDEO_REPRESENTATIVE_WIDTH
        self.ts = None
        self.bbox = None
        self.confidence = None
        self._score = 0.0
        self._frame = None

    def offer(self, ts, bbox, confidence, frame_array):
        """Keep this frame if it scores better than every frame so far."""
        score = self.scorer(bbox, frame_array)
        if score <= self._score:
            return False
        self._score = score
        self.ts = ts
        self.bbox = tuple(bbox)
        self.confidence = confidence
        self._frame = _resize_to_width(frame_array, self.max_width)
        return True

    @property
    def chosen(self):
        return self._frame is not None

    @property
    def size(self):
        """(width, height) of the kept frame, after the downscale."""
        height, width = self._frame.shape[:2]
        return width, height

    def encode(self):
        """JPEG bytes of the chosen frame, or None if nothing was ever offered."""
        return encode_jpeg(self._frame) if self.chosen else None