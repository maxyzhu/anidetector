"""Displacement and deformation signals for one track."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage


@dataclass
class MotionSample:
    ts: float
    displacement_bl_per_s: float
    deformation_score: float
    bbox: tuple[float, float, float, float]
    confidence: float


def crop(frame_array, bbox):
    """Pixels inside a normalized xyxy bbox, clipped to at least one pixel."""
    height, width = frame_array.shape[:2]
    x1, y1, x2, y2 = bbox
    left = min(max(int(x1 * width), 0), width - 1)
    top = min(max(int(y1 * height), 0), height - 1)
    right = min(max(int(round(x2 * width)), left + 1), width)
    bottom = min(max(int(round(y2 * height)), top + 1), height)
    return frame_array[top:bottom, left:right]


def _centre(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _diagonal(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return np.hypot((x2 - x1) * width, (y2 - y1) * height)


def _displacement_bl_per_s(pre_bbox, cur_bbox, delta_ts, frame_size):
    """Centre movement in body lengths per second.

    Body length is the bbox diagonal, which makes the reading scale invariant:
    the same species near and far gives the same number, removing most of the
    per-camera threshold tuning.

    Both boxes are scaled to pixels before anything is measured. Measuring in
    normalized space is wrong on a non-square frame — the same normalized step
    is a different real distance horizontally and vertically, so the error flips
    sign with direction and no constant calibrates it away.
    """
    if delta_ts <= 0:
        return 0.0
    width, height = frame_size

    pre_cx, pre_cy = _centre(pre_bbox)
    cur_cx, cur_cy = _centre(cur_bbox)
    moved = np.hypot((cur_cx - pre_cx) * width, (cur_cy - pre_cy) * height)

    # Mean of both diagonals, so an approaching animal's growing box does not
    # make the answer depend on which end you measured from.
    body_length = (
        _diagonal(pre_bbox, width, height) + _diagonal(cur_bbox, width, height)
    ) / 2.0
    if body_length <= 0:
        return 0.0
    return float(moved / body_length / delta_ts)


def _to_patch(crop_array, size):
    """Greyscale patch at a fixed size.

    The fixed size is what keeps deformation independent of displacement:
    without it an animal merely crossing the frame scores as change. Resizing
    before greyscaling costs ~3x less, since converting at full resolution
    produces pixels that are discarded on the next line.
    """
    if crop_array.size == 0:
        return np.zeros((size, size), dtype=np.float32)
    patch = PILImage.fromarray(crop_array).resize((size, size)).convert("L")
    return np.asarray(patch, dtype=np.float32)


def _deformation_score(pre_patch, cur_patch):
    """Mean absolute pixel change, 0..1.

    NOTE: a global illumination shift (IR illuminator flicker) also registers
    here. Worth revisiting once the pet test set shows whether it matters.
    """
    return float(np.abs(pre_patch - cur_patch).mean() / 255.0)


class MotionAccumulator:
    """Running motion measurement for one track.

    Stateful because both signals are differences against the previous
    observation and the frames stream past exactly once.
    """

    def __init__(self, frame_size, patch_size=32):
        self.frame_size = frame_size
        self.patch_size = patch_size
        self._pre_ts = None
        self._pre_bbox = None
        self._pre_patch = None

    def add(self, ts, bbox, confidence, frame_array):
        """Returns a MotionSample, or None for the first observation.

        None rather than a zero sample: a difference needs two points, and a
        fabricated zero would put a fake "resting" reading at the start of every
        single track.
        """
        patch = _to_patch(crop(frame_array, bbox), self.patch_size)

        sample = None
        if self._pre_ts is not None:
            sample = MotionSample(
                ts=ts,
                displacement_bl_per_s=_displacement_bl_per_s(
                    self._pre_bbox, bbox, ts - self._pre_ts, self.frame_size
                ),
                deformation_score=_deformation_score(self._pre_patch, patch),
                bbox=tuple(bbox),
                confidence=confidence,
            )

        self._pre_ts = ts
        self._pre_bbox = tuple(bbox)
        self._pre_patch = patch
        return sample