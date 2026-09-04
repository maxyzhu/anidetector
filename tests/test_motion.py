"""Tests for video.motion.

The numbers are hand-computed from the pixel geometry, so a regression shows up
as a wrong value rather than just a changed one.
"""

import numpy as np
import pytest

from video.motion import (
    MotionAccumulator,
    crop,
    _deformation_score,
    _displacement_bl_per_s,
    _to_patch,
)

WIDE = (1920, 1080)


def _box(cx, cy, w=0.1, h=0.1):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# --- displacement -----------------------------------------------------------


def test_a_still_animal_has_zero_displacement():
    box = _box(0.5, 0.5)
    assert _displacement_bl_per_s(box, box, 0.2, WIDE) == 0.0


def test_displacement_matches_the_hand_computed_pixel_geometry():
    # 200x100 frame, box 0.2x0.2 -> 40x20 px, diagonal sqrt(2000).
    # Centre moves 0.1 normalized -> 20 px. 20/sqrt(2000) = 1/sqrt(5).
    previous = (0.1, 0.1, 0.3, 0.3)
    current = (0.2, 0.1, 0.4, 0.3)

    got = _displacement_bl_per_s(previous, current, 1.0, (200, 100))

    assert got == pytest.approx(1 / np.sqrt(5))


def test_the_same_motion_at_twice_the_distance_reads_the_same():
    """Scale invariance — the reason body lengths are the unit at all."""
    near = _displacement_bl_per_s(
        (0.1, 0.1, 0.5, 0.5), (0.3, 0.1, 0.7, 0.5), 1.0, (200, 100)
    )
    far = _displacement_bl_per_s(
        (0.1, 0.1, 0.3, 0.3), (0.2, 0.1, 0.4, 0.3), 1.0, (200, 100)
    )

    assert near == pytest.approx(far)


def test_the_frame_aspect_ratio_is_respected():
    """Regression for the normalized-space trap.

    The same normalized step is 1920/1080 times further horizontally than
    vertically, so the two readings must differ by exactly the aspect ratio.
    Measuring in normalized space would make them equal.
    """
    horizontal = _displacement_bl_per_s(_box(0.4, 0.5), _box(0.5, 0.5), 1.0, WIDE)
    vertical = _displacement_bl_per_s(_box(0.5, 0.4), _box(0.5, 0.5), 1.0, WIDE)

    assert horizontal / vertical == pytest.approx(1920 / 1080)


def test_the_rate_scales_with_elapsed_time():
    slow = _displacement_bl_per_s(_box(0.4, 0.5), _box(0.5, 0.5), 1.0, WIDE)
    fast = _displacement_bl_per_s(_box(0.4, 0.5), _box(0.5, 0.5), 0.5, WIDE)

    assert fast == pytest.approx(slow * 2)


@pytest.mark.parametrize("dt", [0.0, -0.1])
def test_a_non_positive_interval_yields_zero_rather_than_infinity(dt):
    assert _displacement_bl_per_s(_box(0.4, 0.5), _box(0.5, 0.5), dt, WIDE) == 0.0


# --- deformation ------------------------------------------------------------


def _noise(height, width, seed=0):
    return np.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )


def test_an_unchanged_patch_scores_zero():
    patch = _noise(40, 40)
    assert _deformation_score(patch, patch) == 0.0


def test_pure_translation_barely_registers():
    """The independence property: translation is displacement's job, not this one."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[40:60, 20:40] = 255
    left = crop(frame, (0.1, 0.4, 0.2, 0.6))
    moved = np.zeros_like(frame)
    moved[40:60, 100:120] = 255
    right = crop(moved, (0.5, 0.4, 0.6, 0.6))

    assert _deformation_score(left, right) < 0.01


def test_a_changing_appearance_registers():
    assert _deformation_score(_noise(40, 40, seed=1), _noise(40, 40, seed=2)) > 0.1


def test_an_empty_crop_does_not_crash():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert _deformation_score(_to_patch(empty, 32), _to_patch(_noise(10, 10), 32)) >= 0.0


# --- crop -------------------------------------------------------------------


def test_crop_returns_the_requested_pixels():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[50:70, 100:140] = 255

    got = crop(frame, (0.5, 0.5, 0.7, 0.7))

    assert got.shape == (20, 40, 3)
    assert (got == 255).all()


@pytest.mark.parametrize("bbox", [(0.0, 0.0, 0.0, 0.0), (0.99, 0.99, 1.5, 1.5)])
def test_a_degenerate_or_out_of_bounds_box_still_yields_pixels(bbox):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    got = crop(frame, bbox)

    assert got.shape[0] >= 1 and got.shape[1] >= 1


# --- accumulator ------------------------------------------------------------


def test_the_first_observation_yields_no_sample():
    acc = MotionAccumulator(frame_size=(200, 100))
    frame = _noise(100, 200)

    assert acc.add(0.0, _box(0.5, 0.5), 0.9, frame) is None


def test_subsequent_observations_yield_samples():
    acc = MotionAccumulator(frame_size=(200, 100))
    frame = _noise(100, 200)
    acc.add(0.0, _box(0.4, 0.5), 0.9, frame)

    sample = acc.add(0.2, _box(0.5, 0.5), 0.77, frame)

    assert sample is not None
    assert sample.ts == 0.2
    assert sample.displacement_bl_per_s > 0
    assert sample.bbox == pytest.approx(_box(0.5, 0.5))
    assert sample.confidence == 0.77