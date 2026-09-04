"""Tests for video.frame — representative frame selection."""

import numpy as np
import pytest
from django.core.exceptions import ImproperlyConfigured
from PIL import Image as PILImage

from video.frame import RepresentativePicker, get_scorer


def _frame(width=320, height=240, value=120):
    return np.full((height, width, 3), value, dtype=np.uint8)


def _noise(width=320, height=240, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def _box(size, cx=0.5, cy=0.5):
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


# --- scorers ----------------------------------------------------------------


def test_the_default_scorer_ranks_by_area(settings):
    settings.VIDEO_FRAME_SELECTOR = "area"
    scorer = get_scorer()
    frame = _frame()

    assert scorer(_box(0.4), frame) > scorer(_box(0.2), frame)


def test_the_sharpness_scorer_prefers_detail_at_equal_area():
    scorer = get_scorer("area_sharpness")
    box = _box(0.5)

    assert scorer(box, _noise()) > scorer(box, _frame())


def test_an_unknown_selector_is_rejected():
    with pytest.raises(ImproperlyConfigured, match="VIDEO_FRAME_SELECTOR"):
        get_scorer("no-such-scorer")


# --- picker -----------------------------------------------------------------


def test_a_picker_offered_nothing_has_no_frame():
    picker = RepresentativePicker()

    assert not picker.chosen
    assert picker.encode() is None


def test_the_largest_box_wins():
    picker = RepresentativePicker()

    picker.offer(0.0, _box(0.2), 0.5, _frame())
    picker.offer(1.0, _box(0.6), 0.9, _frame())
    picker.offer(2.0, _box(0.3), 0.7, _frame())

    assert picker.ts == 1.0
    assert picker.confidence == 0.9
    assert picker.bbox == _box(0.6)


def test_offer_reports_whether_it_took_the_frame():
    picker = RepresentativePicker()

    assert picker.offer(0.0, _box(0.2), 0.5, _frame()) is True
    assert picker.offer(1.0, _box(0.1), 0.9, _frame()) is False


def test_the_kept_frame_is_downscaled_on_the_way_in():
    """The point of resizing eagerly: a live track holds a display-sized frame,
    not the 4K one it came from."""
    picker = RepresentativePicker(max_width=160)

    picker.offer(0.0, _box(0.5), 0.9, _frame(width=1920, height=1080))

    assert picker._frame.shape[1] == 160


def test_a_frame_narrower_than_the_cap_is_left_alone():
    picker = RepresentativePicker(max_width=1920)

    picker.offer(0.0, _box(0.5), 0.9, _frame(width=320, height=240))

    assert picker._frame.shape[1] == 320


def test_encode_returns_a_readable_jpeg():
    import io

    picker = RepresentativePicker(max_width=160)
    picker.offer(0.0, _box(0.5), 0.9, _noise(width=640, height=480))

    with PILImage.open(io.BytesIO(picker.encode())) as image:
        assert image.format == "JPEG"
        assert image.width == 160
