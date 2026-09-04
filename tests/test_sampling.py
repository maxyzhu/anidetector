"""Tests for video.sampling."""

import pytest

from video.sampling import expected_frame_count, sample_at_fps


class _FakeFrame:
    """Only the timestamp matters to sampling."""

    def __init__(self, timestamp):
        self.timestamp = timestamp


def _frames(timestamps):
    return [_FakeFrame(t) for t in timestamps]


def _stamps(frames):
    return [f.timestamp for f in frames]


def test_a_constant_rate_source_is_thinned_to_the_target():
    # 25fps for 1s -> 5fps should keep every 5th frame.
    source = _frames([i * 0.04 for i in range(25)])

    kept = list(sample_at_fps(source, 5))

    assert _stamps(kept) == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])


def test_the_first_frame_is_always_kept():
    kept = list(sample_at_fps(_frames([0.0, 0.04, 0.08]), 5))

    assert _stamps(kept)[0] == 0.0


def test_gaps_never_fall_below_the_target_period():
    stamps = _stamps(list(sample_at_fps(_frames([i * 0.04 for i in range(100)]), 7)))

    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert min(gaps) >= (1/7) - 1e-6


def test_an_irregular_source_still_yields_even_time_spacing():
    """The reason sampling is time-based: a count-based stride would not.

    This source stutters — dense, then a long gap, then dense again. Every 5th
    frame would land at wildly uneven instants.
    """
    dense = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    stall = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05]
    kept = list(sample_at_fps(_frames(dense + stall), 5))

    gaps = [b - a for a, b in zip(_stamps(kept), _stamps(kept)[1:])]
    assert all(g >= 0.2 - 1e-6 for g in gaps)


def test_a_source_sparser_than_the_target_passes_through_untouched():
    # Nothing to thin: 1fps asked to produce 5fps.
    source = _frames([0.0, 1.0, 2.0])

    assert _stamps(list(sample_at_fps(source, 5))) == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("target", [None, 0, -1])
def test_a_disabled_target_passes_everything_through(target):
    source = _frames([0.0, 0.04, 0.08])

    assert len(list(sample_at_fps(source, target))) == 3


def test_expected_count_is_duration_times_rate():
    assert expected_frame_count(60.0, 5) == 300
    assert expected_frame_count(None, 5) is None
    assert expected_frame_count(60.0, None) is None