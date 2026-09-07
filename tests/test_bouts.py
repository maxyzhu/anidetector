"""Tests for video.bouts."""

import pytest

from video.bouts import (
    activity_budget,
    classify,
    find_bouts,
    union,
)


def _series(values, step=1.0):
    return [(i * step, v) for i, v in enumerate(values)]


def _bouts(values, enter_threshold, exit_threshold=None, min_duration=0.0,
           max_gap=0.0, step=1.0):
    """The whole pipeline, since classify and find_bouts are only useful paired."""
    return find_bouts(
        classify(_series(values, step), enter_threshold, exit_threshold),
        min_duration,
        max_gap,
    )


# --- classify ---------------------------------------------------------------


def test_classify_keeps_one_judgement_per_sample():
    judgements = classify(_series([1.0, 0.0, 1.0]), 0.5)

    assert [ts for ts, _ in judgements] == [0.0, 1.0, 2.0]
    assert [active for _, active in judgements] == [True, False, True]


def test_classify_treats_the_threshold_as_inclusive():
    assert classify([(0.0, 0.5)], 0.5) == [(0.0, True)]


def test_classify_defaults_the_exit_to_the_enter_threshold():
    values = [0.6, 0.3, 0.6]

    assert classify(_series(values), 0.5) == classify(_series(values), 0.5, 0.5)


def test_classify_has_nothing_to_say_about_an_empty_series():
    assert classify([], 0.5) == []


def test_an_exit_above_the_enter_threshold_is_rejected():
    with pytest.raises(ValueError, match="hysteresis"):
        classify(_series([1.0]), 0.2, 0.5)


# --- hysteresis -------------------------------------------------------------


def test_a_dip_into_the_dead_band_does_not_end_the_bout():
    """The reason hysteresis exists: without it this is three bouts, not one."""
    values = [0.6, 0.3, 0.6]

    with_hysteresis = _bouts(values, 0.5, 0.2)
    without = _bouts(values, 0.5)

    assert len(with_hysteresis) == 1
    assert len(without) == 3


def test_dropping_below_the_exit_threshold_does_end_the_bout():
    bouts = _bouts([0.6, 0.1, 0.6], 0.5, 0.2)

    assert [b.active for b in bouts] == [True, False, True]


def test_the_dead_band_also_holds_a_rest_bout_shut():
    # 0.3 sits between the thresholds, so a resting animal stays resting.
    bouts = _bouts([0.0, 0.3, 0.0], 0.5, 0.2)

    assert len(bouts) == 1
    assert not bouts[0].active


def test_equal_thresholds_are_stable_on_a_constant_signal():
    # The degenerate case a single-threshold sensitivity sweep uses.
    bouts = _bouts([0.5] * 5, 0.5, 0.5)

    assert len(bouts) == 1


# --- union ------------------------------------------------------------------


def test_union_is_an_or_across_channels():
    displacement = [(0.0, True), (1.0, False), (2.0, False)]
    deformation = [(0.0, False), (1.0, True), (2.0, False)]

    assert union(displacement, deformation) == [
        (0.0, True), (1.0, True), (2.0, False)
    ]


def test_union_of_one_channel_is_that_channel():
    channel = [(0.0, True), (1.0, False)]

    assert union(channel) == channel


def test_union_rejects_channels_sampled_at_different_timestamps():
    with pytest.raises(ValueError, match="same timestamps"):
        union([(0.0, True)], [(0.5, True)])


def test_union_of_no_channels_is_empty():
    assert union() == []


# --- basic segmentation -----------------------------------------------------


def test_a_constantly_moving_animal_is_one_active_bout():
    bouts = _bouts([1.0] * 5, 0.5)

    assert len(bouts) == 1
    assert bouts[0].active
    assert (bouts[0].start_ts, bouts[0].end_ts) == (0.0, 4.0)


def test_a_still_animal_is_one_rest_bout():
    bouts = _bouts([0.0] * 5, 0.5)

    assert len(bouts) == 1
    assert not bouts[0].active


def test_crossing_the_threshold_starts_a_new_bout():
    bouts = _bouts([1.0, 1.0, 0.0, 0.0], 0.5)

    assert [b.active for b in bouts] == [True, False]


def test_no_judgements_gives_no_bouts():
    assert find_bouts([]) == []


def test_a_single_sample_gives_one_zero_length_bout():
    bouts = _bouts([1.0], 0.5)

    assert len(bouts) == 1
    assert bouts[0].duration == 0.0


def test_a_zero_length_bout_is_tiled_but_not_counted_as_activity():
    """Turning active on the last sample tiles a zero-length bout. It has to
    exist for the spans to abut, and it must not read as an episode."""
    bouts = _bouts([0.0, 0.0, 1.0], 0.5)
    budget = activity_budget(bouts)

    assert [(b.active, b.duration) for b in bouts] == [(False, 2.0), (True, 0.0)]
    assert budget.active_seconds == 0.0
    assert budget.bout_count == 0


# --- the tiling property ----------------------------------------------------


def test_bouts_abut_and_cover_the_observed_span():
    """Active + rest must equal the span, or an activity budget cannot add up."""
    values = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    bouts = _bouts(values, 0.5)

    assert bouts[0].start_ts == 0.0
    assert bouts[-1].end_ts == float(len(values) - 1)
    for current, following in zip(bouts, bouts[1:]):
        assert current.end_ts == following.start_ts
    budget = activity_budget(bouts)
    assert budget.active_seconds + budget.rest_seconds == pytest.approx(5.0)


# --- gap bridging and short-bout absorption ---------------------------------


def test_a_brief_rest_between_two_active_bouts_is_bridged():
    # active 0-3, rest 3-4, active 4-6.
    values = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]

    without = _bouts(values, 0.5)
    bridged = _bouts(values, 0.5, max_gap=2.0)

    assert len(without) == 3
    assert len(bridged) == 1
    assert bridged[0].active


def test_a_rest_at_the_edge_is_not_bridged():
    """It is the edge of the observation, not a gap inside activity."""
    bouts = _bouts([0.0, 1.0, 1.0, 1.0], 0.5, max_gap=5.0)

    assert [b.active for b in bouts] == [False, True]


def test_a_bout_shorter_than_the_minimum_is_absorbed():
    bouts = _bouts([1.0, 1.0, 1.0, 0.0, 1.0, 1.0], 0.5, min_duration=2.0)

    assert len(bouts) == 1


def test_absorption_repeats_until_everything_is_long_enough():
    bouts = _bouts([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], 0.5, min_duration=2.0)

    assert all(b.duration >= 2.0 for b in bouts)


# --- the shape the sensitivity figure needs ---------------------------------


def test_sweeping_the_threshold_changes_the_budget_monotonically():
    """S9.2's first figure is exactly this sweep, so it has to be cheap and pure."""
    values = [0.1, 0.4, 0.7, 1.0, 0.7, 0.4, 0.1]

    budgets = [activity_budget(_bouts(values, t)) for t in (0.2, 0.5, 0.8)]

    active = [b.active_seconds for b in budgets]
    assert active == sorted(active, reverse=True)
