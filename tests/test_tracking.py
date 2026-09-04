"""Tests for video.tracking."""

import numpy as np
import pytest

from video.tracking import associate, iou_matrix
from video.tracking import KalmanBoxFilter, _box_to_state, _state_to_box
from video.tracking import Tracker


# --- IoU --------------------------------------------------------------------


def test_identical_boxes_overlap_completely():
    box = [(0.1, 0.1, 0.3, 0.3)]
    assert iou_matrix(box, box)[0, 0] == pytest.approx(1.0)


def test_disjoint_boxes_do_not_overlap():
    assert iou_matrix([(0.0, 0.0, 0.1, 0.1)], [(0.5, 0.5, 0.6, 0.6)])[0, 0] == 0.0


def test_partial_overlap_matches_the_hand_computed_ratio():
    # 2x2 boxes offset by (1,1): intersection 1, union 4 + 4 - 1 = 7.
    got = iou_matrix([(0.0, 0.0, 2.0, 2.0)], [(1.0, 1.0, 3.0, 3.0)])
    assert got[0, 0] == pytest.approx(1 / 7)


def test_iou_is_unaffected_by_the_frame_aspect_ratio():
    """The contrast with displacement: a ratio of areas cancels the scaling."""
    normalized_a = [(0.10, 0.10, 0.30, 0.30)]
    normalized_b = [(0.20, 0.20, 0.40, 0.40)]
    scale = np.array([1920, 1080, 1920, 1080])
    pixels_a = [tuple(np.array(normalized_a[0]) * scale)]
    pixels_b = [tuple(np.array(normalized_b[0]) * scale)]

    assert iou_matrix(normalized_a, normalized_b)[0, 0] == pytest.approx(
        iou_matrix(pixels_a, pixels_b)[0, 0]
    )


def test_a_zero_area_box_does_not_divide_by_zero():
    assert iou_matrix([(0.5, 0.5, 0.5, 0.5)], [(0.4, 0.4, 0.6, 0.6)])[0, 0] == 0.0


@pytest.mark.parametrize("a, b, shape", [([], [(0, 0, 1, 1)], (0, 1)),
                                         ([(0, 0, 1, 1)], [], (1, 0)),
                                         ([], [], (0, 0))])
def test_empty_input_gives_an_empty_matrix(a, b, shape):
    assert iou_matrix(a, b).shape == shape


# --- association ------------------------------------------------------------


def test_a_clean_one_to_one_case_matches_everything():
    iou = np.array([[0.9, 0.0], [0.0, 0.8]])

    matches, unmatched_rows, unmatched_cols = associate(iou, 0.3)

    assert matches == [(0, 0), (1, 1)]
    assert unmatched_rows == [] and unmatched_cols == []


def test_the_optimal_assignment_beats_the_greedy_one():
    """Two animals crossing.

    Greedy takes the global maximum 0.60 first, which strands row 1 with a
    zero-overlap column and loses its track. The solver trades 0.60 for
    0.50 + 0.55 and keeps both.
    """
    iou = np.array([[0.60, 0.50], [0.55, 0.00]])

    matches, unmatched_rows, _ = associate(iou, 0.3)

    assert matches == [(0, 1), (1, 0)]
    assert unmatched_rows == []


def test_a_pair_below_the_threshold_is_not_a_match():
    """The solver will pair anything if nothing better is left; the gate stops it."""
    iou = np.array([[0.05]])

    matches, unmatched_rows, unmatched_cols = associate(iou, 0.3)

    assert matches == []
    assert unmatched_rows == [0] and unmatched_cols == [0]


def test_a_new_detection_comes_back_unmatched():
    iou = np.array([[0.9, 0.0]])

    matches, _, unmatched_cols = associate(iou, 0.3)

    assert matches == [(0, 0)]
    assert unmatched_cols == [1]


def test_a_disappeared_object_leaves_its_track_unmatched():
    iou = np.array([[0.9], [0.0]])

    matches, unmatched_rows, _ = associate(iou, 0.3)

    assert matches == [(0, 0)]
    assert unmatched_rows == [1]


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (0, 0)])
def test_an_empty_matrix_leaves_everything_unmatched(shape):
    matches, unmatched_rows, unmatched_cols = associate(np.zeros(shape), 0.3)

    assert matches == []
    assert len(unmatched_rows) == shape[0]
    assert len(unmatched_cols) == shape[1]


# --- Kalman Box Filter -------------------------------------------------------


def _box(cx, cy, w=0.1, h=0.1):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_box_and_state_round_trip():
    box = (0.2, 0.3, 0.5, 0.7)
    assert _state_to_box(_box_to_state(box)) == pytest.approx(box)


def test_a_still_box_stays_where_it_is():
    kf = KalmanBoxFilter(_box(0.5, 0.5))
    for _ in range(10):
        kf.predict(0.2)
        kf.update(_box(0.5, 0.5))

    assert kf.box == pytest.approx(_box(0.5, 0.5), abs=1e-3)


def test_the_filter_learns_a_constant_velocity_and_extrapolates_it():
    """The whole reason for a motion model: predict where it will be, not where
    it was, so association still works when the animal has moved."""
    kf = KalmanBoxFilter(_box(0.10, 0.5))
    for step in range(1, 8):
        kf.predict(0.2)
        kf.update(_box(0.10 + 0.05 * step, 0.5))

    predicted = kf.predict(0.2)

    # Truth would be 0.50; a filter with no motion model would still say 0.45.
    centre = (predicted[0] + predicted[2]) / 2
    assert centre == pytest.approx(0.50, abs=0.02)


def test_prediction_scales_with_the_time_step():
    def filtered_then_predicted(dt):
        kf = KalmanBoxFilter(_box(0.10, 0.5))
        for step in range(1, 8):
            kf.predict(0.2)
            kf.update(_box(0.10 + 0.05 * step, 0.5))
        before = (kf.box[0] + kf.box[2]) / 2      # measure the baseline, do not assume it
        box = kf.predict(dt)
        return (box[0] + box[2]) / 2 - before

    assert filtered_then_predicted(0.4) == pytest.approx(
        filtered_then_predicted(0.2) * 2, rel=0.1
    )


def test_filtering_beats_trusting_the_last_measurement():
    """Detector boxes jitter even on a motionless animal; smoothing that is what
    keeps the displacement signal from reading noise as movement."""
    rng = np.random.default_rng(0)
    truth = 0.5
    kf = KalmanBoxFilter(_box(truth, 0.5))
    last_measurement = truth
    for _ in range(30):
        last_measurement = truth + rng.normal(0, 0.01)
        kf.predict(0.2)
        kf.update(_box(last_measurement, 0.5))

    filtered = (kf.box[0] + kf.box[2]) / 2

    assert abs(filtered - truth) < abs(last_measurement - truth)


def test_a_long_gap_never_produces_an_inside_out_box():
    """A lost track only predicts. Without a floor the height goes negative, the
    box flips, every IoU becomes 0 and the track can never be recovered."""
    kf = KalmanBoxFilter(_box(0.5, 0.5, h=0.1))
    for step in range(1, 6):
        kf.predict(0.2)
        kf.update(_box(0.5, 0.5, h=0.1 - 0.015 * step))   # shrinking fast

    for _ in range(50):
        x1, y1, x2, y2 = kf.predict(0.2)
        assert x2 >= x1 and y2 >= y1


def test_an_update_pulls_the_state_toward_the_measurement():
    kf = KalmanBoxFilter(_box(0.5, 0.5))
    kf.predict(0.2)
    before = (kf.box[0] + kf.box[2]) / 2

    kf.update(_box(0.7, 0.5))
    after = (kf.box[0] + kf.box[2]) / 2

    assert before < after < 0.7          # moved toward it, not all the way


@pytest.mark.parametrize("dt", [0.0, -0.5])
def test_a_stalled_or_backwards_gap_never_rewinds_the_state(dt):
    """decode.py warns on backwards timestamps; the filter must not act on them."""
    kf = KalmanBoxFilter(_box(0.10, 0.5))
    for step in range(1, 8):
        kf.predict(0.2)
        kf.update(_box(0.10 + 0.05 * step, 0.5))
    before = (kf.box[0] + kf.box[2]) / 2

    after = kf.predict(dt)

    assert (after[0] + after[2]) / 2 == pytest.approx(before, abs=1e-4)

# --- Tracker ----------------------------------------------------------------


def _feed(tracker, positions, step=0.2, start_frame=0, confidence=0.9):
    """positions[i] is the list of box centres visible at observation i."""
    seen = []
    for index, centres in enumerate(positions):
        detections = [(_box(cx, 0.5), confidence) for cx in centres]
        seen.append(tracker.update(detections, start_frame * step + index * step, index))
    return seen


def test_one_animal_walking_across_is_one_track():
    tracker = Tracker(min_hits=2)

    seen = _feed(tracker, [[0.1 + 0.03 * i] for i in range(15)])

    ids = {t.track_id for frame in seen for t in frame}
    assert ids == {1}


def test_the_matched_detection_confidence_comes_back_out():
    """The smoothed bbox cannot be matched to a detection afterwards, so the
    confidence has to ride through the tracker."""
    tracker = Tracker(min_hits=2)

    seen = _feed(tracker, [[0.1 + 0.03 * i] for i in range(5)], confidence=0.62)

    assert all(t.confidence == 0.62 for frame in seen for t in frame)


def test_two_animals_keep_separate_ids():
    tracker = Tracker(min_hits=2)

    seen = _feed(tracker, [[0.1 + 0.02 * i, 0.8 - 0.02 * i] for i in range(10)])

    assert {t.track_id for t in seen[-1]} == {1, 2}


def test_a_track_is_not_reported_until_it_is_confirmed():
    tracker = Tracker(min_hits=3)

    seen = _feed(tracker, [[0.5], [0.5], [0.5]])

    assert seen[0] == [] and seen[1] == []
    assert len(seen[2]) == 1


def test_a_tentative_track_that_misses_once_is_dropped_silently():
    tracker = Tracker(min_hits=3)

    _feed(tracker, [[0.5], [], [], []])

    assert tracker.finished() == []
    assert tracker.flush() == []


def test_a_short_occlusion_keeps_the_same_id():
    tracker = Tracker(max_age_seconds=2.0, min_hits=2)

    seen = _feed(tracker, [[0.5]] * 4 + [[]] * 5 + [[0.5]] * 3)   # 1.0s gap

    assert {t.track_id for t in seen[-1]} == {1}
    assert tracker.finished() == []


def test_an_occlusion_longer_than_max_age_fragments_the_track():
    """This is the max_age sensitivity in one test: too small and one animal
    becomes several, inflating the individual count."""
    tracker = Tracker(max_age_seconds=0.5, min_hits=2)

    seen = _feed(tracker, [[0.5]] * 4 + [[]] * 6 + [[0.5]] * 4)   # 1.2s gap

    assert {t.track_id for t in seen[-1]} == {2}
    assert [t.track_id for t in tracker.finished()] == [1]


def test_max_age_in_seconds_does_not_depend_on_the_sampling_rate():
    """Expressed in frames this would need retuning whenever VIDEO_TRACK_FPS
    changed; expressed in seconds the same occlusion gives the same verdict."""

    def id_after_a_one_second_gap(step):
        tracker = Tracker(max_age_seconds=2.0, min_hits=2)
        gap = [[]] * int(1.0 / step)
        seen = _feed(tracker, [[0.5]] * 5 + gap + [[0.5]] * 3, step=step)
        return {t.track_id for t in seen[-1]}

    assert id_after_a_one_second_gap(0.2) == id_after_a_one_second_gap(0.1) == {1}


def test_a_finished_track_ends_at_its_last_observation():
    """Not at the moment we gave up, or every track gains max_age of phantom
    duration and every activity budget is biased high."""
    tracker = Tracker(max_age_seconds=0.5, min_hits=2)

    _feed(tracker, [[0.5]] * 4 + [[]] * 6)

    finished = tracker.finished()[0]
    assert finished.start_ts == pytest.approx(0.0)
    assert finished.end_ts == pytest.approx(0.6)     # observation index 3
    assert (finished.start_frame, finished.end_frame) == (0, 3)


def test_flush_retires_whatever_is_still_alive():
    tracker = Tracker(min_hits=2)
    _feed(tracker, [[0.5]] * 5)

    assert tracker.finished() == []
    assert [t.track_id for t in tracker.flush()] == [1]