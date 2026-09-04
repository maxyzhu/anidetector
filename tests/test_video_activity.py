"""Tests for the activity layer of video.services: stored signals -> bouts.

These go through the database rather than calling video.bouts directly, because
the part worth guarding here is the grouping and the threshold plumbing, not the
segmentation maths that tests/test_bouts.py already covers.
"""

import pytest

from django.utils import timezone

from core.models import (
    Deployment, Detection, Image, Media, SpeciesClassification,
)
from video.models import MotionSignal, Track
from video.services import (
    Thresholds,
    activity_report,
    sensitivity_table,
)


def _thresholds(displacement_enter=0.5, displacement_exit=0.25,
                deformation_enter=0.5, deformation_exit=0.25,
                min_duration=0.0, max_gap=0.0):
    return Thresholds(
        displacement_enter_threshold=displacement_enter,
        displacement_exit_threshold=displacement_exit,
        deformation_enter_threshold=deformation_enter,
        deformation_exit_threshold=deformation_exit,
        min_duration=min_duration,
        max_gap=max_gap,
    )


@pytest.fixture
def media(db):
    deployment = Deployment.objects.create(
        camera_id="cam1", location="somewhere", country="USA",
        start_ts=timezone.now(),
    )
    return Media.objects.create(
        deployment=deployment, path="/tmp/activity.mp4", kind=Media.Kind.VIDEO
    )


def _classify(media, track, label):
    """Give a track the representative Detection a real run would leave it, plus
    one SpeciesNet row on it."""
    image = Image.objects.create(
        deployment=media.deployment,
        path=f"/tmp/track_{track.id}.jpg",
        source=Image.Source.VIDEO_FRAME,
        status=Image.Status.PROCESSED,
    )
    detection = Detection.objects.create(
        image=image, category=Detection.Category.ANIMAL, confidence=0.9,
        bbox_x1=0.0, bbox_y1=0.0, bbox_x2=0.1, bbox_y2=0.1,
    )
    SpeciesClassification.objects.create(detection=detection, category=label)
    track.rep_detection = detection
    track.save(update_fields=["rep_detection"])


@pytest.fixture
def make_track(media):
    """Build a track whose signals are the (displacement, deformation) pairs given,
    one per second."""
    def build(samples, species_label=None, step=1.0, confidence=0.9):
        track = Track.objects.create(
            media=media,
            start_frame=0,
            end_frame=max(len(samples) - 1, 0),
            start_ts=0.0,
            end_ts=max(len(samples) - 1, 0) * step,
            hits=len(samples),
        )
        if species_label is not None:
            _classify(media, track, species_label)
        MotionSignal.objects.bulk_create([
            MotionSignal(
                track=track,
                ts=index * step,
                displacement_bl_per_s=displacement,
                deformation_score=deformation,
                confidence=confidence,
                bbox_x1=0.0, bbox_y1=0.0, bbox_x2=0.1, bbox_y2=0.1,
            )
            for index, (displacement, deformation) in enumerate(samples)
        ])
        return track
    return build


# --- activity_report --------------------------------------------------------


@pytest.mark.django_db
def test_the_report_carries_the_track_identity_through(make_track):
    track = make_track([(1.0, 1.0)] * 4, species_label="Vulpes vulpes")

    (item,) = activity_report([track], _thresholds())

    assert item.track_id == track.id
    assert item.species_label == "Vulpes vulpes"
    assert (item.start_ts, item.end_ts) == (track.start_ts, track.end_ts)


@pytest.mark.django_db
def test_a_moving_animal_is_all_active(make_track):
    track = make_track([(1.0, 1.0)] * 4)

    (item,) = activity_report([track], _thresholds())

    assert item.budget.active_seconds == pytest.approx(3.0)
    assert item.budget.rest_seconds == 0.0
    assert item.budget.bout_count == 1


@pytest.mark.django_db
def test_a_still_animal_is_all_rest(make_track):
    track = make_track([(0.0, 0.0)] * 4)

    (item,) = activity_report([track], _thresholds())

    assert item.budget.active_seconds == 0.0
    assert item.budget.rest_seconds == pytest.approx(3.0)
    assert item.budget.bout_count == 0


@pytest.mark.django_db
def test_grooming_in_place_counts_as_active(make_track):
    """The whole point of OR-ing the channels: displacement stays at the noise
    floor while the animal is plainly doing something."""
    track = make_track([(0.0, 1.0)] * 4)

    (item,) = activity_report([track], _thresholds())

    assert item.budget.active_seconds == pytest.approx(3.0)
    assert item.budget.bout_count == 1


@pytest.mark.django_db
def test_the_budget_adds_up_to_the_observed_span(make_track):
    track = make_track([(1.0, 0.0), (1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0)])

    (item,) = activity_report([track], _thresholds())

    total = item.budget.active_seconds + item.budget.rest_seconds
    assert total == pytest.approx(4.0)


@pytest.mark.django_db
def test_min_duration_absorbs_a_flicker(make_track):
    samples = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    track = make_track(samples)

    without = activity_report([track], _thresholds())[0]
    absorbed = activity_report([track], _thresholds(min_duration=2.0))[0]

    assert without.budget.bout_count == 2
    assert absorbed.budget.bout_count == 1


@pytest.mark.django_db
def test_max_gap_bridges_a_brief_rest(make_track):
    samples = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    track = make_track(samples)

    bridged = activity_report([track], _thresholds(max_gap=2.0))[0]

    assert bridged.budget.bout_count == 1
    assert bridged.budget.rest_seconds == 0.0


@pytest.mark.django_db
def test_a_track_with_no_signals_reports_an_empty_budget(make_track):
    track = make_track([])

    (item,) = activity_report([track], _thresholds())

    assert item.bouts == []
    assert item.budget.active_seconds == 0.0
    assert item.budget.bout_count == 0


@pytest.mark.django_db
def test_each_track_is_segmented_from_its_own_signals(make_track):
    moving = make_track([(1.0, 1.0)] * 4)
    still = make_track([(0.0, 0.0)] * 4)

    report = activity_report([moving, still], _thresholds())

    assert [item.track_id for item in report] == [moving.id, still.id]
    assert report[0].budget.bout_count == 1
    assert report[1].budget.bout_count == 0


@pytest.mark.django_db
def test_the_query_count_does_not_grow_with_the_tracks(
    make_track, django_assert_num_queries
):
    """One query for the signals and one for the labels, however many tracks."""
    tracks = [
        make_track([(1.0, 0.0)] * 4, species_label="Vulpes vulpes")
        for _ in range(3)
    ]

    with django_assert_num_queries(2):
        activity_report(tracks, _thresholds())


# --- sensitivity_table ------------------------------------------------------


@pytest.mark.django_db
def test_the_sweep_has_a_row_per_threshold_and_duration(make_track):
    track = make_track([(1.0, 0.0)] * 4)

    rows = sensitivity_table(
        [track], _thresholds(), [0.2, 0.5], min_durations=[0.0, 1.0]
    )

    assert len(rows) == 4
    assert [(r.enter_threshold, r.min_duration) for r in rows] == [
        (0.2, 0.0), (0.2, 1.0), (0.5, 0.0), (0.5, 1.0)
    ]


@pytest.mark.django_db
def test_raising_the_threshold_reduces_active_time(make_track):
    """The regression that matters: the sweep must actually use its own enter
    value rather than the base thresholds for every row."""
    track = make_track([(0.1, 0.0), (0.4, 0.0), (0.7, 0.0), (1.0, 0.0), (0.4, 0.0)])

    rows = sensitivity_table([track], _thresholds(), [0.2, 0.5, 0.9])

    active = [row.active_seconds for row in rows]
    assert active == sorted(active, reverse=True)
    assert len(set(active)) > 1


@pytest.mark.django_db
def test_the_sweep_totals_across_tracks(make_track):
    tracks = [make_track([(1.0, 0.0)] * 4) for _ in range(2)]

    (row,) = sensitivity_table(tracks, _thresholds(), [0.5])

    assert row.bouts == 2
    assert row.active_seconds == pytest.approx(6.0)
    assert row.observed_seconds == pytest.approx(6.0)


@pytest.mark.django_db
def test_the_hysteresis_ratio_sets_the_exit_threshold(make_track):
    # 0.3 sits in the dead band at ratio 0.5 (enter 0.6, exit 0.3) but not at 1.0.
    track = make_track([(0.7, 0.0), (0.3, 0.0), (0.7, 0.0)])

    with_dead_band = sensitivity_table(
        [track], _thresholds(), [0.6], hysteresis_ratio=0.5
    )[0]
    without = sensitivity_table(
        [track], _thresholds(), [0.6], hysteresis_ratio=1.0
    )[0]

    assert with_dead_band.bouts == 1
    assert without.bouts == 2


@pytest.mark.django_db
def test_active_fraction_is_none_when_nothing_was_observed(make_track):
    track = make_track([])

    (row,) = sensitivity_table([track], _thresholds(), [0.5])

    assert row.observed_seconds == 0.0
    assert row.active_fraction is None


@pytest.mark.django_db
def test_active_fraction_is_the_share_of_observed_time(make_track):
    track = make_track([(1.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])

    (row,) = sensitivity_table([track], _thresholds(), [0.5])

    assert row.active_fraction == pytest.approx(row.active_seconds / 4.0)
