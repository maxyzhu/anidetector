"""End-to-end tests for the video pipeline, on a clip with known ground truth."""

import pytest
from celery.exceptions import OperationalError

from core.models import Deployment, Detection, Image, Media
from django.utils import timezone
from inference import ParsedDetection
from video import services
from video.models import MotionSignal, Track

from video_fixtures import write_moving_rectangle

from pathlib import Path


class _FakeDetector:
    """Reports the rectangle the fixture drew, so the pipeline sees a real track
    without loading any weights."""

    def __init__(self, size=(160, 120), box=(20, 20), x_per_frame=1.0, fps=25):
        self.width, self.height = size
        self.box_w, self.box_h = box
        self.x_per_frame = x_per_frame
        self.fps = fps
        self.calls = 0

    def detect_batch(self, images, conf_threshold):
        self.calls += 1
        # The fixture moves the rectangle x_per_frame pixels per source frame.
        x = ((self.calls - 1) * self.x_per_frame * self.fps / 5) % self.width
        y = (self.height - self.box_h) // 2
        return [[ParsedDetection(
            "animal", 0.9,
            x / self.width, y / self.height,
            min(x + self.box_w, self.width) / self.width,
            (y + self.box_h) / self.height,
        )]]


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    write_moving_rectangle(path, frames=100, fps=25, gop=10)
    return path


@pytest.fixture
def media(clip, db):
    deployment = Deployment.objects.create(
        camera_id="cam1", location="somewhere", country="USA",
        start_ts=timezone.now(),
    )
    return Media.objects.create(
        deployment=deployment, path=str(clip), kind=Media.Kind.VIDEO
    )


@pytest.fixture
def fake_detector(monkeypatch):
    detector = _FakeDetector()
    monkeypatch.setattr(services, "get_detector", lambda device="cpu": detector)
    return detector


@pytest.mark.django_db
def test_probing_fills_in_the_media_metadata(media, fake_detector):
    list(services.process_media(media))

    media.refresh_from_db()
    assert (media.width, media.height) == (160, 120)
    assert media.fps == pytest.approx(25)
    assert media.gop_size == 10


@pytest.mark.django_db
def test_a_single_moving_object_produces_one_track(media, fake_detector):
    list(services.process_media(media))

    assert Track.objects.count() == 1
    track = Track.objects.get()
    assert track.media_id == media.id
    assert track.end_ts > track.start_ts


@pytest.mark.django_db
def test_the_track_carries_motion_signals(media, fake_detector):
    list(services.process_media(media))

    signals = MotionSignal.objects.filter(track__media=media).order_by("ts")
    assert signals.count() > 1
    assert all(s.displacement_bl_per_s >= 0 for s in signals)
    # Detector confidence has to survive the tracker, or the species queue —
    # which filters on it — would never pick these boxes up.
    assert all(s.confidence == pytest.approx(0.9) for s in signals)
    # Timestamps must be strictly increasing, or bouts cannot segment them.
    stamps = [s.ts for s in signals]
    assert stamps == sorted(stamps)


@pytest.mark.django_db
def test_the_track_gets_a_representative_frame(media, fake_detector, settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path
    list(services.process_media(media))

    track = Track.objects.get()
    assert track.rep_detection is not None
    assert track.rep_ts is not None

    image = track.rep_detection.image
    assert image.source == Image.Source.VIDEO_FRAME
    assert Path(image.path).exists()
    assert (image.width, image.height) == (160, 120)


@pytest.mark.django_db
def test_the_representative_detection_waits_in_the_species_queue(
    media, fake_detector, settings, tmp_path
):
    """The whole point of shaping it as a Detection: the existing SpeciesNet drain
    picks the track up without knowing it came from video."""
    settings.MEDIA_ROOT = tmp_path
    list(services.process_media(media))

    detection = Track.objects.get().rep_detection
    assert detection.status == Detection.Status.PENDING
    assert detection.category == Detection.Category.ANIMAL
    assert detection.confidence == pytest.approx(0.9)


@pytest.mark.django_db
def test_finishing_a_track_wakes_the_species_drain(
    media, fake_detector, settings, tmp_path, monkeypatch,
    django_capture_on_commit_callbacks,
):
    """By task name, so no import of the image package is needed — and only after
    commit, or the worker could claim a row that is not there yet."""
    settings.MEDIA_ROOT = tmp_path
    sent = []
    monkeypatch.setattr(
        services.current_app, "send_task", lambda name, *a, **kw: sent.append(name)
    )

    with django_capture_on_commit_callbacks(execute=True):
        list(services.process_media(media))

    assert sent == ["image.tasks.classify_pending_task"]


@pytest.mark.django_db
def test_a_broker_that_is_down_does_not_fail_the_run(
    media, fake_detector, settings, tmp_path, monkeypatch,
    django_capture_on_commit_callbacks,
):
    def refuse(*args, **kwargs):
        raise OperationalError("no broker")

    settings.MEDIA_ROOT = tmp_path
    monkeypatch.setattr(services.current_app, "send_task", refuse)

    with django_capture_on_commit_callbacks(execute=True):
        list(services.process_media(media))

    # The detection is still queued, so a later drain picks it up.
    assert Track.objects.get().rep_detection.status == Detection.Status.PENDING


@pytest.mark.django_db
def test_the_photo_gallery_does_not_show_video_frames(media, fake_detector, settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path
    list(services.process_media(media))

    assert Image.objects.count() == 1
    assert not Image.objects.filter(source=Image.Source.INGEST).exists()


@pytest.mark.django_db
def test_the_media_row_records_that_it_finished(media, fake_detector):
    list(services.process_media(media))

    media.refresh_from_db()
    assert media.status == Media.Status.PROCESSED
    assert media.error == ""


@pytest.mark.django_db
def test_a_failure_is_recorded_on_the_row(media, monkeypatch):
    """A batch run that resumes must be able to skip or retry a bad file rather
    than rediscover the failure."""
    class BadDetector:
        def detect_batch(self, frames, conf):
            raise RuntimeError("bad model")

    monkeypatch.setattr(
        services, "get_detector", lambda device="cpu": BadDetector()
    )

    with pytest.raises(RuntimeError):
        list(services.process_media(media))

    media.refresh_from_db()
    assert media.status == Media.Status.FAILED
    assert "bad model" in media.error


@pytest.mark.django_db
def test_progress_reports_position_and_counts(media, fake_detector):
    events = list(services.process_media(media, progress_every=5))

    assert events[-1].frames > 0
    assert events[-1].tracks == Track.objects.count()
    assert events[-1].fraction == pytest.approx(1.0, abs=0.2)


@pytest.mark.django_db
def test_only_pending_video_media_is_queued(media, fake_detector):
    assert list(services.pending_video_media()) == [media]

    list(services.process_media(media))

    assert list(services.pending_video_media()) == []