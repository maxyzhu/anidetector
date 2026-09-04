"""Tests for the ingest services.

The detector is faked throughout — no weights, no torch — so these cover the part
that actually broke historically: which rows get picked up, which get marked
failed, and whether EXIF time survives the trip into the database.
"""

from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from PIL import Image as PILImage

from django.utils import timezone

from core.models import Deployment, Detection, Image
from image import services
from image.services import Phase, ingest_directory, read_captured_at
from inference import ParsedDetection


class _FakeDetector:
    """Stand-in for inference.Detector: one animal box per image, no weights."""

    def __init__(self, per_image=1):
        self.per_image = per_image
        self.seen = 0

    def detect_batch(self, images, conf_threshold):
        self.seen += len(images)
        return [
            [ParsedDetection("animal", 0.9, 0.1, 0.1, 0.5, 0.5)] * self.per_image
            for _ in images
        ]


@pytest.fixture
def fake_detector(monkeypatch):
    detector = _FakeDetector()
    monkeypatch.setattr(services, "get_detector", lambda device="cpu": detector)
    return detector


def _write_images(folder, count, exif_time=None):
    """Write `count` tiny PNG/JPEG files, optionally carrying an EXIF timestamp."""
    paths = []
    for i in range(count):
        path = folder / f"img{i:02d}.jpg"
        img = PILImage.new("RGB", (16, 16), (i * 10 % 255, 120, 120))
        if exif_time:
            exif = img.getexif()
            exif[306] = exif_time(i)  # DateTime
            img.save(path, exif=exif)
        else:
            img.save(path)
        paths.append(path)
    return paths


def _deployment():
    return Deployment.objects.get_or_create(
        camera_id="cam1", location="somewhere", country="USA",
        defaults={"start_ts": timezone.now()},
    )[0]


def _run(folder, **kwargs):
    return list(ingest_directory(str(folder), _deployment(), **kwargs))


# --- registration -----------------------------------------------------------


@pytest.mark.django_db
def test_registration_creates_one_row_per_file(tmp_path, fake_detector):
    _write_images(tmp_path, 3)

    events = _run(tmp_path)

    registered = events[0]
    assert registered.phase is Phase.REGISTER
    assert (registered.total, registered.done) == (3, 3)
    assert Image.objects.count() == 3


@pytest.mark.django_db
def test_rerunning_registers_nothing_new(tmp_path, fake_detector):
    _write_images(tmp_path, 3)
    _run(tmp_path)

    registered = _run(tmp_path)[0]

    assert (registered.total, registered.done) == (3, 0)
    assert Image.objects.count() == 3


@pytest.mark.django_db
def test_limit_caps_new_rows(tmp_path, fake_detector):
    _write_images(tmp_path, 5)

    registered = _run(tmp_path, limit=2)[0]

    assert registered.done == 2
    assert Image.objects.count() == 2


@pytest.mark.django_db
def test_a_missing_directory_is_reported_not_swallowed(tmp_path):
    with pytest.raises(NotADirectoryError):
        _run(tmp_path / "nope")


# --- detection --------------------------------------------------------------


@pytest.mark.django_db
def test_every_pending_image_is_processed_across_batches(tmp_path, fake_detector):
    """Regression: the old slice-by-offset loop skipped rows.

    Rows leave the pending queryset as they are processed, so a fixed offset
    stepped over the ones that shifted down behind it — with 5 images and a batch
    size of 2, only the first and last batches' worth ever got detected.
    """
    _write_images(tmp_path, 5)

    _run(tmp_path, batch_size=2)

    assert fake_detector.seen == 5
    assert Image.objects.filter(status=Image.Status.PROCESSED).count() == 5
    assert Detection.objects.count() == 5


@pytest.mark.django_db
def test_progress_reaches_the_total(tmp_path, fake_detector):
    _write_images(tmp_path, 5)

    events = _run(tmp_path, batch_size=2)

    detect_events = [e for e in events if e.phase is Phase.DETECT]
    assert [e.done for e in detect_events] == [2, 4, 5]
    assert all(e.total == 5 for e in detect_events)


@pytest.mark.django_db
def test_an_image_with_no_boxes_is_marked_blank(tmp_path, monkeypatch):
    _write_images(tmp_path, 1)
    monkeypatch.setattr(
        services, "get_detector", lambda device="cpu": _FakeDetector(per_image=0)
    )

    _run(tmp_path)

    image = Image.objects.get()
    assert image.is_blank
    assert image.status == Image.Status.PROCESSED


@pytest.mark.django_db
def test_an_unreadable_file_fails_alone(tmp_path, fake_detector):
    _write_images(tmp_path, 2)
    (tmp_path / "broken.jpg").write_bytes(b"not an image")

    events = _run(tmp_path)

    assert Image.objects.filter(status=Image.Status.FAILED).count() == 1
    assert Image.objects.filter(status=Image.Status.PROCESSED).count() == 2
    assert events[-1].failed == 1


@pytest.mark.django_db
def test_nothing_pending_yields_no_detect_events(tmp_path, fake_detector):
    _write_images(tmp_path, 2)
    _run(tmp_path)

    events = _run(tmp_path)

    assert not [e for e in events if e.phase is Phase.DETECT]


# --- EXIF capture time ------------------------------------------------------


@pytest.mark.django_db
def test_exif_capture_time_reaches_the_database(tmp_path, fake_detector):
    """Regression: read_captured_at raised inside a blanket except and returned
    None for every image ever ingested."""
    _write_images(tmp_path, 1, exif_time=lambda i: "2024:03:05 14:22:31")

    _run(tmp_path)

    captured = Image.objects.get().captured_at
    assert captured is not None
    assert (captured.year, captured.month, captured.day) == (2024, 3, 5)
    assert (captured.hour, captured.minute, captured.second) == (14, 22, 31)


def test_an_image_without_exif_has_no_capture_time(tmp_path):
    path = tmp_path / "plain.png"
    PILImage.new("RGB", (8, 8)).save(path)

    assert read_captured_at(path) is None


def test_a_malformed_exif_timestamp_is_not_fatal(tmp_path):
    path = tmp_path / "odd.jpg"
    img = PILImage.new("RGB", (8, 8))
    exif = img.getexif()
    exif[306] = "not a timestamp"
    img.save(path, exif=exif)

    assert read_captured_at(path) is None


# --- the CLI shells ---------------------------------------------------------


def _run_command(name, *args, **kwargs):
    out = StringIO()
    call_command(name, *args, stdout=out, **kwargs)
    return out.getvalue()


@pytest.mark.django_db
def test_ingest_command_reports_both_phases(tmp_path, fake_detector):
    _write_images(tmp_path, 3)

    output = _run_command("ingest", str(tmp_path), deployment=_deployment().id, batch_size=2)

    assert "Discovered 3 image(s), created 3 image(s)." in output
    assert "3/3 images processed" in output
    assert "Done. processed=3 failed=0" in output


@pytest.mark.django_db
def test_ingest_command_says_so_when_there_is_nothing_to_do(tmp_path, fake_detector):
    _write_images(tmp_path, 1)
    _run_command("ingest", str(tmp_path), deployment=_deployment().id)

    output = _run_command("ingest", str(tmp_path), deployment=_deployment().id)

    assert "No pending images to process" in output


@pytest.mark.django_db
def test_ingest_command_rejects_a_bad_path(tmp_path):
    # Previously this wrote to stderr and exited 0, so a typo in a script looked
    # like a successful run that found nothing.
    with pytest.raises(CommandError, match="Not a directory"):
        _run_command("ingest", str(tmp_path / "nope"), deployment=_deployment().id)
