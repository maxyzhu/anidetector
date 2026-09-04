"""Tests for video registration."""

import pytest
from django.utils import timezone

from core.models import Deployment, Media
from video.services import register_directory

from video_fixtures import write_moving_rectangle


@pytest.fixture
def deployment(db):
    return Deployment.objects.create(
        camera_id="cam1", location="backyard", country="USA",
        start_ts=timezone.now(),
    )


def _clips(folder, count, seconds=2):
    for i in range(count):
        write_moving_rectangle(folder / f"clip{i:02d}.mp4", frames=seconds * 25, fps=25)


@pytest.mark.django_db
def test_every_video_file_gets_a_pending_row(tmp_path, deployment):
    _clips(tmp_path, 3)

    discovered, created = register_directory(tmp_path, deployment)

    assert (discovered, created) == (3, 3)
    assert Media.objects.filter(status=Media.Status.PENDING).count() == 3
    assert all(m.kind == Media.Kind.VIDEO for m in Media.objects.all())


@pytest.mark.django_db
def test_probing_at_registration_fills_the_catalogue(tmp_path, deployment):
    _clips(tmp_path, 1)

    register_directory(tmp_path, deployment)

    media = Media.objects.get()
    assert (media.width, media.height) == (160, 120)
    assert media.duration == pytest.approx(2.0, abs=0.2)
    assert media.gop_size is not None


@pytest.mark.django_db
def test_rerunning_registers_nothing_new(tmp_path, deployment):
    _clips(tmp_path, 2)
    register_directory(tmp_path, deployment)

    discovered, created = register_directory(tmp_path, deployment)

    assert (discovered, created) == (2, 0)
    assert Media.objects.count() == 2


@pytest.mark.django_db
def test_offsets_are_zero_unless_contiguity_is_asserted(tmp_path, deployment):
    """0 reads as unknown; a guessed offset would be a silently wrong timeline."""
    _clips(tmp_path, 3)

    register_directory(tmp_path, deployment)

    assert set(Media.objects.values_list("start_offset_ts", flat=True)) == {0.0}


@pytest.mark.django_db
def test_contiguous_files_are_laid_out_end_to_end(tmp_path, deployment):
    _clips(tmp_path, 3, seconds=2)

    register_directory(tmp_path, deployment, contiguous=True)

    offsets = list(
        Media.objects.order_by("path").values_list("start_offset_ts", flat=True)
    )
    assert offsets[0] == 0.0
    assert offsets[1] == pytest.approx(2.0, abs=0.2)
    assert offsets[2] == pytest.approx(4.0, abs=0.4)


@pytest.mark.django_db
def test_an_unreadable_file_is_recorded_not_fatal(tmp_path, deployment):
    _clips(tmp_path, 1)
    (tmp_path / "broken.mp4").write_bytes(b"not a video")

    discovered, created = register_directory(tmp_path, deployment)

    assert (discovered, created) == (2, 2)
    failed = Media.objects.get(status=Media.Status.FAILED)
    assert "probe:" in failed.error


@pytest.mark.django_db
def test_a_missing_directory_is_reported(tmp_path, deployment):
    with pytest.raises(NotADirectoryError):
        register_directory(tmp_path / "nope", deployment)


@pytest.mark.django_db
def test_non_video_files_are_ignored(tmp_path, deployment):
    _clips(tmp_path, 1)
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "photo.jpg").write_bytes(b"x")

    discovered, created = register_directory(tmp_path, deployment)

    assert (discovered, created) == (1, 1)