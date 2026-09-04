"""Tests for video.decode against a synthetic clip with known properties."""

import pytest

from video.decode import decode_frames, frames_at, probe

from video_fixtures import write_moving_rectangle


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    positions = write_moving_rectangle(path, frames=100, fps=25, gop=10)
    return path, positions


def test_probe_reads_the_geometry_we_encoded(clip):
    path, _ = clip
    info = probe(path)

    assert (info.width, info.height) == (160, 120)
    assert info.fps == pytest.approx(25)
    assert info.duration == pytest.approx(4.0, abs=0.2)


def test_probe_recovers_the_keyframe_interval(clip):
    path, _ = clip
    assert probe(path).gop_size == 10


def test_timestamps_come_from_pts_and_advance_by_one_frame(clip):
    path, _ = clip
    frames = list(decode_frames(path, media_id="1"))

    assert len(frames) == 100
    assert frames[0].timestamp == pytest.approx(0.0)
    # 25fps -> 0.04s apart. This is the check that would fail if the code used
    # frame.index, or converted time_base to float too early.
    assert frames[1].timestamp - frames[0].timestamp == pytest.approx(0.04)
    assert frames[-1].timestamp == pytest.approx(99 * 0.04)


def test_frame_index_is_the_decode_position(clip):
    path, _ = clip
    frames = list(decode_frames(path, media_id="1"))

    assert [f.frame_index for f in frames[:5]] == [0, 1, 2, 3, 4]


def test_frames_arrive_as_rgb_arrays(clip):
    path, _ = clip
    frame = next(decode_frames(path, media_id="1"))

    assert frame.array.shape == (120, 160, 3)
    assert frame.array.dtype.name == "uint8"


def test_keyframes_only_yields_far_fewer_frames(clip):
    path, _ = clip
    every = list(decode_frames(path, media_id="1"))
    keyed = list(decode_frames(path, media_id="1", keyframes_only=True))

    # GOP 10, so roughly a tenth — the concrete version of "0.5fps at GOP=50".
    assert len(keyed) < len(every) / 5


def test_seeking_lands_at_or_after_the_requested_time(clip):
    path, _ = clip
    got = list(frames_at(path, [1.0, 2.5], media_id="1"))

    assert len(got) == 2
    assert got[0].timestamp >= 1.0
    assert got[1].timestamp >= 2.5
    assert got[0].timestamp < 1.1