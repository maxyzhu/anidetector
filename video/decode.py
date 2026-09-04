"""
Produce a sequence of `inference.types.Frame` from a video file.

Feed frames to PyAV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import av

from inference.types import Frame

logger = logging.getLogger(__name__)

@dataclass
class MediaInfo:
    width: int
    height: int
    fps: float | None
    duration: float | None
    gop_size: int | None


def probe(path, gop_sample_packets=300):
    with av.open(str(path)) as container: # PyAV container
        stream = container.streams.video[0] # extract video stream
        if stream is None:
            raise ValueError(f"No video stream found in {path}")
        
        info = MediaInfo(
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=float(stream.average_rate) if stream.average_rate else None,
            duration=_duration(container, stream),
            gop_size=None,
        )
        info.gop_size = _gop_size(container, stream, gop_sample_packets)
        return info

def _duration(container, stream):
    if stream.duration is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return None

def _gop_size(container, stream, max_packets):
    """GOP=Group of Pictures, count the gap between keyframes
    return the max gap between keyframes to consider the worst case

    NOTE: it is necessary because codec_context.gop_size often reads back as 0
    """
    positions = []
    for index, packet in enumerate(container.demux(stream)):
        if index > max_packets:
            break
        if packet.is_keyframe:
            positions.append(index)
    
    if len(positions) < 2:
        return None
    # return max gap between keyframes to consider the worst case
    return max(b - a for a, b in zip(positions, positions[1:]))


def decode_frames(path, media_id, keyframes_only=False):
    """Yield frames in presentation order.
    keyframes_only: only open when gop_size is within min animal passing time
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream is None:
            raise ValueError(f"No video stream found in {path}")
        
        # PyAV releases GIL while decoding, so frame-level threading is real parallelism.
        stream.thread_type = "AUTO"
        if keyframes_only:
            stream.codec_context.skip_frame = "NONKEY"
        
        time_base = stream.time_base
        previous_ts = None
        index = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                logger.warning("%s: dropping a frame with no pts", path)
                continue
            
            timestamp = float(frame.pts * time_base)

            if previous_ts is not None and timestamp < previous_ts:
                logger.warning(
                    "%s: timestamps went backwards (%.6f after %.6f)",
                    path, timestamp, previous_ts,
                )
            previous_ts = timestamp
        
            yield Frame(
                array=frame.to_ndarray(format="rgb24"), # C + SIMD optimized
                timestamp=timestamp,
                frame_index=index,
                media_id=media_id,
            )
            index += 1

    logger.info("%s: decoded %d frames", path, index)


def frames_at(path, timestamps, media_id):
    """Decode only the frames at or just after each timestamp.

    This is the traceability path: given a track's timestamps, jump straight back
    into the source video. Seeking lands on the keyframe at or before the target,
    so each hit needs a short decode forward — and keyframes_only must never be
    used here, since the point is to reach an arbitrary instant.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream is None:
            raise ValueError(f"No video stream found in {path}")
        
        stream.thread_type = "AUTO"
        time_base = stream.time_base

        for target in sorted(timestamps):
            container.seek(int(target / time_base), stream=stream)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                timestamp = float(frame.pts * time_base)
                if timestamp >= target:
                    yield Frame(
                        array=frame.to_ndarray(format="rgb24"),
                        timestamp=timestamp,
                        frame_index=-1, # not applicable for traceability
                        media_id=media_id,
                    )
                    break


        