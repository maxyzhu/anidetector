"""
Steps Controller + Keyframe Gateway.
"""

from __future__ import annotations
from fractions import Fraction


def sample_at_fps(frames, target_fps):
    """Yield frames at the given FPS.
    Using time as sample rate because frame timestamps 
    are not always evenly spaced, especially for Camtrap videos.

    NOTE: Empirical number of samples < target FPS * duration.
    """
    if not target_fps or target_fps <= 0:
        yield from frames
        return

    min_gap = 1 / Fraction(target_fps).limit_denominator()
    last_emitted = None
    for frame in frames:
        cur_time = Fraction(frame.timestamp).limit_denominator()
        if last_emitted is None or cur_time - last_emitted >= min_gap:
            last_emitted = cur_time
            yield frame


def expected_frame_count(duration, target_fps):
    """Estimated number of frames to be decoded. 
    Help visualize sampling progress.
    """
    if not duration or not target_fps or target_fps <= 0:
        return None
    
    return int(duration * target_fps)
