"""
Calculate bout while querying the database.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Bout:
    start_ts: float
    end_ts: float
    active: bool

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts

@dataclass
class ActivityBudget:
    active_seconds: float
    rest_seconds: float
    bout_count: int # active bouts of non-zero length only


def displacement_signal(sample: MotionSignal):
    return sample.displacement_bl_per_s

def deformation_signal(sample: MotionSignal):
    return sample.deformation_score


def _tile(series_judgements: list[tuple[float, float]]):
    """Classify with hysteresis, group runs, then make the bouts abut."""
    bouts = []
    # Create Bouts
    for ts, active in series_judgements:
        if not bouts or bouts[-1].active != active:
            bouts.append(Bout(ts, ts, active))
    # Make the bouts abut to each other
    for first, second in zip(bouts, bouts[1:]):
        first.end_ts = second.start_ts
    bouts[-1].end_ts = series_judgements[-1][0]
    return bouts

def _bridge_gaps(bouts, max_gap):
    """Merge a brief rest that sits between two active bouts."""
    if max_gap <= 0:
        return bouts
    merged = []
    for bout in bouts:
        gap = merged[-1] if merged else None
        if (
            len(merged) >= 2
            and gap is not None
            and not gap.active
            and gap.duration < max_gap
            and bout.active
        ):
            merged.pop()
            before = merged.pop()
            bout = Bout(before.start_ts, bout.end_ts, True)
        merged.append(bout)
    return merged

def _absorb_short(bouts, min_duration):
    if min_duration <= 0:
        return bouts
    merged = []
    for bout in bouts:
        if merged and merged[-1].duration < min_duration:
            short = merged.pop()
            bout = Bout(short.start_ts, bout.end_ts, bout.active)
            # Removing one can leave two of the same state adjacent.
            if merged and merged[-1].active == bout.active:
                before = merged.pop()
                bout = Bout(before.start_ts, bout.end_ts, bout.active)
        merged.append(bout)

    # Check the last bout.
    while len(merged) > 1 and merged[-1].duration < min_duration:
        short = merged.pop()
        before = merged.pop()
        merged.append(Bout(before.start_ts, short.end_ts, before.active))
    return merged

def classify(series: tuple[float, float], enter_threshold: float, exit_threshold: float=None):
    """Judge a series of (ts, value) to a series of judgements (ts, active)"""
    if not series:
        return []
    if exit_threshold is None:
        exit_threshold = enter_threshold
    if enter_threshold is None:
        raise ValueError("enter_threshold must be provided")
    if exit_threshold > enter_threshold:
        raise ValueError(
            f"exit_threshold {exit_threshold} must not exceed enter_threshold "
            f"{enter_threshold}; hysteresis only works one way round."
        )

    series_judgements = []
    active = None
    for ts, value in series:
        if active is None or active == False:   # Enter threshold as inactive -> active
            active = value >= enter_threshold
        else:                                   # Exit threshold as active -> inactive
            active = value >= exit_threshold
        series_judgements.append((ts, active))
    return series_judgements

def union(*channels):
    """OR the per-sample judgements of several channels."""
    if not channels:
        return []

    timestamps = [ts for ts, _ in channels[0]]
    for other in channels[1:]:
        if [ts for ts, _ in other] != timestamps:
            raise ValueError("channels must be sampled at the same timestamps")
    return [
        (ts, any(channel[index][1] for channel in channels))
        for index, ts in enumerate(timestamps)
    ]

def find_bouts(series_judgements: tuple[float, float], min_duration=0.0, max_gap=0.0):
    """Split a series of (ts, active) judgements into alternating active and rest bouts.

    max_gap: bridges a short rest between two active bouts — the domain calls the
    equivalent idea a bout criterion interval. 
    
    min_duration: absorbs any remaining bout that is too short to mean anything. Gaps 
    are bridged first, because that rule is specific and min_duration is the catch-all.
    """
    if not series_judgements:
        return []
    bouts = _tile(series_judgements)
    bouts = _bridge_gaps(bouts, max_gap)
    return _absorb_short(bouts, min_duration)
    

def activity_budget(bouts):
    active = sum(b.duration for b in bouts if b.active)
    rest = sum(b.duration for b in bouts if not b.active)
    return ActivityBudget(
        active_seconds=active,
        rest_seconds=rest,
        # _tile ends the last bout at the last sample, so a track that turns
        # active on that sample leaves a zero-length bout. It tiles correctly and
        # contributes no seconds, but counting it as an episode of activity would
        # inflate every bout count by one whenever a track ends on a rising edge.
        bout_count=sum(1 for b in bouts if b.active and b.duration > 0),
    )