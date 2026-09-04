"""Orchestration for the video workflow.
Decode -> sample -> detect -> track -> measure motion -> persist. 

Frames cross from decoding to inference through a boundary rather
than a direct call, so in the future we can put a real concurrent 
pipeline behind it without reshaping anything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from celery import current_app
from celery.exceptions import OperationalError

from core.models import Detection, Image, Media, SpeciesClassification
from inference import get_detector              # registry.get_detector()
from video.models import Track, MotionSignal
from video.decode import decode_frames, probe
from video.sampling import sample_at_fps
from video.tracking import Tracker
from video.motion import MotionAccumulator
from video.bouts import (
    ActivityBudget,
    activity_budget,
    classify,
    deformation_signal,
    displacement_signal,
    find_bouts,
    union,
)
from video.frame import (
    RepresentativePicker, candidates_from_signals, encode_jpeg,
    get_selector, frame_at,
)

from django.db import transaction
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.cache import cache

import logging
import hashlib
import json
import math

logger = logging.getLogger(__name__)


class FrameQueue:
    """Hold a queue of frames from a PyAV producer
    
    The producer is a function that returns a generator of frames.
    The depth is the number of frames to hold in the queue.
    The frames are a work queue.

    enter: start the producer and return the queue.
    exit: stop the producer.
    iter: iterate over the frames.
    """
    def __init__(self, produce, depth=1):
        self._produce = produce
        self.depth = depth
        self._frames = None
    
    def __enter__(self):
        self._frames = self._produce()
        return self
    
    def __exit__(self, *exc_info):
        close = getattr(self._frames, 'close', None)
        if close is not None:
            close()
        return False
    
    def __iter__(self):
        return iter(self._frames)


@dataclass
class VideoProgress:
    frames: int
    tracks: int
    position: float         # seconds
    duration: float | None  # seconds
    
    @property
    def fraction(self):
        if not self.duration:
            return None
        return min(self.position / self.duration, 1.0)


def _record_media_info(media: Media):
    info = probe(media.path)
    Media.objects.filter(pk=media.pk).update(
        width=info.width,
        height=info.height,
        fps=info.fps,
        duration=info.duration,
        gop_size=info.gop_size,
    )
    return info

def _captured_at(media: Media, ts: float):
    return media.deployment.start_ts + timedelta(
        seconds=(media.start_offset_ts or 0.0) + ts
    )


def _save_representative(media: Media, track: Track, picker):
    """Write the chosen frame out as an Image carrying one Detection.

    Not a field on Track: shaped this way the existing species queue picks the
    track up with no idea it came from video, and the whole classification
    history lands in SpeciesClassification for free.
    """
    path = Path(settings.MEDIA_ROOT) / "tracks" / f"track_{track.id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(picker.encode())
    width, height = picker.size

    image = Image.objects.create(
        deployment=media.deployment,
        path=str(path),
        source=Image.Source.VIDEO_FRAME,
        width=width,
        height=height,
        status=Image.Status.PROCESSED,
        is_blank=False,
        captured_at=_captured_at(media, picker.ts),
    )
    track.rep_ts = picker.ts
    track.rep_detection = Detection.objects.create(
        image=image,
        category=Detection.Category.ANIMAL,
        confidence=picker.confidence,
        bbox_x1=picker.bbox[0],
        bbox_y1=picker.bbox[1],
        bbox_x2=picker.bbox[2],
        bbox_y2=picker.bbox[3],
    )
    track.save(update_fields=["rep_ts", "rep_detection"])


def _enqueue_classification():
    """Wake the species drain by task name instead of importing it: that queue
    belongs to the image workflow, and video must not depend on that package.

    A missed wake-up is not data loss — the detection stays PENDING and the next
    drain claims it — so a broker that is down must not fail a decode run.
    """
    try:
        current_app.send_task("image.tasks.classify_pending_task")
    except OperationalError as exc:
        logger.warning("could not enqueue species classification: %s", exc)


def _persist(media: Media, finished, samples, picker=None):
    with transaction.atomic():
        track = Track.objects.create(
            media=media,
            start_frame=finished.start_frame,
            end_frame=finished.end_frame,
            start_ts=finished.start_ts,
            end_ts=finished.end_ts,
            hits=finished.hits,
        )
        MotionSignal.objects.bulk_create(
            [MotionSignal(
                track=track,
                ts=sample.ts,
                displacement_bl_per_s=sample.displacement_bl_per_s,
                deformation_score=sample.deformation_score,
                confidence=sample.confidence,
                bbox_x1=sample.bbox[0],
                bbox_y1=sample.bbox[1],
                bbox_x2=sample.bbox[2],
                bbox_y2=sample.bbox[3],
            )
            for sample in samples]
        )
        if picker is not None and picker.chosen:
            _save_representative(media, track, picker)
            # After commit, or the worker can claim a row that is not there yet.
            transaction.on_commit(_enqueue_classification)
    return track

def process_media(media: Media, device='cpu', conf=None, track_fps=None,
                    keyframes_only=None, progress_every=100):
    """Decode one video into tracks and motion signals, yielding VideoProgress.

    A generator, so a CLI and a Celery task drive the identical run. Finished
    tracks are written as they retire rather than accumulated to the end: that
    bounds memory over a night of footage, and it is what makes per-media
    checkpointing possible at all.
    """
    # detector confidence threshold
    conf = settings.DETECTION_CONFIDENCE_THRESHOLD if conf is None else conf
    # targeted track fps
    track_fps = settings.VIDEO_TRACK_FPS if track_fps is None else track_fps
    # keyframes only mode
    keyframes_only = settings.VIDEO_KEYFRAMES_ONLY if keyframes_only is None else keyframes_only
    
    info = _record_media_info(media)                    # fetch media info
    frame_size = (info.width, info.height)
    detector = get_detector(device=device)              # MegaDetector singleton
    tracker = Tracker(
        max_age_seconds=settings.VIDEO_MAX_AGE_SECONDS,
        min_hits=settings.VIDEO_MIN_HITS,
        iou_threshold=settings.VIDEO_IOU_THRESHOLD,
    )

    accumulators = {}
    pickers = {}
    samples = defaultdict(list)
    frames = 0
    tracks = 0
    position = 0.0

    def produce():
        return sample_at_fps(
            decode_frames(media.path, media.id, keyframes_only=keyframes_only),
            track_fps,
        )
    
    Media.objects.filter(pk=media.pk).update(
        status=Media.Status.PROCESSING, error=""
    )
    try:
        with FrameQueue(produce, depth=settings.VIDEO_QUEUE_DEPTH) as queue:
            # FrameQueue.__enter__ -> self._frames = self._produce()
            for frame in queue: # FrameQueue.__iter__ -> self._frames
                frames += 1
                position = frame.timestamp

                # MegaDetector detects animals and return boxes
                detections = [
                    (detected.bbox, detected.confidence)
                    for detected in detector.detect_batch([frame.array], conf)[0]
                    if detected.category == "animal"
                ]

                # Track the frame
                for tracked in tracker.update(detections, frame.timestamp, frame.frame_index):
                    # collect motion signal
                    accumulator = accumulators.setdefault(
                        tracked.track_id, MotionAccumulator(frame_size)
                    )
                    # add active sample
                    sample = accumulator.add(
                        frame.timestamp, tracked.bbox, tracked.confidence, frame.array
                    )
                    if sample is not None:
                        samples[tracked.track_id].append(sample)
                    # Same pass as the motion signal: this frame is in hand once.
                    pickers.setdefault(
                        tracked.track_id, RepresentativePicker()
                    ).offer(frame.timestamp, tracked.bbox, tracked.confidence, frame.array)

                for finished in tracker.finished():
                    _persist(
                        media, finished,
                        samples.pop(finished.track_id, []),
                        pickers.pop(finished.track_id, None),
                    )
                    accumulators.pop(finished.track_id)
                    tracks += 1
                
                # Yield updating progress every `progress_every` frames
                if progress_every and frames % progress_every == 0:
                    yield VideoProgress(frames, tracks, position, info.duration)
            # FrameQueue.__exit__ -> close()

        for finished in tracker.flush():
            _persist(
                media, finished,
                samples.pop(finished.track_id, []),
                pickers.pop(finished.track_id, None),
            )
            accumulators.pop(finished.track_id)
            tracks += 1
    
    except Exception as e:
        Media.objects.filter(pk=media.pk).update(
            status=Media.Status.FAILED, error=str(e)
        )
        raise
    
    Media.objects.filter(pk=media.pk).update(status=Media.Status.PROCESSED)
    yield VideoProgress(frames, tracks, position, info.duration)
            
    
def pending_video_media():
    return Media.objects.filter(
        status=Media.Status.PENDING, kind=Media.Kind.VIDEO
    ).order_by('id')


# -- Video Registry --


VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".m2ts", ".webm"})


def _video_paths(folder):
    return sorted(
        path for path in folder.rglob("*")
        if path.suffix.lower() in VIDEO_SUFFIXES
    )

def register_directory(folder, deployment, limit=None, contiguous=False):
    """Create a pending Media row for every video file not already known.

    Returns (discovered, created). Each file is probed on the way in, so the
    catalogue can answer "how many hours of footage is this" before anything is
    decoded.

    ``contiguous`` asserts that the files are one recording the camera split up,
    in filename order; then each start_offset_ts is the sum of the durations
    before it, which is what lets a track be stitched across a file boundary
    later. It is off by default because inferring contiguity that is not there
    produces a silently wrong timeline, while 0 at least reads as "unknown".
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    discovered = 0
    created = 0

    for path in _video_paths(folder):
        discovered += 1
        resolved = str(path.resolve())

        if Media.objects.filter(path=resolved).exists():
            continue

        try:
            info = probe(resolved)
        except Exception as e:
            Media.objects.create(
                deployment=deployment, path=resolved, kind=Media.Kind.VIDEO,
                status=Media.Status.FAILED, error=f"probe: {e}",
            )
            created += 1
            continue
        
        Media.objects.create(
            deployment=deployment,
            path=resolved,
            kind=Media.Kind.VIDEO,
            start_offset_ts=0.0,
            width=info.width,
            height=info.height,
            fps=info.fps,
            duration=info.duration,
            gop_size=info.gop_size,
        )
        created += 1
        if limit and created >= limit:
            break

    if contiguous:
        # After the loop, not during: the layout depends on every file's
        # duration, including ones registered by an earlier run.
        relayout_contiguous(deployment)

    return discovered, created


def relayout_contiguous(deployment):
    """Recompute start_offset_ts as if this deployment's videos were one recording."""
    offset = 0.0
    updates = []
    for media in deployment.media.filter(kind=Media.Kind.VIDEO).order_by("path"):
        media.start_offset_ts = offset
        updates.append(media)
        offset += media.duration or 0.0
    Media.objects.bulk_update(updates, ["start_offset_ts"])
    return len(updates)


# -- Compute Bout --


@dataclass
class TrackActivity:
    track_id: int
    species_label: str | None
    start_ts: float
    end_ts: float
    bouts: list
    budget: ActivityBudget


@dataclass
class SensitivityRow:
    enter_threshold: float
    min_duration: float
    bouts: int
    active_seconds: float
    observed_seconds: float

    @property
    def active_fraction(self):
        if not self.observed_seconds:
            return None
        return self.active_seconds / self.observed_seconds


@dataclass
class Thresholds:
    displacement_enter_threshold: float
    displacement_exit_threshold: float
    deformation_enter_threshold: float
    deformation_exit_threshold: float
    min_duration: float
    max_gap: float


def _signals_by_track(tracks: list[Track]):
    grouped = defaultdict(list)
    signals = MotionSignal.objects.filter(
        track_id__in=[track.id for track in tracks]).order_by('track_id', 'ts')
    for signal in signals:
        grouped[signal.track_id].append(signal)

    return grouped

def labels_by_track(tracks: list[Track]):
    """Newest SpeciesNet label per track id, in one query.

    Newest rather than all: SpeciesClassification is append-only, so re-running
    the model leaves several rows per detection and only the latest reflects it.
    """
    latest = {}
    rows = SpeciesClassification.objects.filter(
        source=SpeciesClassification.Source.SPECIESNET,
        detection_id__in=[
            track.rep_detection_id for track in tracks if track.rep_detection_id
        ],
    ).order_by("detection_id", "-created_at")
    for row in rows:
        latest.setdefault(row.detection_id, row.category)

    return {track.id: latest.get(track.rep_detection_id) for track in tracks}

def _bouts_for_signals(signals: list[MotionSignal], thresholds: Thresholds):
    """Takes the signals rather than the Track: the segmentation never needs the
    row, and passing it invites re-querying inside a sensitivity sweep.

    The two channels are OR-ed because an animal grooming in place moves the
    deformation channel while displacement sits at the noise floor.
    """
    displacement = classify(
        [(signal.ts, displacement_signal(signal)) for signal in signals],
        thresholds.displacement_enter_threshold,
        thresholds.displacement_exit_threshold,
    )
    deformation = classify(
        [(signal.ts, deformation_signal(signal)) for signal in signals],
        thresholds.deformation_enter_threshold,
        thresholds.deformation_exit_threshold,
    )
    return find_bouts(
        union(displacement, deformation),
        thresholds.min_duration,
        thresholds.max_gap,
    )

def activity_report(tracks: list[Track], thresholds: Thresholds):
    """Bouts and an activity budget for every track, from the stored signals."""
    grouped = _signals_by_track(tracks)
    labels = labels_by_track(tracks)
    report = []
    for track in tracks:
        bouts = _bouts_for_signals(grouped[track.id], thresholds)
        report.append(TrackActivity(
            track_id=track.id,
            species_label=labels[track.id],
            start_ts=track.start_ts,
            end_ts=track.end_ts,
            bouts=bouts,
            budget=activity_budget(bouts),
        ))
    return report


def _sweep_point(base: Thresholds, enter: float, min_duration: float,
                 hysteresis_ratio: float):
    """One cell of the sweep: both channels move together so the table keeps a
    single ``enter`` column; everything else is inherited from the base."""
    exit_threshold = enter * hysteresis_ratio
    return Thresholds(
        displacement_enter_threshold=enter,
        displacement_exit_threshold=exit_threshold,
        deformation_enter_threshold=enter,
        deformation_exit_threshold=exit_threshold,
        min_duration=min_duration,
        max_gap=base.max_gap,
    )


def sensitivity_table(tracks: list[Track], base: Thresholds, enter_thresholds,
                      min_durations=(0.0,), hysteresis_ratio=0.5):
    """The figure S9.2 demands: bout count and active time against the threshold
    and the minimum bout duration.

    If active time moves by 3x across a plausible threshold range the output is
    not usable and that has to be fixed before anything is built on top of it. If
    it is stable, this table is itself the credibility argument.

    hysteresis_ratio is a convenience for sweeping one number: exit sits at that
    fraction of enter. It is not a tuned value.
    """
    grouped = _signals_by_track(tracks)
    rows = []
    for enter in enter_thresholds:
        for min_duration in min_durations:
            thresholds = _sweep_point(base, enter, min_duration, hysteresis_ratio)
            bouts = 0
            active = observed = 0.0
            for track in tracks:
                budget = activity_budget(
                    _bouts_for_signals(grouped[track.id], thresholds)
                )
                bouts += budget.bout_count
                active += budget.active_seconds
                observed += budget.active_seconds + budget.rest_seconds
            rows.append(SensitivityRow(enter, min_duration, bouts, active, observed))
    return rows


# -- Behavior Rep images --


def _fingerprint(track: Track, thresholds: Thresholds, selector_name: str):
    payload = json.dumps(
        asdict(thresholds) | {
            "track": track.id,
            "selector": selector_name,
            "width": settings.VIDEO_REPRESENTATIVE_WIDTH,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]

def _signals_in_bouts(signals, bouts, active: bool):
    spans = [(b.start_ts, b.end_ts) for b in bouts if b.active == active]
    return [
        signal for signal in signals
        if any(start <= signal.ts <= end for start, end in spans)
    ]

def _extract_behaviour_image(track, selector, signals):
    chosen = selector(candidates_from_signals(signals), track.media)
    if chosen is None:
        return None
    frame = frame_at(track.media, chosen.ts)
    return None if frame is None else encode_jpeg(frame.array)

def behaviour_images(track: Track, thresholds: Thresholds):
    """JPEG bytes for the active and the rest behaviour, keyed by {True, False}.

    Deliberately not persisted: which frame represents "moving" is a function of
    the thresholds, and MotionSignal's contract is that no threshold verdict is
    stored. The cache key is the thresholds, so a changed threshold misses
    instead of serving a frame that disagrees with the bouts beside it.
    """
    selector, selector_name = get_selector()
    signals = list(track.signals.order_by("ts"))
    bouts = _bouts_for_signals(signals, thresholds)
    fingerprint = _fingerprint(track, thresholds, selector_name)

    images = {}
    for active in (True, False):
        key = f"video:behaviour:{fingerprint}:{int(active)}"
        payload = cache.get(key)
        if payload is None:
            payload = _extract_behaviour_image(
                track, selector, _signals_in_bouts(signals, bouts, active)
            )
            if payload is not None:
                cache.set(key, payload, settings.VIDEO_BEHAVIOUR_IMAGE_TTL)
        images[active] = payload
    return images


# -- Bout Strips --


@dataclass
class BehaviourStrip:
    active: bool
    spans: list          # the black bars: [{"start_ts", "end_ts", "duration"}]
    total_seconds: float
    bout_count: int
    longest_seconds: float


@dataclass
class TrackTimeline:
    track_id: int
    span_start: float
    span_end: float
    tick_step: float
    strips: list


def _tick_step(span, target_ticks=8):
    """A round step near span/target_ticks, so the axis labels are readable."""
    if span <= 0:
        return 1.0
    raw = span / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 5):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return 10 * magnitude


def _strip(bouts, active: bool):
    spans = [b for b in bouts if b.active == active]
    return BehaviourStrip(
        active=active,
        spans=[
            {"start_ts": b.start_ts, "end_ts": b.end_ts, "duration": b.duration}
            for b in spans
        ],
        total_seconds=sum(b.duration for b in spans),
        bout_count=len(spans),
        longest_seconds=max((b.duration for b in spans), default=0.0),
    )


def track_timeline(track: Track, thresholds: Thresholds):
    """Two strips over one shared time axis: the frontend paints `spans` black
    and leaves the rest of the axis white."""
    bouts = _bouts_for_signals(list(track.signals.order_by("ts")), thresholds)
    span_start = bouts[0].start_ts if bouts else track.start_ts
    span_end = bouts[-1].end_ts if bouts else track.end_ts

    return TrackTimeline(
        track_id=track.id,
        span_start=span_start,
        span_end=span_end,
        tick_step=_tick_step(span_end - span_start),
        strips=[_strip(bouts, True), _strip(bouts, False)],
    )