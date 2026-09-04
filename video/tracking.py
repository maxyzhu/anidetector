"""
SORT Algorithm.

Tracking is serially dependent in time — frame N's association needs frame N-1's
track state — so a single video cannot be tracked in parallel. The scaling axis
is across videos, not within one. Do not try to parallelise this loop.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from dataclasses import dataclass

# -- IoU and Hungarian Assignment --

def iou_matrix(boxes_a, boxes_b):
    """
    Compute the IoU of bboxes in predicted and ground truth boxes.
    Rows index > boxes_a.
    Columns index > boxes_b. 
    Boxes > xyxy.
    Intersection over Union (IoU) > confidence of overlap
                                    (area of overlap / area of union).
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    
    # Compute overlap
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    overlap = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)

    # Compute union
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - overlap

    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)


def associate(iou, iou_threshold):
    """Optimal assignment on IoU, then drop the pairs that are not really pairs.

    Returns (matches, unmatched_rows, unmatched_columns), matches as (row, column).

    `linear_sum_assignment` returns a globally optimal assignment, including pairing
    a track with a detection it does not overlap at all when nothing better is left.

    NOTE: IoU association assumes the same object overlaps itself between
    consecutive observations. Below roughly 5fps a walking animal clears its own
    box and every IoU is 0 — which is what VIDEO_TRACK_FPS exists to prevent, not
    something this function can fix.
    """
    rows, cols = iou.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    
    # The solver minimizes the cost matrix, negate it.
    # row_ind, col_ind are the indices of the matches.
    row_ind, col_ind = linear_sum_assignment(-iou)

    # Drop the pairs that are not really pairs.
    matches = [
        (int(r), int(c))
        for r, c in zip(row_ind, col_ind)
        if iou[r, c] >= iou_threshold
    ]
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    return (
        matches,
        [r for r in range(rows) if r not in matched_rows],
        [c for c in range(cols) if c not in matched_cols],
    )


# -- Kalman Filter --

# Prevent negative bboxes.
_MIN_SIZE = 1e-6
# Standard deviation of the position and velocity. Values follow DeepSORT.
_POSITION_STD = 1.0 / 20
# At birth velocity is unknown, so the prior has to cover a running animal.
_INITIAL_SPEED_BL_PER_S = 2.0
# How much speed may change between observations.
# NOTE: A knob needed to be adjusted to get the best results.
# NOTE: Associated with `exit_threshold` in `bouts.py`.
_ACCELERATION_BL_PER_S2 = 1.0
# Measurement mask to pick [center x, center y, aspect ratio, height] 
# out of the 7-dimensional state.
_OBSERVATION = np.eye(4, 7)


def _box_to_state(box):
    """Convert a bounding box to a state vector."""
    x1, y1, x2, y2 = box
    width = max(x2 - x1, _MIN_SIZE)
    height = max(y2 - y1, _MIN_SIZE)
    return np.array([
        (x1 + x2) / 2,   # center x
        (y1 + y2) / 2,   # center y
        width / height,  # aspect ratio
        height,          # height
    ])

def _state_to_box(mean):
    """Convert a state vector to a bounding box."""
    cx, cy, aspect, height = mean[:4]
    width = max(aspect * height, _MIN_SIZE)
    height = max(height, _MIN_SIZE)
    return (
        cx - width / 2,  # x1
        cy - height / 2, # y1
        cx + width / 2,  # x2
        cy + height / 2, # y2
    )


class KalmanBoxFilter:
    """Constant-velocity filter over one box.
    
    State is [center_x, center_y, aspect_ratio, height, 
                velocity_x, velocity_y, velocity_height].
    
    """
    # initial state from the first detection.
    def __init__(self, box):
        self.mean = np.zeros(7)
        self.mean[:4] = _box_to_state(box)

        height = self.mean[3]
        # Initial standard deviation for the state.
        std = np.array([
            2 * _POSITION_STD * height,
            2 * _POSITION_STD * height,
            1e-2,
            2 * _POSITION_STD * height,
            _INITIAL_SPEED_BL_PER_S * height,
            _INITIAL_SPEED_BL_PER_S * height,
            _INITIAL_SPEED_BL_PER_S * height,
        ])
        self.covariance = np.diag(std ** 2) # No covariance between dimensions.
        
    @property
    def box(self):
        return _state_to_box(self.mean)
    
    def predict(self, delta_t):
        """Predict the state at the next time step."""
        delta_t = max(delta_t, 1e-6)
        # make a new  7 x 7 predition matrix
        transition = np.eye(7) 
        transition[0, 4] = delta_t
        transition[1, 5] = delta_t
        transition[3, 6] = delta_t
        
        height = max(self.mean[3], _MIN_SIZE)
        std = np.array([
            _POSITION_STD * height * delta_t,           # center x
            _POSITION_STD * height * delta_t,           # center y
            1e-3,                                       # aspect ratio
            _POSITION_STD * height * delta_t,           # height
            _ACCELERATION_BL_PER_S2 * height * delta_t, # velocity x
            _ACCELERATION_BL_PER_S2 * height * delta_t, # velocity y
            _ACCELERATION_BL_PER_S2 * height * delta_t, # velocity height
        ])
        # Kalman noise, grows with time.
        # NOTE: A fully derived constant-velocity Q has dt^3/dt^4 cross terms.
        # We use a simple approximation.
        process_noise = np.diag(std ** 2)

        # Predict the state.
        self.mean = transition @ self.mean
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self.mean[3] = max(self.mean[3], _MIN_SIZE)

        return self.box
    

    def update(self, box):
        """Correct the state with an observed box."""
        height = max(self.mean[3], _MIN_SIZE)
        std = np.array([
            _POSITION_STD * height, # center x
            _POSITION_STD * height, # center y
            1e-1,                   # observed aspect ratio is less confident than prediction (1e-2)
            _POSITION_STD * height, # height
        ])
        measurement_noise = np.diag(std ** 2)

        projected_mean = _OBSERVATION @ self.mean
        projected_conv = (
            _OBSERVATION @ self.covariance @ _OBSERVATION.T + measurement_noise
        )
        # Kalman gain. (7,4) @ (4,4) = (7,4)
        # self.covariance bigger, gain bigger, more weight to the measurement.
        # projected_conv bigger, gain smaller, more weight to the prediction.
        gain = self.covariance @ _OBSERVATION.T @ np.linalg.inv(projected_conv)

        # Update the state.
        self.mean = self.mean + gain @ (_box_to_state(box) - projected_mean) # (7,4) @ (4,1) = (7,)
        self.covariance = self.covariance - gain @ projected_conv @ gain.T
        self.mean[3] = max(self.mean[3], _MIN_SIZE)

        return self.box


# -- Tracker --

@dataclass
class TrackedBox:
    track_id: int
    bbox: tuple[float, float, float, float]
    # Of the detection that last matched, not of the smoothed bbox: the filter
    # has no opinion about how sure the detector was.
    confidence: float

@dataclass
class FinishedTrack:
    track_id: int
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    hits: int


class _LiveTrack:
    def __init__(self, track_id, box, confidence, ts, frame_index):
        self.id = track_id
        self.filter = KalmanBoxFilter(box)
        self.confidence = confidence
        self.hits = 1
        self.confirmed = False
        self.start_ts = self.last_ts = ts
        self.start_frame = self.last_frame = frame_index
        self.time_since_update = 0.0
    
    def finish(self):
        return FinishedTrack(
            self.id, self.start_frame, self.last_frame, 
            self.start_ts, self.last_ts, self.hits
        )
    
class Tracker:
    """SORT: predict with a Kalman filter, associate by IoU, age out the losses.

    max_age is in seconds rather than frames on purpose. The physical question is
    how long an animal may be occluded before we give up, which is a time — and
    expressing it that way removes its coupling to the sampling rate, so changing
    VIDEO_TRACK_FPS does not invalidate the tuning.
    """
    def __init__(self, max_age_seconds=2.0, min_hits=3, iou_threshold=0.3):
        self.max_age_seconds = max_age_seconds
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks = []
        self._retired = []
        self._next_id = 1
        self._last_ts = None
    
    def update(self, detections, ts, frame_index):
        """Update the tracker with `(bbox, confidence)` pairs for one frame.

        Confidence is carried in rather than attached afterwards: the boxes that
        come back out are Kalman-smoothed, so there is nothing left to match a
        detection against once this returns.
        """
        delta_t = 0.0 if self._last_ts is None else ts - self._last_ts
        self._last_ts = ts

        boxes = [bbox for bbox, _ in detections]
        predicted = [track.filter.predict(delta_t) for track in self._tracks]
        matches, unmatched_tracks, unmatched_boxes = associate(
            iou_matrix(predicted, boxes), self.iou_threshold
        )

        for track_index, box_index in matches: # track_index matches predicted boxes
            track = self._tracks[track_index]
            bbox, confidence = detections[box_index]
            track.filter.update(bbox)
            track.confidence = confidence
            track.hits += 1
            track.time_since_update = 0.0
            track.last_ts = ts
            track.last_frame = frame_index
            if track.hits >= self.min_hits:
                track.confirmed = True

        for track_index in unmatched_tracks:
            self._tracks[track_index].time_since_update += delta_t

        for box_index in unmatched_boxes:
            bbox, confidence = detections[box_index]
            new_track = _LiveTrack(self._next_id, bbox, confidence, ts, frame_index)
            self._tracks.append(new_track)
            self._next_id += 1

        self._retire()

        return [
            TrackedBox(track.id, track.filter.box, track.confidence)
            for track in self._tracks
            if track.confirmed and track.time_since_update == 0.0 # feed MotionAcumulator
        ]
    
    def finished(self):
        """Tracks retired since the last call. Drained at the end so services can 
        write a night out incrementally - per-media checkpointing.
        """
        retired, self._retired = self._retired, []
        return retired
    

    def flush(self):
        """End of media: retire everything still alive. Call before `self.finished`"""
        for track in self._tracks:
            if track.confirmed:
                self._retired.append(track.finish())
        self._tracks = []
        return self.finished()
    

    def _retire(self):
        alive = []
        for track in self._tracks:
            if not track.confirmed:
                # One miss kills a tentative track.
                if track.time_since_update == 0.0:
                    alive.append(track)
            elif track.time_since_update <= self.max_age_seconds:
                alive.append(track)
            else:
                self._retired.append(track.finish())
        self._tracks = alive


