"""Data contracts shared by everything in the inference layer.

These types are the seam between frame producers and models. A directory of
stills and a video decoder both hand over ``Frame`` objects, so the models never
learn which one they are looking at.

Nothing in this package may import Django — see tests/test_inference_purity.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np


@dataclass
class Frame:
    """One image handed to the models, whatever produced it.

    ``media_id`` identifies the source media: the resolved file path for a
    directory of stills, and the core.Media id once that model exists.
    """

    array: np.ndarray          # HWC, RGB
    timestamp: float           # seconds, relative to the media start
    frame_index: int
    media_id: str


@dataclass
class ParsedDetection:
    """One box normalized and translated."""

    category: str # "person", "vehicle", "animal"
    confidence: float
    # normalized and translated to 0-1
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """The box as one xyxy value, for callers that pass it around whole."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Crop:
    """A detection lifted out of its frame, on its way to the classifier.

    Declared here as part of the contract; nothing constructs it yet. Track-level
    frame selection (Step 4) is what will actually need to carry crops around.
    """

    media_id: str
    frame_index: int
    detection: ParsedDetection
    array: np.ndarray | None = None


@dataclass
class SpeciesVote:
    """One candidate species for a crop, as returned by the classifier."""

    label: str
    score: float
    # SpeciesNet labels already encode the taxonomy
    # ("uuid;class;order;family;genus;species;common name"); splitting them out
    # is a behaviour change, so it waits for core/taxonomy.py in Step 3.
    taxon_path: str = ""


class FrameSource(Protocol):
    """Anything that can produce frames in time order."""

    def __iter__(self) -> Iterator[Frame]: ...
