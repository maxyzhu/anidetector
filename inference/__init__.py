"""Model inference, independent of Django and of any particular workflow.

A plain Python package, not a Django app: no models, no views, no migrations, and
deliberately no `django` import anywhere inside it. That is what lets the image
workflow (detections/) and the video workflow (video/) share one copy of the
model-loading and inference code instead of growing two that drift apart.

Do not add this to INSTALLED_APPS.

Importing this package is cheap — torch, PytorchWildlife and speciesnet are only
imported when a model is actually loaded.
"""

from inference.classifier import (
    Classifier,
    bbox_to_xywh_padded,
    build_detections_dict,
    build_instances_dict,
)
from inference.detector import Detector
from inference.registry import (
    DEFAULT_DETECTOR_VERSION,
    DETECTOR_VARIANTS,
    get_classifier,
    get_detector,
    reset_cache,
    resolve_detector_variant,
)
from inference.sources import IMAGE_SUFFIXES, ImageDirectorySource, load_image_array
from inference.types import Crop, Frame, FrameSource, ParsedDetection, SpeciesVote

__all__ = [
    "Classifier",
    "Crop",
    "DEFAULT_DETECTOR_VERSION",
    "DETECTOR_VARIANTS",
    "Detector",
    "Frame",
    "FrameSource",
    "IMAGE_SUFFIXES",
    "ImageDirectorySource",
    "ParsedDetection",
    "SpeciesVote",
    "bbox_to_xywh_padded",
    "build_detections_dict",
    "build_instances_dict",
    "get_classifier",
    "get_detector",
    "load_image_array",
    "reset_cache",
    "resolve_detector_variant",
]
