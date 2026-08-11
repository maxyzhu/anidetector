"""Model loading, caching and variant/licence policy — one place for all of it.

Why this file exists: model loading used to sit in two different modules
(detections/inference.py for MegaDetector, detections/tasks.py for SpeciesNet).
Splitting it per workflow is how two copies drift apart. Everything that decides
*which* weights get loaded now lives here.

Caches are plain module globals, so a Celery worker loads each model once and
reuses it across tasks. This module reads no Django settings on purpose — the
caller passes them in, which is what keeps the package importable without Django.

TODO(licence): ultralytics (AGPL-3.0) is pulled in whatever we do, because
PytorchWildlife's models/detection/__init__.py starts with
``from .ultralytics_based import *``. Choosing a permissive variant below keeps
AGPL code off our inference path, but removing the dependency outright means
dropping PytorchWildlife for ONNX Runtime directly.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# version -> (PytorchWildlife loader class, licence of that code path).
# The loader class matters as much as the weights: the generic MegaDetectorV6
# runs on ultralytics, which is AGPL-3.0 and viral.
DETECTOR_VARIANTS = {
    "MDV6-apa-rtdetr-c": ("MegaDetectorV6Apache", "Apache-2.0"),
    "MDV6-apa-rtdetr-e": ("MegaDetectorV6Apache", "Apache-2.0"),
    "MDV6-mit-yolov9-c": ("MegaDetectorV6MIT", "MIT"),
    "MDV6-mit-yolov9-e": ("MegaDetectorV6MIT", "MIT"),
    # Legacy ultralytics-backed variants.
    "MDV6-yolov9-c": ("MegaDetectorV6", "AGPL-3.0"),
    "MDV6-yolov9-e": ("MegaDetectorV6", "AGPL-3.0"),
    "MDV6-yolov10-c": ("MegaDetectorV6", "AGPL-3.0"),
    "MDV6-yolov10-e": ("MegaDetectorV6", "AGPL-3.0"),
    "MDV6-rtdetr-c": ("MegaDetectorV6", "AGPL-3.0"),
}

PERMISSIVE_LICENCES = frozenset({"Apache-2.0", "MIT"})

# Unchanged from before the refactor so detections stay byte-identical. Step 1b
# of the refactor plan flips this to MDV6-apa-rtdetr-e, which does change output
# (different architecture and weights) and so needs its own commit.
DEFAULT_DETECTOR_VERSION = "MDV6-yolov9-c"

_detector = None
_classifier = None


def resolve_detector_variant(version=None):
    """Validate a detector variant and return (version, loader class, licence).

    Warns rather than raises on an AGPL variant: the project still defaults to
    one, and failing hard here would break the existing image pipeline.
    """
    version = version or DEFAULT_DETECTOR_VERSION
    try:
        loader_name, licence = DETECTOR_VARIANTS[version]
    except KeyError:
        raise ValueError(
            f"Unknown detector variant {version!r}. "
            f"Known variants: {', '.join(sorted(DETECTOR_VARIANTS))}."
        ) from None

    if licence not in PERMISSIVE_LICENCES:
        logger.warning(
            "Detector variant %s loads via %s under %s. AGPL-3.0 is viral; "
            "switch to MDV6-apa-rtdetr-e (Apache-2.0) or MDV6-mit-yolov9-e (MIT) "
            "before this project is distributed.",
            version, loader_name, licence,
        )

    # Imported here, not at module scope, so importing this module stays cheap
    # and Django-free.
    from PytorchWildlife.models import detection as pw_detection

    loader = getattr(pw_detection, loader_name, None)
    if loader is None:
        raise ValueError(
            f"PytorchWildlife has no {loader_name}; the installed version is too "
            f"old for variant {version!r}."
        )
    return version, loader, licence


def get_detector(device="cpu", version=None):
    """Cached Detector so repeated runs in one process reuse the loaded model.

    Lets a profiler measure model-load time once, separately from throughput.
    """
    global _detector
    if _detector is None:
        from inference.detector import Detector

        _detector = Detector(device=device, version=version)
    return _detector


def get_classifier(model, device=None):
    """Cached Classifier (SpeciesNet weights load on first call).

    ``model`` is passed in rather than read from Django settings, so this package
    keeps working outside the web app.
    """
    global _classifier
    if _classifier is None:
        from inference.classifier import Classifier

        _classifier = Classifier(model, device=device)
    return _classifier


def reset_cache():
    """Drop cached models. For tests and for switching device mid-process."""
    global _detector, _classifier
    _detector = None
    _classifier = None
