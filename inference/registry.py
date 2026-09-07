"""Model loading, caching and variant/licence policy — one place for all of it.

Why this file exists: model loading used to sit in two different modules
(one module for MegaDetector, another for SpeciesNet).
Splitting it per workflow is how two copies drift apart. Everything that decides
*which* weights get loaded now lives here.

Caches are plain module globals, so a Celery worker loads each model once and
reuses it across tasks. This module reads no Django settings on purpose — the
caller passes them in, which is what keeps the package importable without Django.

TODO(licence): ultralytics (AGPL-3.0) still gets imported whatever we do, because
PytorchWildlife's models/detection/__init__.py starts with
``from .ultralytics_based import *``. No AGPL code runs on our inference path any
more, but removing the dependency outright means dropping PytorchWildlife for
ONNX Runtime directly.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class DetectorVariant(NamedTuple):
    """How to load one MegaDetector variant, and the licence its code path carries."""

    loader: str        # PytorchWildlife class name
    licence: str


DETECTOR_VARIANTS = {
    "MDV6-apa-rtdetr-c": DetectorVariant("MegaDetectorV6Apache", "Apache-2.0"),
    "MDV6-apa-rtdetr-e": DetectorVariant("MegaDetectorV6Apache", "Apache-2.0"),
    # WARNING: in PytorchWildlife 1.3.0 YOLOMITBase.single_image_detection calls
    # _load_model() on every call, so these reload their weights once per image.
    # Fine for a smoke test, unusable for ingesting a directory.
    "MDV6-mit-yolov9-c": DetectorVariant("MegaDetectorV6MIT", "MIT"),
    "MDV6-mit-yolov9-e": DetectorVariant("MegaDetectorV6MIT", "MIT"),
}

PERMISSIVE_LICENCES = frozenset({"Apache-2.0", "MIT"})

# The ultralytics-backed variants (MDV6-yolov9-*, MDV6-yolov10-*, MDV6-rtdetr-c)
# are deliberately absent, and this table is the only way to reach a detector.
# They are AGPL-3.0, which is viral — and they lose on speed anyway: their
# `device` argument is a no-op in PytorchWildlife 1.3.0 (yolov8_base.py leaves
# `predictor.args.device` commented out with "Will uncomment later"), so they run
# on CPU whatever you ask for. Measured on an M5 Pro over example_images:
# 362 ms/img batched, against 84 ms/img for MDV6-apa-rtdetr-e on MPS.
_non_permissive = {
    name: variant.licence
    for name, variant in DETECTOR_VARIANTS.items()
    if variant.licence not in PERMISSIVE_LICENCES
}
if _non_permissive:
    # A plain raise, not an assert: assertions vanish under `python -O`, and a
    # licence guard that can be optimised away is not a guard.
    raise ImportError(
        f"inference.registry may only offer permissively licenced detector "
        f"variants, but found {_non_permissive}."
    )
del _non_permissive

DEFAULT_DETECTOR_VERSION = "MDV6-apa-rtdetr-e"

_detector = None
_classifier = None


def _resolve_device(requested):
    """CUDA -> MPS -> CPU for "auto"; anything else passes through.

    Private, and called only from the loaders below, because answering it needs
    torch. This package promises that importing it stays cheap and that torch
    arrives only when a model is genuinely loaded — 611 ms on every management
    command, and what lets the web process deploy with no torch installed at
    all. A public version would let a caller on the web path break both. Anyone
    who wants to know what was chosen reads ``.device`` off the loaded model.
    """
    # None reaches here from callers that let a setting decide and found nothing
    # set; treat it as auto rather than handing None to torch.
    if requested and requested != "auto":
        return requested

    import torch

    if torch.cuda.is_available():
        return "cuda"

    # is_available() is already False on non-Apple builds; it is the attribute
    # that is missing on older torch.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def resolve_detector_variant(version=None):
    """Validate a detector variant and return (version, variant, loader class)."""
    version = version or DEFAULT_DETECTOR_VERSION
    try:
        variant = DETECTOR_VARIANTS[version]
    except KeyError:
        raise ValueError(
            f"Unknown detector variant {version!r}. "
            f"Known variants: {', '.join(sorted(DETECTOR_VARIANTS))}."
        ) from None

    # Imported here, not at module scope, so importing this module stays cheap
    # and Django-free.
    from PytorchWildlife.models import detection as pw_detection

    loader = getattr(pw_detection, variant.loader, None)
    if loader is None:
        raise ValueError(
            f"PytorchWildlife has no {variant.loader}; the installed version is "
            f"too old for variant {version!r}."
        )
    return version, variant, loader


def get_detector(device="auto", version=None):
    """Cached Detector so repeated runs in one process reuse the loaded model.

    Lets a profiler measure model-load time once, separately from throughput.
    """
    global _detector
    if _detector is None:
        from inference.detector import Detector

        device = _resolve_device(device)
        logger.info("loading detector on %s", device)
        _detector = Detector(device=device, version=version)
    return _detector


def get_classifier(model, device="auto"):
    """Cached Classifier (SpeciesNet weights load on first call).

    ``model`` is passed in rather than read from Django settings, so this package
    keeps working outside the web app.
    """
    global _classifier
    if _classifier is None:
        from inference.classifier import Classifier

        device = _resolve_device(device)
        logger.info("loading classifier on %s", device)
        _classifier = Classifier(model, device=device)
    return _classifier


def reset_cache():
    """Drop cached models. For tests and for switching device mid-process."""
    global _detector, _classifier
    _detector = None
    _classifier = None
