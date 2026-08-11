"""Tests for the inference package.

The purity test is the one that earns its keep: the moment inference/ imports
django.db the package can never be lifted out again, and nothing else would catch
that until someone tries to reuse it from a non-Django context.

Everything here is offline — no weights are loaded, no model classes constructed.
"""

import re
from pathlib import Path

import pytest

from inference import bbox_to_xywh_padded
from inference.classifier import _parse_votes
from inference.registry import (
    DEFAULT_DETECTOR_VERSION,
    DETECTOR_VARIANTS,
    PERMISSIVE_LICENCES,
    resolve_detector_variant,
)

PACKAGE = Path(__file__).resolve().parent.parent / "inference"
_DJANGO_IMPORT = re.compile(r"^\s*(?:from\s+django|import\s+django)", re.MULTILINE)


# --- the dependency-direction rule, as an executable check ---


def test_inference_does_not_import_django():
    offenders = sorted(
        str(p.relative_to(PACKAGE.parent))
        for p in PACKAGE.rglob("*.py")
        if _DJANGO_IMPORT.search(p.read_text())
    )
    assert not offenders, f"inference/ must stay Django-free, but: {offenders}"


def test_inference_is_not_an_installed_app():
    from django.conf import settings

    # It has no models, views or migrations; registering it would be a lie that
    # eventually justifies putting a model in it.
    assert "inference" not in settings.INSTALLED_APPS


# --- detector variant / licence policy ---


def test_default_variant_is_a_known_variant():
    assert DEFAULT_DETECTOR_VERSION in DETECTOR_VARIANTS


def test_unknown_variant_is_rejected_without_loading_anything():
    with pytest.raises(ValueError, match="Unknown detector variant"):
        resolve_detector_variant("MDV6-does-not-exist")


def test_a_permissively_licenced_variant_is_available():
    permissive = {
        v for v, (_loader, licence) in DETECTOR_VARIANTS.items()
        if licence in PERMISSIVE_LICENCES
    }
    # Step 1b of the refactor plan switches the default to one of these.
    assert "MDV6-apa-rtdetr-e" in permissive
    assert "MDV6-mit-yolov9-e" in permissive


# --- bbox padding (moved out of detections/utils.py) ---


def test_padding_of_one_is_identity():
    assert bbox_to_xywh_padded(0.2, 0.2, 0.6, 0.6, 1.0) == pytest.approx(
        [0.2, 0.2, 0.4, 0.4]
    )


def test_padding_grows_around_the_center():
    assert bbox_to_xywh_padded(0.4, 0.4, 0.6, 0.6, 1.5) == pytest.approx(
        [0.35, 0.35, 0.3, 0.3]
    )


def test_padding_a_full_frame_box_stays_in_bounds():
    assert bbox_to_xywh_padded(0.0, 0.0, 1.0, 1.0, 1.2) == pytest.approx(
        [0.0, 0.0, 1.0, 1.0]
    )


def test_padding_a_corner_box_shrinks_rather_than_spilling_out():
    x, y, w, h = bbox_to_xywh_padded(0.9, 0.9, 1.0, 1.0, 2.0)
    assert (x, y) == pytest.approx((0.85, 0.85))
    assert x + w <= 1.0 and y + h <= 1.0


# --- classifier result parsing ---


def test_parse_votes_pairs_classes_with_scores():
    votes = _parse_votes(
        {"classifications": {"classes": ["deer", "fox"], "scores": [0.8, 0.1]}}
    )
    assert [(v.label, v.score) for v in votes] == [("deer", 0.8), ("fox", 0.1)]


@pytest.mark.parametrize(
    "result",
    [None, {}, {"classifications": None}, {"classifications": {"classes": [], "scores": []}}],
)
def test_parse_votes_returns_empty_for_unusable_results(result):
    # An empty list is what tells classify_pending to mark the detection FAILED.
    assert _parse_votes(result) == []


def test_parse_votes_drops_classes_without_a_score():
    # zip() truncates; a class with no score is not a usable vote.
    votes = _parse_votes(
        {"classifications": {"classes": ["deer", "fox"], "scores": [0.8]}}
    )
    assert len(votes) == 1
