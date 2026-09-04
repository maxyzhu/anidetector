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
from inference.detector import _clamp
from inference.registry import (
    DEFAULT_DETECTOR_VERSION,
    DETECTOR_VARIANTS,
    PERMISSIVE_LICENCES,
    resolve_detector_variant,
)

PACKAGE = Path(__file__).resolve().parent.parent / "inference"
_DJANGO_IMPORT = re.compile(r"^\s*(?:from\s+django|import\s+django)", re.MULTILINE)


# --- detector variant / licence policy ---


def test_default_variant_is_a_known_variant():
    assert DEFAULT_DETECTOR_VERSION in DETECTOR_VARIANTS


def test_unknown_variant_is_rejected_without_loading_anything():
    with pytest.raises(ValueError, match="Unknown detector variant"):
        resolve_detector_variant("MDV6-does-not-exist")


def test_no_variant_carries_a_copyleft_licence():
    # The table is the only way to reach a detector, so keeping AGPL out of it is
    # the whole guard. registry.py raises at import if this is ever violated;
    # this test says so out loud, where someone adding a variant will read it.
    offenders = {
        name: v.licence for name, v in DETECTOR_VARIANTS.items()
        if v.licence not in PERMISSIVE_LICENCES
    }
    assert not offenders


def test_the_ultralytics_variants_are_not_reachable():
    # AGPL-3.0, and their device argument is a no-op in PytorchWildlife 1.3.0.
    with pytest.raises(ValueError, match="Unknown detector variant"):
        resolve_detector_variant("MDV6-yolov9-c")


# --- box normalization ---


def test_clamp_keeps_coordinates_inside_the_frame():
    # RT-DETR does not clip its boxes; measured overshoot was 0.00027.
    assert _clamp(1.00027) == 1.0
    assert _clamp(-0.0004) == 0.0
    assert _clamp(0.42) == 0.42


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
