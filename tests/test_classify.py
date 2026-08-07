"""Tests for the SpeciesNet classification stage.

These avoid the real model weights (a fake classifier is injected) and the real
broker (Celery runs eagerly), so they are fast and offline. They still exercise
the real PIL load + bbox padding + DB state machine.
"""

import pytest
from PIL import Image as PILImage

from detections import tasks
from detections.models import Detection, Image, SpeciesClassification


class _FakeClassifier:
    """Stand-in for SpeciesNetClassifier: no weights, canned predictions."""

    def preprocess(self, img, bboxes=None, resize=True):
        return {"bboxes": bboxes}  # opaque sentinel; batch_predict ignores it

    def batch_predict(self, filepaths, imgs):
        return [
            {
                "filepath": fp,
                "classifications": {
                    "classes": ["Odocoileus virginianus", "Vulpes vulpes"],
                    "scores": [0.83, 0.10],
                },
            }
            for fp in filepaths
        ]


@pytest.fixture
def fake_classifier(monkeypatch):
    clf = _FakeClassifier()
    monkeypatch.setattr(tasks, "get_classifier", lambda device=None: clf)
    return clf


@pytest.fixture
def animal_detection(tmp_path):
    """A processed, non-blank image with one animal detection above threshold."""
    path = tmp_path / "img.png"
    PILImage.new("RGB", (32, 32), (120, 120, 120)).save(path)
    image = Image.objects.create(
        path=str(path),
        width=32,
        height=32,
        status=Image.Status.PROCESSED,
        is_blank=False,
    )
    return Detection.objects.create(
        image=image,
        category=Detection.Category.ANIMAL,
        confidence=0.9,
        bbox_x1=0.2,
        bbox_y1=0.2,
        bbox_x2=0.6,
        bbox_y2=0.6,
    )


@pytest.mark.django_db
def test_classify_pending_writes_species(fake_classifier, animal_detection):
    processed, failed = tasks.classify_pending(batch_size=8)

    assert (processed, failed) == (1, 0)
    animal_detection.refresh_from_db()
    assert animal_detection.status == Detection.Status.PROCESSED

    sc = SpeciesClassification.objects.get(detection=animal_detection)
    assert sc.source == SpeciesClassification.Source.SPECIESNET
    assert sc.category == "Odocoileus virginianus"  # top-1
    assert sc.confidence == pytest.approx(0.83)
    assert len(sc.top_k) == 2


@pytest.mark.django_db
def test_below_threshold_is_not_claimed(fake_classifier, animal_detection):
    animal_detection.confidence = 0.01  # below SPECIES_CONF_THRESHOLD (0.2)
    animal_detection.save(update_fields=["confidence"])

    processed, failed = tasks.classify_pending(batch_size=8)

    assert (processed, failed) == (0, 0)
    animal_detection.refresh_from_db()
    assert animal_detection.status == Detection.Status.PENDING  # left untouched


@pytest.mark.django_db
def test_celery_task_runs_eagerly(fake_classifier, animal_detection):
    from config.celery import app

    app.conf.task_always_eager = True  # run inline, no broker needed
    app.conf.task_eager_propagates = True

    result = tasks.classify_pending_task.delay(batch_size=8)

    assert tuple(result.get()) == (1, 0)
    assert SpeciesClassification.objects.filter(detection=animal_detection).exists()
