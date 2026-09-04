from django.db import models
from core.models import Media

class Track(models.Model):
    """One continuous observation.

    NOT an individual: an animal that walks out and comes back produces a second
    track, and nothing here links the two. v1 does no re-identification.
    """
    media = models.ForeignKey(Media, related_name="tracks", on_delete=models.CASCADE)
    # Position in the decoded sequence, NOT the video's frame number
    # one decode configuration; use the timestamps as the real index.
    start_frame = models.IntegerField()
    end_frame = models.IntegerField()
    # Seconds from the media start. Absolute time is derived through
    # deployment.start_ts + media.start_offset_ts; seconds are what seeks back
    # into the source video exactly.
    start_ts = models.FloatField()
    end_ts = models.FloatField()
    # Consecutive hits before a tentative track is confirmed.
    # Used in judging valid tracks when human-in-the-loop
    hits = models.IntegerField()

    # The representative frame, as a Detection on an Image: that is what lets the
    # existing species queue classify a track without knowing it came from video.
    # Species labels live in SpeciesClassification off this detection — appending
    # there keeps the model-version history a flattened column here would lose.
    rep_ts = models.FloatField(null=True, blank=True)
    rep_detection = models.ForeignKey(
        "core.Detection", related_name="+", null=True, blank=True,
        on_delete=models.SET_NULL,
    )

    class Meta:
        indexes = [models.Index(fields=["media", "start_ts"])]

    def __str__(self):
        return f"Track#{self.pk} @{self.start_ts:.1f}s w/{self.hits} hits"


class MotionSignal(models.Model):
    """One sampled motion measurement.

    Bouts are computed from these at query time; there is deliberately no Bout
    table. Thresholds will be argued about, and storing the verdict would mean
    re-decoding terabytes every time one changes.
    """
    track = models.ForeignKey(Track, related_name="signals", on_delete=models.CASCADE)
    ts = models.FloatField()
    # Body lengths per second: Scale invariant for displacement over the bbox diagonal. 
    # The same species near and far reads the same. Removes most of the per-camera threshold tuning.
    displacement_bl_per_s = models.FloatField()
    # Pixel change inside the bbox.
    deformation_score = models.FloatField()
    # Detector confidence for the box that matched here. Kept because the species
    # queue filters on it, and because it is otherwise thrown away at decode time.
    confidence = models.FloatField()
    # Bbox standard xyxy format
    bbox_x1 = models.FloatField()
    bbox_y1 = models.FloatField()
    bbox_x2 = models.FloatField()
    bbox_y2 = models.FloatField()

    class Meta:
        indexes = [models.Index(fields=['track', 'ts'])]