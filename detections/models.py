from django.conf import settings
from django.db import models


class Image(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSED = "processed", "Processed"
        FAILED = "failed", "Failed"

    path = models.CharField(max_length=1024, unique=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)

    is_blank = models.BooleanField(default=False)
    error = models.TextField(blank=True, default="")

    # Metadata for event clustering; captured_at comes from EXIF, camera_site
    # is derived from the ingest folder for now (a CameraSite table may come later).
    captured_at = models.DateTimeField(null=True, blank=True)
    camera_site = models.CharField(max_length=255, blank=True, default="")
    event = models.ForeignKey(
        "Event", related_name="images", null=True, blank=True, on_delete=models.SET_NULL
    )

    class Meta:
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["is_blank"]),
            models.Index(fields=["camera_site", "captured_at"]),  # burst/event grouping
            models.Index(fields=["event"]),
        ]

    def __str__(self):
        return f"Image#{self.pk} {self.path} ({self.status})"


class Detection(models.Model):
    """Bounding box by MegaDetector (bbox stored as normalized xyxy, 0-1)."""

    class Category(models.TextChoices):
        PERSON = "person", "Person"
        VEHICLE = "vehicle", "Vehicle"
        ANIMAL = "animal", "Animal"

    class Status(models.TextChoices):
        # Species-classification stage status (distinct from Image ingest status).
        PENDING = "pending", "Pending"
        PROCESSED = "processed", "Processed"
        FAILED = "failed", "Failed"

    image = models.ForeignKey(Image, related_name="detections", on_delete=models.CASCADE)
    category = models.CharField(max_length=16, choices=Category.choices)
    confidence = models.FloatField()

    bbox_x1 = models.FloatField()
    bbox_y1 = models.FloatField()
    bbox_x2 = models.FloatField()
    bbox_y2 = models.FloatField()

    # Drives the async species-classification queue.
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)

    class Meta:
        indexes = [
            models.Index(fields=["confidence"]),
            models.Index(fields=["category"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self):
        return f"{self.category} {self.confidence:.2f} on Image#{self.image_id}"


class Event(models.Model):
    """Temporal grouping of images at a camera site; may contain multiple species."""

    camera_site = models.CharField(max_length=255, blank=True, default="")
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)
    image_count = models.PositiveIntegerField(default=0)  # denormalized for fast queries
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["camera_site", "start_time"])]

    def __str__(self):
        return f"Event#{self.pk} @{self.camera_site} ({self.image_count} imgs)"


class EventSpecies(models.Model):
    """One row per distinct species detected within an event (multi-species support)."""

    event = models.ForeignKey(Event, related_name="species", on_delete=models.CASCADE)
    category = models.CharField(max_length=255)  # species label (free-text, not the Detection enum)
    confidence = models.FloatField()  # aggregated (max/mean) over the event's classifications
    detection_count = models.PositiveIntegerField(default=0)
    representative_image = models.ForeignKey(
        Image, null=True, blank=True, on_delete=models.SET_NULL
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("event", "category")]

    def __str__(self):
        return f"{self.category} x{self.detection_count} in Event#{self.event_id}"


class SpeciesClassification(models.Model):
    """Append-only classification history for a detection (machine + human)."""

    class Source(models.TextChoices):
        SPECIESNET = "speciesnet", "SpeciesNet"
        HUMAN = "human", "Human"

    detection = models.ForeignKey(
        Detection, related_name="classifications", on_delete=models.CASCADE
    )
    source = models.CharField(max_length=16, choices=Source.choices, default=Source.SPECIESNET)
    category = models.CharField(max_length=255)  # species label; NOT the Detection.Category enum
    confidence = models.FloatField(null=True, blank=True)  # null for human annotation
    top_k = models.JSONField(default=list)  # [{"category": str, "confidence": float}, ...]
    model_version = models.CharField(max_length=64, blank=True, default="")
    annotator = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL
    )  # set when source=human
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["detection", "source"])]

    def __str__(self):
        return f"{self.source}:{self.category} on Detection#{self.detection_id}"
