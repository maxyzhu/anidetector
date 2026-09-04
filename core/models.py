from django.conf import settings
from django.db import models


class Deployment(models.Model):
    """A camera at one location over one stretch of time — not a single file."""

    class Modality(models.TextChoices):
        RGB = "rgb", "RGB"
        THERMAL = "thermal", "Thermal"

    camera_id = models.CharField(max_length=64)
    location = models.CharField(max_length=255)
    country = models.CharField(max_length=64)
    lat = models.FloatField(null=True, blank=True)
    lon = models.FloatField(null=True, blank=True)
    start_ts = models.DateTimeField()
    end_ts = models.DateTimeField(null=True, blank=True)
    modality = models.CharField(
        max_length=16, choices=Modality.choices, default=Modality.RGB
    )

    class Meta:
        indexes = [models.Index(fields=["camera_id", "start_ts"])]

    def __str__(self):
        return f"{self.camera_id} @{self.location}"


class Media(models.Model):
    """One media file. Images and videos share this table: it is Camtrap DP's shape,
    so exporting to it later costs no schema change."""
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        PROCESSED = "processed", "Processed"
        FAILED = "failed", "Failed"

    class Kind(models.TextChoices):
        IMAGE = "image", "Image"
        VIDEO = "video", "Video"

    deployment = models.ForeignKey(
        Deployment, related_name="media", on_delete=models.CASCADE
    )
    path = models.CharField(max_length=1024, unique=True)
    kind = models.CharField(max_length=16, choices=Kind.choices)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # W/H to calculate absolute resolution
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.PENDING
    )
    error = models.TextField(blank=True, default="")

    # Video-specific fields
    # Seconds from the deployment start. Continuous recording is split into files
    # arbitrarily, so this is what keeps one animal's crossing on one timeline.
    start_offset_ts = models.FloatField(default=0)
    fps = models.FloatField(null=True, blank=True)
    # Keyframe interval; decoding with skip_frame="NONKEY" samples at fps/gop_size.
    gop_size = models.IntegerField(null=True, blank=True)
    duration = models.FloatField(null=True, blank=True)
    

    class Meta:
        indexes = [
            models.Index(fields=["deployment", "start_offset_ts"]),
            models.Index(fields=["status"]),
            # Tracks hang off media, so a reviewer finds a track by finding its
            # media first; this is the ordering the templates browse by.
            models.Index(fields=["uploaded_at"]),
        ]

    def __str__(self):
        return f"{self.kind}:{self.path}"


class Image(models.Model):
    """One still image. Separate from Media on purpose: Media carries a dozen
    video-only fields, and half a table of nulls is worse than two tables."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSED = "processed", "Processed"
        FAILED = "failed", "Failed"

    class Source(models.TextChoices):
        INGEST = "ingest", "Ingest"
        VIDEO_FRAME = "video_frame", "Video frame"

    deployment = models.ForeignKey(
        Deployment, related_name="images", on_delete=models.CASCADE
    )
    path = models.CharField(max_length=1024, unique=True)
    # A track's representative frame is an Image so it can carry a Detection and
    # go through the same species queue; this keeps it out of the photo gallery.
    source = models.CharField(
        max_length=16, choices=Source.choices, default=Source.INGEST
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)

    is_blank = models.BooleanField(default=False)
    error = models.TextField(blank=True, default="")

    captured_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["source", "status"]),
            models.Index(fields=["is_blank"]),
            models.Index(fields=["deployment", "captured_at"]),
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
        PROCESSING = "processing", "Processing"
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
