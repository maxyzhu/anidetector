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

    class Meta:
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["is_blank"]),
        ]
    
    def __str__(self):
        return f"Image#{self.pk} {self.path} ({self.status})"
    

class Detection(models.Model):
    """Bounding box by MegaDetector"""

    class Category(models.TextChoices):
        PERSON = "person", "Person"
        VEHICLE = "vehicle", "Vehicle"
        ANIMAL = "animal", "Animal"
    
    image = models.ForeignKey(Image, related_name="detections", on_delete=models.CASCADE)
    category = models.CharField(max_length=16, choices=Category.choices)
    confidence = models.FloatField()

    bbox_x1 = models.FloatField()
    bbox_y1 = models.FloatField()
    bbox_x2 = models.FloatField()
    bbox_y2 = models.FloatField()

    class Meta:
        indexes = [
            models.Index(fields=["confidence"]),
            models.Index(fields=["category"]),
        ]
    
    def __str__(self):
        return f"{self.category} {self.confidence:.2f} on Image#{self.image_id}"
