from rest_framework import serializers

from .models import Image, Detection


class DetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Detection
        fields = ["id", "category", "confidence",
                "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    

class ImageSerializer(serializers.ModelSerializer):
    detections = DetectionSerializer(many=True, read_only=True)
    class Meta:
        model = Image
        fields = ["id", "path", "uploaded_at", "width", "height",
                "status", "is_blank", "detections"]
