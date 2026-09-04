from rest_framework import serializers

from core.models import Image, Detection, SpeciesClassification


class DetectionSerializer(serializers.ModelSerializer):
    species = serializers.SerializerMethodField()
    def get_species(self, obj):
        rows = obj.classifications.all()
        return SpeciesClassificationSerializer(rows[0]).data if rows else None
    class Meta:
        model = Detection
        fields = ["id", "category", "confidence",
                "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "species"]
    

class ImageSerializer(serializers.ModelSerializer):
    detections = DetectionSerializer(many=True, read_only=True)
    class Meta:
        model = Image
        fields = ["id", "path", "uploaded_at", "width", "height", "status",
                "is_blank", "detections", "captured_at", "deployment"]


class SpeciesClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = SpeciesClassification
        fields = ["id", "detection", "source", "category", "confidence", 
                "top_k", "model_version", "annotator", "error"]
