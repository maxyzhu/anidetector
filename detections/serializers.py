from rest_framework import serializers

from .models import Image, Detection, Event, EventSpecies, SpeciesClassification


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
        fields = ["id", "path", "uploaded_at", "width", "height",
                "status", "is_blank", "detections", "captured_at", "camera_site", "event"]


class EventSpeciesSerializer(serializers.ModelSerializer):
    class Meta:
        model = EventSpecies
        fields = ["category", "confidence", "detection_count", "representative_image"]


class EventSerializer(serializers.ModelSerializer):
    species = EventSpeciesSerializer(many=True, read_only=True)
    class Meta:
        model = Event
        fields = ["id", "camera_site", "start_time", "end_time", "image_count", "species"]


class SpeciesClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = SpeciesClassification
        fields = ["id", "detection", "source", "category", "confidence", 
                "top_k", "model_version", "annotator", "error"]
