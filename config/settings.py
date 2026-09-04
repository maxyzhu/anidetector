import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only-change-me")
DEBUG = os.getenv("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["*"] if DEBUG else os.getenv(
    "DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1"
).split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "core",
    "image",
    "video",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("POSTGRES_DB", "anidetector"),
        "USER": os.getenv("POSTGRES_USER", "anidetector"),
        "PASSWORD": os.getenv("POSTGRES_PASSWORD", "anidetector"),
        "HOST": os.getenv("POSTGRES_HOST", "localhost"),
        "PORT": os.getenv("POSTGRES_PORT", 5432),
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 25,
}

# -- Detection --
# Detection thresholds - the "filter blank frames" knob lives here.
# 0.5, not the 0.2 that suited the old ultralytics variant: RT-DETR is NMS-free
# and emits 300 queries per image, so its low-confidence tail is noise a YOLO+NMS
# pipeline never surfaced. On example_images, 0.2 yields 85 boxes against 12
# before, and every extra animal box is another SpeciesNet call.
DETECTION_CONFIDENCE_THRESHOLD = float(
    os.getenv("DETECTION_CONFIDENCE_THRESHOLD", "0.5")
)

# -- SpeciesNet / event pipeline config --
# Second-stage species classifier (PyTorch backend, crop-based v4).
SPECIESNET_MODEL = os.getenv(
    "SPECIESNET_MODEL", "kaggle:google/speciesnet/pytorch/v4.0.3a/1"
)
# Min MegaDetector confidence for an animal box to be sent to the classifier.
# Distinct from DETECTION_CONFIDENCE_THRESHOLD (which filters blank frames).
SPECIES_CONF_THRESHOLD = float(os.getenv("SPECIES_CONF_THRESHOLD", "0.5"))
# Enlarge each bbox by this factor before cropping (1.1 = +10%), clamped to [0, 1].
CROP_PADDING = float(os.getenv("CROP_PADDING", "1.1"))
SPECIES_BATCH_SIZE = int(os.getenv("SPECIES_BATCH_SIZE", "16"))

# -- Video pipeline --
# Two independent FPS configs: 
# 1. Tracking needs higher rate for temporal density or IoU association.
# 2. Classification needs lower rate, we use N frames per track (NOT rate).
VIDEO_TRACK_FPS = float(os.getenv("VIDEO_TRACK_FPS", "5"))
VIDEO_FRAME_PER_TRACK = float(os.getenv("VIDEO_FRAME_PER_TRACK", "10"))
# Only decode keyframes when filtering empty frames.
VIDEO_KEYFRAMES_ONLY = os.getenv("VIDEO_KEYFRAMES_ONLY", "0") == "1"
# Frame offsets for a lost track to stay alive.
# Tightly coupled to VIDEO_TRACK_FPS: too low fragments tracks and inflates the count;
# too high merges different animals.
# TODO: fine-tune it with test datasets and then associate it with VIDEO_TRACK_FPS.
VIDEO_MAX_AGE_SECONDS = int(os. getenv("VIDEO_MAX_AGE_SECONDS", "1"))
# Consecutive hits before a tentative track is confirmed.
VIDEO_MIN_HITS = int(os. getenv("VIDEO_MIN_HITS", "3"))
# IoU threshold for tracking association.
VIDEO_IOU_THRESHOLD = float(os. getenv("VIDEO_IOU_THRESHOLD", "0.3"))
# Depth 1 is still a queue. The point is that the seam exists in v1, so M2 swaps
# the implementation instead of rewriting the pipeline.
VIDEO_QUEUE_DEPTH = int(os. getenv("VIDEO_QUEUE_DEPTH", "1"))

# -- Celery config --
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")


# -- Cache config --
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": os.getenv("CACHE_URL", "redis://localhost:6379/2"),
    }
}

# -- Web image quality config --
# The representative frame is both what the UI shows and what SpeciesNet crops
# from, so this width is the one quality knob for track classification: an animal
# filling 15% of the frame is ~192px here, against a ~480px model input.
VIDEO_REPRESENTATIVE_WIDTH = int(os.getenv("VIDEO_REPRESENTATIVE_WIDTH", "1280"))
VIDEO_REPRESENTATIVE_QUALITY = int(os.getenv("VIDEO_REPRESENTATIVE_QUALITY", "85"))
VIDEO_FRAME_SELECTOR = os.getenv("VIDEO_FRAME_SELECTOR", "area")
VIDEO_BEHAVIOUR_IMAGE_TTL = int(os.getenv("VIDEO_BEHAVIOUR_IMAGE_TTL", str(24 * 3600)))