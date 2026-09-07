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

# -- Torch device --
# "auto" is resolved inside inference's model loaders, at the moment a model is
# actually loaded, so nothing on the web path imports torch to answer it. Pin it
# here to override the probe — e.g. TORCH_DEVICE=cpu on a box whose GPU is busy.
# Auto rather than a hard "cpu" because the alternative is a run far slower than
# the hardware allows, silently: the detector forward pass is 97% of the loop.
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "auto")

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

# -- Tiled detection --
# The model resizes every input to 640x640 whatever its size, so a 2560x1440
# frame arrives 4x downsampled while a 1066x900 tile arrives 1.67x — the animal
# is 2.4x larger in the tile. On the pet set a sleeping cat scored 0.11-0.17
# whole-frame and 0.53-0.77 on 3x2 tiles: 90% recall against 0%.
# 2x2 is not enough (47%): wider tiles shrink the animal again.
VIDEO_TILE_GRID = os.getenv("VIDEO_TILE_GRID", "3x2")
VIDEO_TILE_OVERLAP = float(os.getenv("VIDEO_TILE_OVERLAP", "0.25"))
# Use NMS to filter out duplicate boxes.
VIDEO_TILE_NMS_IOU = float(os.getenv("VIDEO_TILE_NMS_IOU", "0.45"))
# Tile every Nth sampled frame; 0 disables tiling. Cost is strictly linear in
# tile count and batching does not help (78-82 ms per tile at batch 1 through
# 6), so frequency is the only lever: every frame is 6.6x, every 5th is 2.1x.
# A motionless animal does not need rediscovering more than once a second.
VIDEO_TILE_EVERY = int(os.getenv("VIDEO_TILE_EVERY", "5"))
# Whole-frame gate for the frames between tiled passes. Low enough to keep a
# still animal's track alive (it reads 0.11-0.16 there), and that is all it may
# do — starting a track still needs DETECTION_CONFIDENCE_THRESHOLD, so a
# reflection or a query-tail box cannot become one.
VIDEO_TRACK_MIN_CONFIDENCE = float(os.getenv("VIDEO_TRACK_MIN_CONFIDENCE", "0.10"))

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