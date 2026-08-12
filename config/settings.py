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
    "detections",
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

# Detection thresholds - the "filter blank frames" knob lives here.
# 0.5, not the 0.2 that suited the old ultralytics variant: RT-DETR is NMS-free
# and emits 300 queries per image, so its low-confidence tail is noise a YOLO+NMS
# pipeline never surfaced. On example_images, 0.2 yields 85 boxes against 12
# before, and every extra animal box is another SpeciesNet call.
DETECTION_CONFIDENCE_THRESHOLD = float(
    os.getenv("DETECTION_CONFIDENCE_THRESHOLD", "0.5")
)

# --- SpeciesNet / event pipeline config ---
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
# Gap larger than this (seconds) at one camera site starts a new event.
EVENT_GAP_SECONDS = int(os.getenv("EVENT_GAP_SECONDS", "1800"))

# --- Celery config ---
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")