import os
from celery import Celery


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("anidetector")
# Read config from Django settings, only keys prefixed CELERY_.
app.config_from_object("django.conf:settings", namespace="CELERY")
# Auto-find tasks.py in every installed app.
app.autodiscover_tasks()