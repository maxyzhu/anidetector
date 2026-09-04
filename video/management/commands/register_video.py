"""python manage.py register_video <folder> --deployment <id>

A CLI shell around video.services.register_directory. Deployments are created
separately, in the admin or the shell: camera id, location and country code are
human knowledge, not something a directory tree carries.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from core.models import Deployment
from video.services import register_directory


class Command(BaseCommand):
    help = "Register a folder of video files against a deployment."

    def add_arguments(self, parser):
        parser.add_argument("folder", type=str)
        parser.add_argument(
            "--deployment", type=int, required=True,
            help="Deployment id these files belong to.",
        )
        parser.add_argument(
            "--contiguous", action="store_true",
            help="These files are one recording split by the camera, in filename "
                 "order. Sets start_offset_ts from the accumulated durations.",
        )
        parser.add_argument("--limit", type=int, default=None)

    def handle(self, *args, **opts):
        try:
            deployment = Deployment.objects.get(pk=opts["deployment"])
        except Deployment.DoesNotExist:
            raise CommandError(
                f"No Deployment with id {opts['deployment']}. "
                f"Create one in the admin or the shell first."
            ) from None

        try:
            discovered, created = register_directory(
                opts["folder"], deployment,
                limit=opts["limit"], contiguous=opts["contiguous"],
            )
        except NotADirectoryError as exc:
            raise CommandError(f"Not a directory: {exc}") from None

        self.stdout.write(
            self.style.SUCCESS(
                f"Discovered {discovered} video(s), registered {created} "
                f"against {deployment}."
            )
        )