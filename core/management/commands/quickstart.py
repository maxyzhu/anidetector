"""python manage.py quickstart <folder> - register, process and identify in one go.

Deliberately a sequence of call_command() rather than new logic: every step stays
independently runnable, which is what you want the moment one of them fails, and
this file cannot drift away from what the individual commands actually do.

It lives in core because it drives both workflows, and reaches them by command
name rather than by import — core may not import image or video (see
tests/test_boundries.py), and a string does not break that.
"""

from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import Deployment


class Command(BaseCommand):
    help = "Register a folder of video, process it, and identify the species."

    def add_arguments(self, parser):
        parser.add_argument("folder", type=str)
        parser.add_argument(
            "--deployment", type=int, default=None,
            help="Existing deployment id. One is created if you omit it.",
        )
        # Only used when creating a deployment. Camera id, location and country
        # are human knowledge, so the defaults are placeholders, not guesses.
        parser.add_argument("--camera-id", default="cam1")
        parser.add_argument("--location", default="unknown")
        parser.add_argument("--country", default="unknown")
        parser.add_argument(
            "--device", default=None,
            help="Torch device: 'auto', 'cpu', 'mps' or 'cuda'. Left to each command's own default when omitted.",
        )
        parser.add_argument(
            "--contiguous", action="store_true",
            help="The files are one recording the camera split up, in filename order.",
        )

    def handle(self, *args, **opts):
        deployment_id = opts["deployment"]
        if deployment_id is None:
            deployment = Deployment.objects.create(
                camera_id=opts["camera_id"],
                location=opts["location"],
                country=opts["country"],
                start_ts=timezone.now(),
            )
            deployment_id = deployment.id
            self._step(f"created deployment {deployment_id} — {deployment}")
            self.stdout.write(
                "  edit it in /admin; camera id and location are not inferable\n"
            )

        self._step("register_video")
        call_command(
            "register_video", opts["folder"],
            deployment=deployment_id, contiguous=opts["contiguous"],
        )

        # Passed through only when given, or it would override each command's own
        # default rather than leaving it to choose.
        device = {"device": opts["device"]} if opts["device"] else {}

        self._step("process_video")
        call_command("process_video", **device)

        self._step("classify_species")
        call_command("classify_species", **device)

        self.stdout.write(self.style.SUCCESS(
            "\nDone. http://127.0.0.1:8000/ lists what was registered, "
            "and /video/ shows the tracks."
        ))

    def _step(self, message):
        self.stdout.write(self.style.MIGRATE_HEADING(f"\n== {message}"))
