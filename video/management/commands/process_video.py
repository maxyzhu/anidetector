from __future__ import annotations

from django.core.management.base import BaseCommand

from core.models import Media
from video import services


class Command(BaseCommand):
    help = "Decode pending video media into tracks and motion signals."

    def add_arguments(self, parser):
        parser.add_argument(
            "--media", type=int, default=None,
            help="Process one Media id instead of everything pending.",
        )
        parser.add_argument(
            "--limit", type=int, default=None,
            help="Process at most this many pending videos.",
        )
        parser.add_argument(
            "--device", type=str, default="cpu",
            help="Torch device: 'cpu', 'mps', 'cuda' (default 'cpu').",
        )
        parser.add_argument(
            "--track-fps", type=float, default=None,
            help="Override settings.VIDEO_TRACK_FPS for this run. \
            Target track frames per second (default auto).",
        )
        parser.add_argument(
            "--async", action="store_true", default=False,
            help="Hand the work to Celery instead of running it here.",
        )
    
    def handle(self, *args, **opts):
        targets = self._targets(opts)

        if opts["async"]:
            from video.tasks import process_media_task

            for media in targets:
                process_media_task.delay(media.id, device=opts["device"])
            self.stdout.write(self.style.SUCCESS(f"Enqueued {len(targets)} video(s) to Celery."))
            return
        
        for media in targets:
            self.stdout.write(f"{media.path}")
            last = None
            for last in services.process_media(media, device=opts["device"], track_fps=opts["track_fps"]):
                self._write_progress(last)
            self.stdout.write("")
            if last is not None:
                self.stdout.write(self.style.SUCCESS(
                    f"Processed {last.frames} frames, {last.tracks} tracks.",
                ))
    
    def _targets(self, opts):
        if opts["media"] is not None:
            try:
                return [Media.objects.get(pk=opts["media"])]
            except Media.DoesNotExist:
                raise CommandError(f"Media #{opts['media']} not found.") from None
        
        queryset = services.pending_video_media()
        if opts["limit"] is not None:
            queryset = queryset[:opts["limit"]]
        return list(queryset)
    
    def _write_progress(self, progress):
        fraction = progress.fraction
        percent = f"{fraction * 100:5.1f}%" if fraction is not None else "  ?  "
        self.stdout.write(
            f"\r  {percent} {progress.position:8.1f}s "
            f"frames={progress.frames} tracks={progress.tracks}\x1b[K",
            ending="",
        )
        self.stdout.flush()
