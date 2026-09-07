"""Server-rendered video results: the track gallery.

Thresholds stay query parameters here, same names as the API and the
activity_report command, so a URL from one is readable by the others.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, render
from django.urls import reverse

from core.models import Deployment, Media, SpeciesClassification
from core.taxonomy import parse_taxon_path
from video import services
from video.models import Track

# activity_report reads every MotionSignal of every track it is handed, so this
# is a memory bound, not a layout preference.
PAGE_SIZE = 24


def _common_name(label):
    """The stored label is SpeciesNet's whole taxonomy path. The API keeps it; a
    page has room for the common name. Human annotations are free text and do not
    parse, so they fall through unchanged."""
    if not label:
        return None
    taxon = parse_taxon_path(label)
    return taxon.common_name if taxon else label


@dataclass
class TrackCard:
    track: Track
    activity: services.TrackActivity
    bars: list  # active bouts as {left, width} percentages of the track span

    @property
    def filename(self):
        return Path(self.track.media.path).name

    @property
    def species(self):
        return _common_name(self.activity.species_label)


def _bars(activity):
    span = activity.end_ts - activity.start_ts
    if span <= 0:
        return []
    return [
        {
            "left": (bout.start_ts - activity.start_ts) / span * 100,
            "width": bout.duration / span * 100,
        }
        for bout in activity.bouts
        if bout.active
    ]


def _filtered(params):
    qs = Track.objects.select_related(
        "media", "media__deployment", "rep_detection"
    )

    deployment = params.get("deployment")
    if deployment:
        qs = qs.filter(media__deployment_id=deployment)

    media = params.get("media")
    if media:
        qs = qs.filter(media_id=media)

    species = params.get("species")
    if species:
        qs = qs.filter(
            rep_detection__classifications__source=(
                SpeciesClassification.Source.SPECIESNET
            ),
            rep_detection__classifications__category=species,
        )

    # The species join duplicates track rows; ordered because pagination over an
    # unordered queryset can repeat or drop rows between pages.
    return qs.distinct().order_by("media_id", "start_ts")


def _species_options():
    """The option value stays the stored label — that is what the filter matches
    on — while the text shown is the common name."""
    rep_ids = Track.objects.exclude(rep_detection=None).values("rep_detection_id")
    labels = (
        SpeciesClassification.objects.filter(
            source=SpeciesClassification.Source.SPECIESNET,
            detection_id__in=rep_ids,
        )
        .values_list("category", flat=True)
        .distinct()
    )
    options = [{"value": label, "name": _common_name(label)} for label in labels]
    return sorted(options, key=lambda option: option["name"])


def tracks_page(request):
    """GET /video/ — gallery of tracks with their activity budget.

    Query params: ?deployment= ?media= ?species= ?page=, plus every threshold
    the API takes (displacement_enter_threshold, min_duration, ...).
    """
    thresholds = services.thresholds_from(request.GET)
    page = Paginator(_filtered(request.GET), PAGE_SIZE).get_page(
        request.GET.get("page")
    )
    tracks = list(page.object_list)

    cards, error = [], None
    try:
        # activity_report preserves input order, so zip is the join.
        report = services.activity_report(tracks, thresholds)
        cards = [
            TrackCard(track, activity, _bars(activity))
            for track, activity in zip(tracks, report)
        ]
    except ValueError as exc:
        error = str(exc)

    # Pagination links have to carry the filters and thresholds forward.
    carried = request.GET.copy()
    carried.pop("page", None)

    return render(
        request,
        "video/tracks.html",
        {
            "page": page,
            "cards": cards,
            "error": error,
            "carried": carried.urlencode(),
            "deployments": Deployment.objects.filter(
                media__tracks__isnull=False
            ).distinct().order_by("camera_id"),
            "media_list": [
                {"id": media.id, "name": Path(media.path).name}
                for media in Media.objects.filter(
                    tracks__isnull=False
                ).distinct().order_by("path")
            ],
            "species_list": _species_options(),
            "f": request.GET,
            # So the form's placeholders cannot claim a default the page did not use.
            "defaults": services.THRESHOLD_DEFAULTS,
        },
    )


def _span_bars(spans, span_start, span):
    return [
        {
            "left": (span_["start_ts"] - span_start) / span * 100,
            "width": span_["duration"] / span * 100,
        }
        for span_ in spans
    ]


def _axis_ticks(timeline):
    """Round tick marks across the shared span, as percentages.

    Anchored to a multiple of tick_step rather than to span_start, or a track
    beginning at 39.8s would label its axis 39.8, 40.8, 41.8.
    """
    span = timeline.span_end - timeline.span_start
    if span <= 0:
        return []
    step = timeline.tick_step
    ticks, value = [], math.ceil(timeline.span_start / step) * step
    while value <= timeline.span_end:
        ticks.append({
            "label": value,
            "left": (value - timeline.span_start) / span * 100,
        })
        value += step
    return ticks


def _behaviour_views(track, timeline, query):
    """One entry per strip: the bars to paint and the frame that represents it.

    The image URL carries the same query string the page was rendered with, so
    the frame cannot disagree with the bars beside it — that is exactly what
    behaviour_images caches against.
    """
    span = timeline.span_end - timeline.span_start
    views = []
    for strip in timeline.strips:
        name = "active" if strip.active else "rest"
        url = reverse("track-behaviour-image", args=[track.id, name])
        views.append({
            "active": strip.active,
            "title": "Moving" if strip.active else "Resting",
            "strip": strip,
            "bars": _span_bars(strip.spans, timeline.span_start, span) if span > 0 else [],
            # No spans means no frames to choose from, so the endpoint would 404.
            "image": (f"{url}?{query}" if query else url) if strip.spans else None,
        })
    return views


def track_detail_page(request, pk):
    """GET /video/tracks/<pk>/ — one track's bout timeline and behaviour frames.

    Takes the same threshold query params as the list page and the API.
    """
    track = get_object_or_404(
        Track.objects.select_related(
            "media", "media__deployment", "rep_detection"
        ),
        pk=pk,
    )
    thresholds = services.thresholds_from(request.GET)

    timeline, behaviours, ticks, error = None, [], [], None
    try:
        timeline = services.track_timeline(track, thresholds)
        ticks = _axis_ticks(timeline)
        behaviours = _behaviour_views(track, timeline, request.GET.urlencode())
    except ValueError as exc:
        error = str(exc)

    return render(
        request,
        "video/track_detail.html",
        {
            "track": track,
            "species": _common_name(
                services.labels_by_track([track])[track.id]
            ),
            "timeline": timeline,
            "ticks": ticks,
            "behaviours": behaviours,
            "error": error,
            "defaults": services.THRESHOLD_DEFAULTS,
            "f": request.GET,
        },
    )
