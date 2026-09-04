"""Species aggregation shared by the image and video workflows.

Pure string and dict handling: no models, no ORM, so a Track rollup and an Event
rollup can use the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

_RANKS = ("taxon_class", "order", "family", "genus", "species")
_LABEL_PARTS = 7
# SpeciesNet's UNKNOWN sentinel fills every rank with this instead of leaving it
# blank, which would otherwise read as a species-level identification.
_NOT_A_RANK = "no cv result"


@dataclass(frozen=True)
class Taxon:
    uuid: str
    taxon_class: str
    order: str
    family: str
    genus: str
    species: str
    common_name: str

    @property
    def ranks(self):
        return (self.taxon_class, self.order, self.family, self.genus, self.species)

    @property
    def rank(self):
        """Deepest rank actually filled in — what SpeciesNet rolled the prediction
        up to. None for a label carrying no taxonomy at all (blank, vehicle)."""
        for name, value in reversed(tuple(zip(_RANKS, self.ranks))):
            if value:
                return name
        return None

    @property
    def path(self):
        return ";".join(self.ranks)


def parse_taxon_path(label):
    """Split a SpeciesNet label into its ranks, or None if it is not one.

    Human annotations write free text into the same column, so a label that does
    not conform is expected rather than exceptional.
    """
    parts = str(label).split(";")
    if len(parts) != _LABEL_PARTS:
        return None
    uuid, *ranks, common_name = parts
    return Taxon(uuid, *("" if r == _NOT_A_RANK else r for r in ranks), common_name)


@dataclass
class SpeciesTally:
    label: str
    confidence: float
    count: int
    representative: object | None


def aggregate_votes(votes):
    """Collapse (label, confidence, representative) triples into one tally per label.

    Confidence is the max, not the mean: the question a reviewer asks is "show me
    the best evidence for this species", not "how sure were you on average".
    """
    tallies: dict[str, SpeciesTally] = {}
    for label, confidence, representative in votes:
        confidence = float(confidence or 0.0)
        tally = tallies.get(label)
        if tally is None:
            tallies[label] = SpeciesTally(label, confidence, 1, representative)
            continue
        tally.count += 1
        if confidence > tally.confidence:
            tally.confidence = confidence
            tally.representative = representative
    return tallies
