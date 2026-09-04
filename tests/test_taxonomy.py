"""Tests for core.taxonomy — pure functions, no database."""

import pytest

from core.taxonomy import aggregate_votes, parse_taxon_path

# Real labels, copied from speciesnet.constants.Classification.
HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
UNKNOWN = (
    "f2efdae9-efb8-48fb-8a91-eccf79ab4ffb;no cv result;no cv result;no cv result;"
    "no cv result;no cv result;no cv result"
)
# What a prediction rolled up to family level looks like.
DEER_FAMILY = "aaaa;mammalia;cetartiodactyla;cervidae;;;deer family"


# --- parse_taxon_path -------------------------------------------------------


def test_a_species_level_label_splits_into_every_rank():
    taxon = parse_taxon_path(HUMAN)

    assert taxon.taxon_class == "mammalia"
    assert taxon.order == "primates"
    assert taxon.family == "hominidae"
    assert taxon.genus == "homo"
    assert taxon.species == "sapiens"
    assert taxon.common_name == "human"


def test_path_drops_the_uuid_and_common_name():
    assert parse_taxon_path(HUMAN).path == "mammalia;primates;hominidae;homo;sapiens"


@pytest.mark.parametrize(
    "label, expected",
    [
        (HUMAN, "species"),
        (DEER_FAMILY, "family"),
        ("aaaa;mammalia;cetartiodactyla;;;;deer order", "order"),
        ("aaaa;mammalia;;;;;mammal", "taxon_class"),
    ],
)
def test_rank_is_the_deepest_one_filled_in(label, expected):
    assert parse_taxon_path(label).rank == expected


def test_a_label_carrying_no_taxonomy_has_no_rank():
    assert parse_taxon_path(BLANK).rank is None


def test_the_unknown_sentinel_is_not_mistaken_for_a_species():
    # Every rank holds "no cv result", which would otherwise read as an
    # identification all the way down to species.
    taxon = parse_taxon_path(UNKNOWN)

    assert taxon.ranks == ("", "", "", "", "")
    assert taxon.rank is None


@pytest.mark.parametrize("label", ["", "roe deer", "mammalia;cervidae", "a;b;c;d;e;f;g;h"])
def test_free_text_is_not_a_taxon_path(label):
    # Human annotations share the column with SpeciesNet labels.
    assert parse_taxon_path(label) is None


# --- aggregate_votes --------------------------------------------------------


def test_repeated_votes_for_one_species_collapse_into_one_tally():
    tallies = aggregate_votes([
        ("deer", 0.4, "img1"),
        ("deer", 0.9, "img2"),
        ("deer", 0.7, "img3"),
    ])

    assert list(tallies) == ["deer"]
    tally = tallies["deer"]
    assert tally.count == 3
    assert tally.confidence == 0.9
    assert tally.representative == "img2"


def test_each_species_gets_its_own_tally():
    tallies = aggregate_votes([("deer", 0.9, "img1"), ("fox", 0.5, "img2")])

    assert set(tallies) == {"deer", "fox"}
    assert tallies["fox"].count == 1


def test_a_missing_confidence_counts_as_zero():
    tallies = aggregate_votes([("deer", None, "img1")])

    assert tallies["deer"].confidence == 0.0


def test_a_tie_keeps_the_first_representative():
    tallies = aggregate_votes([("deer", 0.8, "img1"), ("deer", 0.8, "img2")])

    assert tallies["deer"].representative == "img1"


def test_no_votes_produce_no_tallies():
    assert aggregate_votes([]) == {}
