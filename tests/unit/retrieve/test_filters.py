from datetime import UTC, datetime

import pytest
from qdrant_client import models

from scholar_rag.core.errors import InvalidFilterError
from scholar_rag.retrieve.filters import translate_filter

WHITELIST = [
    "doc_id",
    "section",
    "doi",
    "journal",
    "title",
    "first_author",
    "year",
    "added_after",
]


def _cond(qfilter, index: int = 0) -> models.FieldCondition:
    assert isinstance(qfilter, models.Filter)
    assert qfilter.must is not None
    return qfilter.must[index]


def test_empty_dict_returns_none():
    assert translate_filter({}) is None


def test_doc_id_scalar():
    cond = _cond(translate_filter({"doc_id": "abc123"}))
    assert cond.key == "doc_id"
    assert cond.match.value == "abc123"


def test_doc_id_list():
    cond = _cond(translate_filter({"doc_id": ["a", "b"]}))
    assert cond.key == "doc_id"
    assert cond.match.any == ["a", "b"]


def test_section_scalar():
    cond = _cond(translate_filter({"section": "results"}))
    assert cond.key == "section"
    assert cond.match.value == "results"


def test_doi_scalar():
    cond = _cond(translate_filter({"doi": "10.1/x"}))
    assert cond.key == "doi"
    assert cond.match.value == "10.1/x"


@pytest.mark.parametrize(
    ("field", "target", "value"),
    [
        ("journal", "journal_norm", "Nature"),
        ("title", "title_norm", "Attention"),
        ("first_author", "first_author_norm", "Smith"),
    ],
)
def test_norm_fields_lower_scalar(field, target, value):
    cond = _cond(translate_filter({field: value}))
    assert cond.key == target
    assert cond.match.value == value.lower()


def test_journal_list_lower():
    cond = _cond(translate_filter({"journal": ["Nature", "TMLR"]}))
    assert cond.key == "journal_norm"
    assert cond.match.any == ["nature", "tmlr"]


def test_year_scalar_match():
    cond = _cond(translate_filter({"year": 2024}))
    assert cond.key == "year"
    assert cond.match.value == 2024


def test_year_range_both_bounds():
    cond = _cond(translate_filter({"year": {"gte": 2018, "lte": 2024}}))
    assert cond.key == "year"
    assert cond.range.gte == 2018
    assert cond.range.lte == 2024


def test_year_range_gte_only():
    cond = _cond(translate_filter({"year": {"gte": 2018}}))
    assert cond.key == "year"
    assert cond.range.gte == 2018
    assert cond.range.lte is None


def test_year_range_lte_only():
    cond = _cond(translate_filter({"year": {"lte": 2018}}))
    assert cond.key == "year"
    assert cond.range.gte is None
    assert cond.range.lte == 2018


def test_added_after_iso_z():
    cond = _cond(translate_filter({"added_after": "2026-01-01T00:00:00Z"}))
    assert cond.key == "added_ts"
    expected = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    assert cond.range.gte == expected


def test_added_after_naive_assumed_utc():
    cond = _cond(translate_filter({"added_after": "2026-01-01T00:00:00"}))
    assert cond.key == "added_ts"
    expected = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    assert cond.range.gte == expected


def test_multiple_fields_combined_in_must():
    qfilter = translate_filter({"journal": "TMLR", "year": 2024, "section": "results"})
    assert isinstance(qfilter, models.Filter)
    assert qfilter.must is not None
    assert len(qfilter.must) == 3
    assert sorted(cond.key for cond in qfilter.must) == ["journal_norm", "section", "year"]


def test_unknown_field_error_lists_whitelist():
    with pytest.raises(InvalidFilterError) as exc:
        translate_filter({"author": "x"})
    for field in WHITELIST:
        assert field in str(exc.value)


def test_unknown_field_error_mentions_offending_field():
    with pytest.raises(InvalidFilterError) as exc:
        translate_filter({"author": "x"})
    assert "author" in str(exc.value)


@pytest.mark.parametrize("value", ["2020", 2020.5, {"gte": "2018"}, {"gte": 2018, "foo": 1}])
def test_invalid_year_scalar_raises(value):
    with pytest.raises(InvalidFilterError):
        translate_filter({"year": value})


def test_year_list_raises():
    with pytest.raises(InvalidFilterError):
        translate_filter({"year": [2020, 2021]})


def test_added_after_invalid_date_raises():
    with pytest.raises(InvalidFilterError):
        translate_filter({"added_after": "not a date"})


def test_added_after_non_string_raises():
    with pytest.raises(InvalidFilterError):
        translate_filter({"added_after": 123456789.0})


def test_keyword_field_non_string_raises():
    with pytest.raises(InvalidFilterError):
        translate_filter({"section": 3})


def test_empty_list_becomes_empty_match_any():
    cond = _cond(translate_filter({"journal": []}))
    assert cond.key == "journal_norm"
    assert cond.match.any == []
