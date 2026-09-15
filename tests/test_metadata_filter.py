import pytest

from src.retrieval.metadata_filter import metadata_matches, metadata_score


def test_metadata_score_soft_matching_and_aliases():
    candidate = {"weather": "rainy", "timeofday": "night", "scene": "city street", "objects": ["car", "person", "motor"]}
    query = {"weather": "rainy", "timeofday": "nighttime", "scene": None, "objects": ["person", "motorcycle"]}
    assert metadata_score(candidate, query) == pytest.approx(1.0)
    assert metadata_matches(candidate, query)


def test_no_conditions_score_zero():
    assert metadata_score({}, {"objects": []}) == 0.0

