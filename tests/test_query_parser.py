from src.query.query_parser import parse_query


def test_query_parser_preserves_semantic_query_and_extracts_terms():
    parsed = parse_query("Pedestrian on rainy city street at night")
    assert parsed["semantic_query"] == "Pedestrian on rainy city street at night"
    assert parsed["weather"] == "rainy"
    assert parsed["timeofday"] == "nighttime"
    assert parsed["scene"] == "city street"
    assert parsed["objects"] == ["person"]


def test_synonyms():
    assert parse_query("bicycle and motorbike near cars")["objects"] == ["motorcycle", "bike", "car"]

