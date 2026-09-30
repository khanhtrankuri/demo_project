from src.data.query_generator import generate_queries
from src.data.rich_caption import generate_scene_captions, generate_view_captions
from src.evaluation.semantic_retrieval import evaluate_rankings


def _record(sample, side="left", count=3, distance=12.0, camera="CAM_FRONT"):
    objects = [
        {"class": "car", "instance_token": f"{sample}-car-{i}", "spatial": side,
         "distance_m": distance, "distance_bin": "near", "attribute": "moving", "camera": camera}
        for i in range(count)
    ]
    objects.append({"class": "pedestrian", "instance_token": f"{sample}-ped", "spatial": side,
                    "distance_m": 8.0, "distance_bin": "near", "attribute": "standing", "camera": camera})
    view = {"image_id": sample + "-image", "camera": camera, "object_instances": objects,
            "object_counts": {"car": count, "pedestrian": 1}}
    return {"sample_token": sample, "views": {camera: view}}


def test_rich_captions_are_multiple_deterministic_and_grounded():
    record = _record("a")
    view = next(iter(record["views"].values()))
    first = generate_view_captions(view)
    assert first == generate_view_captions(view)
    assert 3 <= len(first) <= 5
    assert any("three cars" in text.lower() for text in first)
    assert any("left" in text.lower() for text in first)
    scene = generate_scene_captions(record)
    assert 3 <= len(scene) <= 5
    assert any("front" in text.lower() for text in scene)


def test_query_ground_truth_comes_from_predicates_not_caption_strings():
    records = [_record("left", "left"), _record("right", "right", count=1, distance=35.0)]
    queries = generate_queries(records)
    left = next(q for q in queries if q["category"] == "spatial" and q["predicate"].get("spatial") == "left"
                and q["predicate"].get("class") == "pedestrian")
    assert left["relevant_ids"] == ["left"]
    count = next(q for q in queries if q["category"] == "count" and q["predicate"].get("class") == "car")
    assert count["relevant_ids"] == ["left"]


def test_semantic_metrics_support_many_relevant_items_and_categories():
    queries = [
        {"query_id": "q1", "category": "object", "relevant_ids": ["a", "b"]},
        {"query_id": "q2", "category": "spatial", "relevant_ids": ["c"]},
    ]
    result = evaluate_rankings(queries, {"q1": ["x", "a", "b"], "q2": ["c", "x", "a"]})
    assert result["overall"]["recall_at_1"] == 0.5
    assert result["overall"]["recall_at_5"] == 1.0
    assert result["by_category"]["spatial"]["map"] == 1.0
    assert "caption equality" in result["relevance"]
