import pytest

from src.retrieval.hybrid_search import normalize_semantic_scores, rerank_candidates


def test_normalization_and_hybrid_reranking():
    candidates = [
        {"image_id": "a", "clip_score": 0.9, "weather": "clear", "objects": []},
        {"image_id": "b", "clip_score": 0.8, "weather": "rainy", "objects": []},
    ]
    results = rerank_candidates(candidates, {"weather": "rainy", "objects": []}, alpha=0.5, beta=0.5, top_k=2)
    assert results[0]["image_id"] == "a"  # 0.5 semantic vs 0.5 metadata is tie-broken by CLIP.
    assert all(0 <= item["semantic_score_normalized"] <= 1 for item in results)


def test_constant_scores_normalize_safely():
    assert normalize_semantic_scores([0.3, 0.3]) == [1.0, 1.0]

