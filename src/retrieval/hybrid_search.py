from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.query.query_parser import parse_query
from src.retrieval.metadata_filter import apply_metadata, requested_conditions


def normalize_semantic_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    values = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return [0.0] * len(scores)
    floor = float(values[finite].min())
    ceiling = float(values[finite].max())
    values = np.where(finite, values, floor)
    if ceiling - floor < 1e-12:
        return [1.0] * len(scores)
    return ((values - floor) / (ceiling - floor)).clip(0.0, 1.0).tolist()


def rerank_candidates(
    candidates: list[dict[str, Any]],
    parsed_query: dict[str, Any],
    alpha: float = 0.8,
    beta: float = 0.2,
    hard_filter: bool = False,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    enriched = apply_metadata(candidates, parsed_query, hard=hard_filter)
    normalized = normalize_semantic_scores([float(item["clip_score"]) for item in enriched])
    has_metadata = bool(requested_conditions(parsed_query))
    for item, semantic_score in zip(enriched, normalized):
        item["clip_score_raw"] = float(item["clip_score"])
        item["semantic_score_normalized"] = float(semantic_score)
        item["final_score"] = (
            alpha * semantic_score + beta * float(item["metadata_score"])
            if has_metadata
            else semantic_score
        )
    enriched.sort(key=lambda item: (-item["final_score"], -item["clip_score_raw"], str(item.get("image_id", ""))))
    for rank, item in enumerate(enriched[:top_k], start=1):
        item["rank"] = rank
    return enriched[:top_k]


class HybridSearchEngine:
    def __init__(self, semantic_engine: Any, alpha: float = 0.8, beta: float = 0.2) -> None:
        if alpha < 0 or beta < 0 or alpha + beta <= 0:
            raise ValueError("Hybrid weights must be non-negative with a positive sum")
        total = alpha + beta
        self.semantic_engine = semantic_engine
        self.alpha = alpha / total
        self.beta = beta / total

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 100,
        hard_filter: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        started = time.perf_counter()
        parsed = parse_query(query)
        if overrides:
            for key in ("weather", "timeofday", "scene", "objects"):
                value = overrides.get(key)
                if value not in (None, "", []):
                    parsed[key] = value if key == "objects" else value
        candidates, latency_ms = self.semantic_engine.search(query, max(candidate_k, top_k))
        results = rerank_candidates(candidates, parsed, self.alpha, self.beta, hard_filter, top_k)
        return results, {
            "latency_ms": (time.perf_counter() - started) * 1000,
            "semantic_latency_ms": latency_ms,
            "candidate_count": len(candidates),
            "parsed_query": parsed,
            "mode": "hybrid",
        }
