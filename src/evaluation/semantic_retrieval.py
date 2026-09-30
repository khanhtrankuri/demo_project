"""Ranking metrics for queries with annotation-derived relevance sets."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable


def _average_precision(ranked: list[str], relevant: set[str]) -> float:
    hits, score = 0, 0.0
    for rank, item in enumerate(ranked, 1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant) if relevant else 0.0


def _metrics(rows: list[tuple[list[str], set[str]]], ks=(1, 5, 10)) -> dict[str, float]:
    if not rows:
        return {**{f"recall_at_{k}": 0.0 for k in ks}, "mrr": 0.0, "map": 0.0, "ndcg_at_10": 0.0, "queries": 0}
    totals = defaultdict(float)
    for ranked, relevant in rows:
        if not relevant:
            raise ValueError("Every benchmark query needs at least one relevant item")
        for k in ks:
            totals[f"recall_at_{k}"] += bool(relevant.intersection(ranked[:k]))
        ranks = [i for i, item in enumerate(ranked, 1) if item in relevant]
        totals["mrr"] += 1.0 / min(ranks) if ranks else 0.0
        totals["map"] += _average_precision(ranked, relevant)
        dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(ranked[:10], 1) if item in relevant)
        ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(10, len(relevant)) + 1))
        totals["ndcg_at_10"] += dcg / ideal
    return {**{key: value / len(rows) for key, value in totals.items()}, "queries": len(rows)}


def evaluate_rankings(queries: Iterable[dict], rankings: dict[str, list[str]], ks=(1, 5, 10)) -> dict:
    by_category = defaultdict(list)
    all_rows = []
    for query in queries:
        query_id = query["query_id"]
        if query_id not in rankings:
            raise ValueError(f"Missing ranking for query {query_id}")
        row = (list(rankings[query_id]), set(query["relevant_ids"]))
        all_rows.append(row)
        by_category[query["category"]].append(row)
    return {
        "overall": _metrics(all_rows, ks),
        "by_category": {category: _metrics(rows, ks) for category, rows in sorted(by_category.items())},
        "relevance": "annotation predicates; caption equality is not used",
    }
