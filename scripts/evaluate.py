#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.clip_encoder import CLIPEncoder
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.metadata_filter import metadata_matches
from src.retrieval.semantic_search import SemanticSearchEngine
from src.utils.config import load_config, resolve_path


def metrics(result_ids: list[str], relevant: set[str], latency_ms: float) -> dict[str, float | int]:
    def precision(k: int) -> float:
        viewed = result_ids[:k]
        return sum(item in relevant for item in viewed) / k if len(viewed) >= k else 0.0
    reciprocal_rank = next((1.0 / rank for rank, item in enumerate(result_ids, 1) if item in relevant), 0.0)
    recall10 = sum(item in relevant for item in result_ids[:10]) / len(relevant) if relevant else 0.0
    return {"precision_at_5": precision(5), "precision_at_10": precision(10), "mrr": reciprocal_rank, "recall_at_10": recall10, "latency_ms": latency_ms, "relevant_count": len(relevant)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic and hybrid retrieval against metadata ground truth")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="all", choices=("all", "dev", "test"))
    args = parser.parse_args()
    config = load_config(args.config)
    artifacts = resolve_path(config, config["paths"]["artifacts_dir"])
    model_cfg = config["model"]
    encoder = CLIPEncoder(model_cfg["name"], model_cfg["pretrained"], mixed_precision=bool(config["runtime"].get("mixed_precision", True)))
    semantic = SemanticSearchEngine(encoder, artifacts / "indexes" / f"{args.split}.faiss", artifacts / "mappings" / f"{args.split}.json")
    hybrid = HybridSearchEngine(semantic, config["hybrid"]["semantic_weight"], config["hybrid"]["metadata_weight"])
    encoder.encode_text("warmup")
    queries = json.loads(resolve_path(config, config["paths"]["benchmark_queries"]).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in queries:
        query, conditions = entry["query"], entry["conditions"]
        relevant = {str(item["image_id"]) for item in semantic.mapping if metadata_matches(item, conditions)}
        semantic_results, semantic_latency = semantic.search(query, 10)
        hybrid_results, hybrid_info = hybrid.search(query, top_k=10, candidate_k=int(config["retrieval"]["candidate_k"]), overrides=conditions)
        for mode, results, latency in (("semantic", semantic_results, semantic_latency), ("hybrid", hybrid_results, float(hybrid_info["latency_ms"]))):
            row = {"query": query, "mode": mode, **metrics([str(item["image_id"]) for item in results], relevant, latency)}
            rows.append(row)
            print(row)
    overall = []
    for mode in ("semantic", "hybrid"):
        selected = [row for row in rows if row["mode"] == mode]
        overall.append({"query": "__mean__", "mode": mode, **{key: statistics.mean(float(row[key]) for row in selected) for key in ("precision_at_5", "precision_at_10", "mrr", "recall_at_10", "latency_ms")}, "relevant_count": sum(int(row["relevant_count"]) for row in selected)})
    rows.extend(overall)
    reports = resolve_path(config, config["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    fields = ["query", "mode", "precision_at_5", "precision_at_10", "mrr", "recall_at_10", "latency_ms", "relevant_count"]
    with (reports / "baseline_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (reports / "baseline_metrics.json").write_text(json.dumps({"queries": rows[:-2], "summary": overall}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
