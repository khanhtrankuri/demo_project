#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.clip_encoder import CLIPEncoder
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.semantic_search import SemanticSearchEngine
from src.retrieval.vector_index import backend_name, read_index
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an end-to-end SceneSearch smoke test")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="all", choices=("all", "dev", "test"))
    args = parser.parse_args()
    config = load_config(args.config)
    processed = resolve_path(config, config["paths"]["processed_dir"])
    artifacts = resolve_path(config, config["paths"]["artifacts_dir"])
    metadata_path = processed / "metadata.csv"
    assert metadata_path.is_file(), f"Missing {metadata_path}"
    metadata = pd.read_csv(metadata_path)
    assert len(metadata) == 10_000, f"Expected 10,000 metadata rows, got {len(metadata)}"
    index_path = artifacts / "indexes" / f"{args.split}.faiss"
    mapping_path = artifacts / "mappings" / f"{args.split}.json"
    assert index_path.is_file(), f"Missing {index_path}"
    index = read_index(index_path)
    expected = 10_000 if args.split == "all" else int((metadata["split"] == args.split).sum())
    assert index.ntotal == expected, f"Expected ntotal={expected}, got {index.ntotal}"
    model_cfg = config["model"]
    encoder = CLIPEncoder(model_cfg["name"], model_cfg["pretrained"], mixed_precision=bool(config["runtime"].get("mixed_precision", True)))
    text_embedding = encoder.encode_text("pedestrian at night")
    assert text_embedding.shape[0] == 1
    semantic = SemanticSearchEngine(encoder, index_path, mapping_path)
    engine = HybridSearchEngine(semantic, config["hybrid"]["semantic_weight"], config["hybrid"]["metadata_weight"])
    results, info = engine.search("pedestrian at night", top_k=5, candidate_k=100)
    assert len(results) == 5
    assert all(Path(item["image_path"]).is_file() for item in results)
    print(f"PASS: index ntotal={index.ntotal}, backend={backend_name()}, latency={info['latency_ms']:.2f} ms")
    for item in results:
        print(f"#{item['rank']} {item['image_id']} final={item['final_score']:.4f} clip={item['clip_score_raw']:.4f} metadata={item['metadata_score']:.4f}")


if __name__ == "__main__":
    main()
