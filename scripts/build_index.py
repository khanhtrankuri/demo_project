#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.clip_encoder import CLIPEncoder
from src.retrieval.vector_index import FAISS_IMPORT_ERROR, backend_name, create_flat_ip, write_index
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized OpenCLIP embeddings and a FAISS IndexFlatIP")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    args = parser.parse_args()
    config = load_config(args.config)
    processed_dir = resolve_path(config, config["paths"]["processed_dir"])
    artifacts_dir = resolve_path(config, config["paths"]["artifacts_dir"])
    metadata_path = processed_dir / str(config.get("dataset", {}).get("metadata_file", "metadata.csv"))
    if not metadata_path.is_file():
        raise SystemExit(f"Missing {metadata_path}; run the configured dataset preparation script first")
    frame = pd.read_csv(metadata_path, keep_default_na=False)
    if args.split != "all":
        frame = frame[frame["split"] == args.split].reset_index(drop=True)
    if frame.empty:
        raise SystemExit(f"No records available for split={args.split}")

    model_cfg = config["model"]
    inference_pretrained = model_cfg.get("inference_pretrained", model_cfg["pretrained"])
    batch_size = args.batch_size or int(model_cfg["batch_size"])
    encoder = CLIPEncoder(
        model_name=model_cfg["name"],
        pretrained=inference_pretrained,
        device=args.device,
        mixed_precision=bool(config["runtime"].get("mixed_precision", True)),
    )
    print(f"model={encoder.model_name}")
    print(f"pretrained={encoder.pretrained}")
    print(f"device={encoder.device}")
    print(f"number_of_images={len(frame)}")
    print(f"batch_size={batch_size}")
    started = time.perf_counter()
    embeddings, valid_indices, errors = encoder.encode_images(frame["image_path"].tolist(), batch_size)
    embedding_seconds = time.perf_counter() - started
    if not len(embeddings):
        raise SystemExit("No image embeddings were produced")
    valid_frame = frame.iloc[valid_indices].reset_index(drop=True)
    mapping = []
    for row in valid_frame.to_dict(orient="records"):
        row["objects"] = [value for value in str(row.get("objects", "")).split("|") if value]
        row["num_objects"] = int(
            row.get("num_objects") or row.get("num_annotations") or len(row["objects"])
        )
        # The retrieval layer consumes one canonical schema for both BDD100K
        # and nuScenes while retaining all dataset-specific columns.
        row["weather"] = str(row.get("weather") or row.get("weather_tag") or "unknown")
        row["timeofday"] = str(row.get("timeofday") or row.get("timeofday_tag") or "unknown")
        row["scene"] = str(row.get("scene") or "unknown")
        mapping.append(row)

    index_started = time.perf_counter()
    index = create_flat_ip(int(embeddings.shape[1]))
    index.add(np.ascontiguousarray(embeddings, dtype="float32"))
    index_seconds = time.perf_counter() - index_started
    embeddings_dir = artifacts_dir / "embeddings"
    indexes_dir = artifacts_dir / "indexes"
    mappings_dir = artifacts_dir / "mappings"
    for directory in (embeddings_dir, indexes_dir, mappings_dir):
        directory.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_dir / f"{args.split}_embeddings.npy", embeddings)
    write_index(index, indexes_dir / f"{args.split}.faiss")
    (mappings_dir / f"{args.split}.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    if errors:
        (mappings_dir / f"{args.split}_errors.json").write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    print(f"embedding_dimension={embeddings.shape[1]}")
    print(f"embedding_time_seconds={embedding_seconds:.3f}")
    print(f"index_build_time_seconds={index_seconds:.3f}")
    print(f"indexed_images={index.ntotal}")
    print(f"skipped_images={len(errors)}")
    print(f"index_backend={backend_name()}")
    if FAISS_IMPORT_ERROR:
        print(f"faiss_unavailable_reason={FAISS_IMPORT_ERROR}")


if __name__ == "__main__":
    main()
