#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bdd100k import BDDRecord, find_dataset_root, image_is_decodable, iter_split_records
from src.utils.config import load_config, resolve_path


CSV_FIELDS = ["image_id", "image_path", "split", "weather", "timeofday", "scene", "objects", "num_objects"]


def select_valid(records: list[BDDRecord], count: int, seed: int) -> tuple[list[BDDRecord], list[dict[str, str]]]:
    pool = sorted(records, key=lambda row: row.image_id.lower())
    random.Random(seed).shuffle(pool)
    selected: list[BDDRecord] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in pool:
        key = record.image_id.lower()
        if key in seen:
            rejected.append({"image_id": record.image_id, "reason": "duplicate image id"})
            continue
        seen.add(key)
        valid, error = image_is_decodable(record.image_path)
        if not valid:
            rejected.append({"image_id": record.image_id, "reason": error or "image decode failed"})
            continue
        selected.append(record)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} valid records found; {count} required")
    return selected, rejected


def csv_row(record: BDDRecord, split: str) -> dict[str, object]:
    return {
        "image_id": record.image_id,
        "image_path": record.image_path,
        "split": split,
        "weather": record.weather,
        "timeofday": record.timeofday,
        "scene": record.scene,
        "objects": "|".join(record.objects),
        "num_objects": record.num_objects,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def distribution(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    return dict(Counter(str(row[field]) or "<missing>" for row in rows).most_common())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a deterministic 8K/2K BDD100K SceneSearch subset")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    configured_root = args.dataset_root or resolve_path(config, config["paths"]["dataset_root"])
    dataset_root = find_dataset_root(configured_root)
    output_dir = args.output_dir or resolve_path(config, config["paths"]["processed_dir"])
    output_dir = output_dir.resolve()
    seed = int(config["seed"])
    dev_size = int(config["dataset"]["dev_size"])
    test_size = int(config["dataset"]["test_size"])

    parsed: dict[str, list[BDDRecord]] = {}
    parse_errors: list[dict[str, str]] = []
    for split in ("train", "val"):
        try:
            candidates = list(iter_split_records(dataset_root, split, parse_errors))
        except Exception as exc:
            raise RuntimeError(f"Failed while parsing {split}: {exc}") from exc
        parsed[split] = [record for record in candidates if record.has_meaningful_metadata]
        print(f"{split}: parsed={len(candidates):,}, meaningful_metadata={len(parsed[split]):,}")

    if len(parsed.get("train", [])) >= dev_size and len(parsed.get("val", [])) >= test_size:
        dev, dev_rejected = select_valid(parsed["train"], dev_size, seed)
        test, test_rejected = select_valid(parsed["val"], test_size, seed)
        strategy = "official train/val boundary: 8,000 seeded train + 2,000 seeded val"
    else:
        combined = parsed.get("train", []) + parsed.get("val", [])
        selected, rejected = select_valid(combined, dev_size + test_size, seed)
        dev, test = selected[:dev_size], selected[dev_size:]
        dev_rejected, test_rejected = rejected, []
        strategy = "internal deterministic 80/20 split because native splits were insufficient"

    dev_rows = [csv_row(record, "dev") for record in dev]
    test_rows = [csv_row(record, "test") for record in test]
    all_rows = dev_rows + test_rows
    write_csv(output_dir / "metadata.csv", all_rows)
    write_csv(output_dir / "dev.csv", dev_rows)
    write_csv(output_dir / "test.csv", test_rows)

    object_counts: Counter[str] = Counter()
    for row in all_rows:
        object_counts.update(value for value in str(row["objects"]).split("|") if value)
    missing = sum(
        not all(str(row[field]).strip() for field in ("weather", "timeofday", "scene"))
        for row in all_rows
    )
    stats = {
        "dataset_root": str(dataset_root),
        "seed": seed,
        "strategy": strategy,
        "total_valid_images": len(all_rows),
        "dev_count": len(dev_rows),
        "test_count": len(test_rows),
        "weather_distribution": distribution(all_rows, "weather"),
        "timeofday_distribution": distribution(all_rows, "timeofday"),
        "scene_distribution": distribution(all_rows, "scene"),
        "top_object_categories_by_images": object_counts.most_common(20),
        "missing_metadata_count": missing,
        "missing_metadata_percentage": 100 * missing / len(all_rows) if all_rows else 0.0,
        "decode_or_duplicate_rejections": dev_rejected + test_rejected,
        "parse_errors": parse_errors,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
