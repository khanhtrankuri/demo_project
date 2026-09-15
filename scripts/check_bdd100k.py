#!/usr/bin/env python3
"""Audit a DatasetNinja/Supervisely-style BDD100K export without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_METADATA = ("weather", "timeofday", "scene")
PLACEHOLDER_METADATA_VALUES = {"undefined"}


def nonempty(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def inspect_image(path: Path, fully_decode: bool = False) -> tuple[str, str | None]:
    try:
        with Image.open(path) as image:
            if fully_decode:
                image.load()
            else:
                image.verify()
        return str(path), None
    except Exception as exc:  # Pillow exposes several format-specific exceptions.
        return str(path), f"{type(exc).__name__}: {exc}"


def inspect_annotation(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "error": None,
        "malformed": False,
        "metadata": {},
        "classes": [],
    }
    try:
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if not isinstance(record, dict):
        result["malformed"] = True
        result["error"] = f"top-level value is {type(record).__name__}, expected object"
        return result

    tags = record.get("tags")
    objects = record.get("objects")
    if not isinstance(tags, list) or not isinstance(objects, list):
        result["malformed"] = True
        result["error"] = "tags and/or objects is not a list"
        return result

    malformed_item = False
    for tag in tags:
        if not isinstance(tag, dict):
            malformed_item = True
            continue
        name = tag.get("name")
        if name in REQUIRED_METADATA and name not in result["metadata"]:
            result["metadata"][name] = tag.get("value")
    for obj in objects:
        if not isinstance(obj, dict):
            malformed_item = True
            continue
        class_title = obj.get("classTitle")
        if nonempty(class_title):
            result["classes"].append(str(class_title))
        else:
            malformed_item = True
    if malformed_item:
        result["malformed"] = True
        result["error"] = "one or more tag/object entries is malformed"
    return result


def stable_digest(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def audit(root: Path, sample_size: int, seed: int, workers: int) -> dict[str, Any]:
    root = root.resolve()
    split_dirs = [p for p in root.iterdir() if p.is_dir() and (p / "img").is_dir()]
    preferred = {"train": 0, "val": 1, "test": 2}
    split_dirs.sort(key=lambda p: (preferred.get(p.name.lower(), 99), p.name.lower()))
    if not split_dirs:
        raise SystemExit(f"No split/img directories found under {root}")

    report: dict[str, Any] = {
        "dataset_root": str(root),
        "format": "DatasetNinja/Supervisely-style per-image JSON sidecars",
        "seed": seed,
        "sample_size": sample_size,
        "splits": {},
        "totals": {},
        "distributions": {key: Counter() for key in REQUIRED_METADATA},
        "object_instances": Counter(),
        "errors": {
            "missing_annotations": [],
            "orphan_annotations": [],
            "broken_images": [],
            "invalid_json": [],
            "malformed_records": [],
            "duplicate_image_ids": {},
        },
    }

    all_images: list[tuple[str, Path]] = []
    all_annotations: list[tuple[str, Path]] = []
    image_id_locations: dict[str, list[str]] = defaultdict(list)
    annotation_for_image: dict[tuple[str, str], Path] = {}
    image_for_key: dict[tuple[str, str], Path] = {}

    for split_dir in split_dirs:
        split = split_dir.name
        img_dir = split_dir / "img"
        ann_dir = split_dir / "ann"
        images = sorted(
            (p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda p: str(p.relative_to(img_dir)).lower(),
        )
        annotations = sorted(ann_dir.rglob("*.json")) if ann_dir.is_dir() else []
        report["splits"][split] = {
            "image_dir": str(img_dir),
            "annotation_dir": str(ann_dir) if ann_dir.is_dir() else None,
            "images": len(images),
            "annotations": len(annotations),
        }
        for image in images:
            rel = str(image.relative_to(img_dir)).replace("\\", "/")
            key = (split, rel.lower())
            image_for_key[key] = image
            image_id_locations[image.stem.lower()].append(str(image))
            all_images.append((split, image))
        for annotation in annotations:
            rel = str(annotation.relative_to(ann_dir)).replace("\\", "/")
            image_rel = rel[:-5] if rel.lower().endswith(".json") else rel
            annotation_for_image[(split, image_rel.lower())] = annotation
            all_annotations.append((split, annotation))

    report["errors"]["duplicate_image_ids"] = {
        image_id: locations
        for image_id, locations in sorted(image_id_locations.items())
        if len(locations) > 1
    }
    for key, image in image_for_key.items():
        if key not in annotation_for_image:
            report["errors"]["missing_annotations"].append(str(image))
    for key, annotation in annotation_for_image.items():
        if key not in image_for_key:
            report["errors"]["orphan_annotations"].append(str(annotation))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        image_checks = list(pool.map(lambda item: inspect_image(item[1]), all_images))
        annotation_checks = list(pool.map(lambda item: inspect_annotation(item[1]), all_annotations))

    broken_set = {path for path, error in image_checks if error}
    report["errors"]["broken_images"] = [
        {"path": path, "error": error} for path, error in image_checks if error
    ]
    annotation_results = {item["path"]: item for item in annotation_checks}
    invalid_annotation_set: set[str] = set()
    for result in annotation_checks:
        if result["error"]:
            invalid_annotation_set.add(result["path"])
            target = "malformed_records" if result["malformed"] else "invalid_json"
            report["errors"][target].append(
                {"path": result["path"], "error": result["error"]}
            )
        for field in REQUIRED_METADATA:
            value = result["metadata"].get(field)
            report["distributions"][field][str(value) if nonempty(value) else "<missing>"] += 1
        report["object_instances"].update(result["classes"])

    eligible: list[dict[str, str]] = []
    missing_any = 0
    placeholder_any = 0
    missing_by_field = Counter()
    placeholder_by_field = Counter()
    valid_images = 0
    paired_valid_images = 0
    per_split_valid = Counter()
    per_split_eligible = Counter()
    per_split_missing = Counter()
    per_split_placeholder = Counter()
    for split, image in all_images:
        if str(image) not in broken_set:
            valid_images += 1
            per_split_valid[split] += 1
        rel = str(image.relative_to(root / split / "img")).replace("\\", "/")
        annotation = annotation_for_image.get((split, rel.lower()))
        if str(image) in broken_set or annotation is None:
            continue
        result = annotation_results.get(str(annotation))
        if not result or str(annotation) in invalid_annotation_set:
            continue
        paired_valid_images += 1
        missing_fields = [
            field for field in REQUIRED_METADATA if not nonempty(result["metadata"].get(field))
        ]
        if missing_fields:
            missing_any += 1
            per_split_missing[split] += 1
            missing_by_field.update(missing_fields)
        placeholder_fields = [
            field
            for field in REQUIRED_METADATA
            if str(result["metadata"].get(field)).strip().lower() in PLACEHOLDER_METADATA_VALUES
        ]
        if placeholder_fields:
            placeholder_any += 1
            per_split_placeholder[split] += 1
            placeholder_by_field.update(placeholder_fields)
        if not missing_fields and not placeholder_fields:
            per_split_eligible[split] += 1
            eligible.append(
                {"id": image.stem, "split": split, "image": str(image), "annotation": str(annotation)}
            )

    rng = random.Random(seed)
    random_samples = rng.sample(all_images, min(sample_size, len(all_images)))
    sample_checks = []
    for split, image in random_samples:
        rel = str(image.relative_to(root / split / "img")).replace("\\", "/")
        annotation = annotation_for_image.get((split, rel.lower()))
        _, decode_error = inspect_image(image, fully_decode=True)
        ann_result = annotation_results.get(str(annotation)) if annotation else None
        sample_checks.append(
            {
                "split": split,
                "image": str(image),
                "image_exists": image.is_file(),
                "fully_decodes": decode_error is None,
                "decode_error": decode_error,
                "annotation": str(annotation) if annotation else None,
                "filename_matches_sidecar": bool(annotation and annotation.name == image.name + ".json"),
                "json_valid": bool(ann_result and not ann_result["error"]),
                "metadata": ann_result["metadata"] if ann_result else {},
            }
        )

    report["totals"] = {
        "images": len(all_images),
        "annotations": len(all_annotations),
        "decodable_images": valid_images,
        "valid_image_annotation_pairs": paired_valid_images,
        "scenesearch_eligible_meaningful_metadata": len(eligible),
        "missing_any_required_metadata": missing_any,
        "missing_metadata_percentage_of_valid_pairs": round(
            100 * missing_any / paired_valid_images, 4
        ) if paired_valid_images else None,
        "missing_by_field": dict(missing_by_field),
        "placeholder_any_required_metadata": placeholder_any,
        "placeholder_metadata_percentage_of_valid_pairs": round(
            100 * placeholder_any / paired_valid_images, 4
        ) if paired_valid_images else None,
        "placeholder_by_field": dict(placeholder_by_field),
    }
    for split, split_report in report["splits"].items():
        split_report["decodable_images"] = per_split_valid[split]
        split_report["scenesearch_eligible_meaningful_metadata"] = per_split_eligible[split]
        split_report["missing_any_required_metadata"] = per_split_missing[split]
        split_report["placeholder_any_required_metadata"] = per_split_placeholder[split]

    report["distributions"] = {
        field: dict(counter.most_common()) for field, counter in report["distributions"].items()
    }
    report["top_20_object_classes_by_instance"] = report.pop("object_instances").most_common(20)
    report["random_sample_checks"] = sample_checks

    eligible.sort(key=lambda item: (item["split"].lower(), item["id"].lower()))
    eligible_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in eligible:
        eligible_by_split[item["split"].lower()].append(item)
    # Prefer the official train/val boundary. Independent RNGs keep either
    # selection stable if the other native split later changes.
    train_pool = eligible_by_split.get("train", [])
    val_pool = eligible_by_split.get("val", [])
    random.Random(seed).shuffle(train_pool)
    random.Random(seed).shuffle(val_pool)
    if len(train_pool) >= 8_000 and len(val_pool) >= 2_000:
        dev = train_pool[:8_000]
        evaluation = val_pool[:2_000]
        report["recommended_split"] = {
            "possible": True,
            "algorithm": "within each native split, sort eligible records by image id and shuffle with Python random.Random(42); take 8000 from train for development and 2000 from val for evaluation",
            "development_count": len(dev),
            "evaluation_count": len(evaluation),
            "development_sha256_of_ordered_ids": stable_digest([x["id"] for x in dev]),
            "evaluation_sha256_of_ordered_ids": stable_digest([x["id"] for x in evaluation]),
            "development_native_split_counts": dict(Counter(x["split"] for x in dev)),
            "evaluation_native_split_counts": dict(Counter(x["split"] for x in evaluation)),
            "development_first_10_ids": [x["id"] for x in dev[:10]],
            "evaluation_first_10_ids": [x["id"] for x in evaluation[:10]],
        }
    else:
        report["recommended_split"] = {
            "possible": False,
            "eligible_count": len(eligible),
            "eligible_train": len(train_pool),
            "eligible_val": len(val_pool),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, nargs="?",
        default=Path("bdd100k/bdd100k_-images-100k-DatasetNinja"),
        help="Dataset root containing train/val/test",
    )
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = audit(args.root, max(20, args.sample_size), args.seed, max(1, args.workers))
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
