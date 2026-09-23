#!/usr/bin/env python3
"""Prepare a validated, scene-separated nuScenes CAM_FRONT keyframe subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.nuscenes import (
    normalize_object,
    normalize_scene,
    normalize_timeofday,
    normalize_weather,
)


METADATA_TABLES = {
    "scene.json", "sample.json", "sample_data.json", "sensor.json",
    "calibrated_sensor.json", "sample_annotation.json", "category.json",
    "instance.json", "ego_pose.json", "log.json",
}
CSV_FIELDS = [
    "image_id", "image_path", "sample_token", "sample_data_token",
    "scene_token", "scene_name", "scene_description", "timestamp",
    "camera_channel", "is_key_frame", "location", "objects", "objects_raw",
    "num_annotations", "num_unique_objects", "weather_tag", "timeofday_tag",
    "weather", "timeofday", "scene", "num_objects", "source_split", "split",
]


@dataclass(frozen=True)
class ArchiveInfo:
    path: str
    format: str
    classification: str
    file_count: int
    sample_paths: list[str]
    metadata_tables: list[str]
    camera_counts: dict[str, int]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config


def configured_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def discover_archives(archive_dir: Path, pattern: str = "*.taz") -> list[Path]:
    return sorted((path for path in archive_dir.glob(pattern) if path.is_file()), key=lambda p: p.name.lower())


def no_archive_message(archive_dir: Path, pattern: str) -> str:
    message = f"No archives matching {pattern!r} found in {archive_dir.resolve()}."
    parent = archive_dir.parent
    near_dirs = [p for p in parent.iterdir() if p.is_dir() and p.name.lower() == archive_dir.name.lower()] if parent.exists() else []
    # Also flag common spelling mistakes and alternate compressed-tar suffixes.
    near_dirs.extend(p for p in parent.glob("nuScen*") if p.is_dir() and p not in near_dirs) if parent.exists() else None
    near_files: list[Path] = []
    for directory in near_dirs:
        near_files.extend(p for suffix in ("*.tgz", "*.tar.gz", "*.taz") for p in directory.glob(suffix))
    if near_files:
        rendered = ", ".join(str(p.resolve()) for p in sorted(set(near_files)))
        message += f" Similar files exist but do not match the configured directory/pattern: {rendered}"
    return message


def archive_signature(path: Path) -> str:
    with path.open("rb") as handle:
        head = handle.read(8)
    if head.startswith(b"\x1f\x8b"):
        return "tar.gz (gzip-compressed tar)"
    if head.startswith(b"BZh"):
        return "tar.bz2 (bzip2-compressed tar)"
    if head.startswith(b"\xfd7zXZ\x00"):
        return "tar.xz (xz-compressed tar)"
    if head.startswith(b"\x1f\x9d"):
        return "legacy UNIX compress (.Z), tar payload unconfirmed"
    return "tar (uncompressed or tar-compatible)"


def classify_member_names(names: Iterable[str]) -> tuple[str, list[str], dict[str, int]]:
    tables: set[str] = set()
    cameras: Counter[str] = Counter()
    for raw_name in names:
        normalized = raw_name.replace("\\", "/").lstrip("./")
        base = PurePosixPath(normalized).name
        if base in METADATA_TABLES:
            tables.add(base)
        parts = PurePosixPath(normalized).parts
        for index, part in enumerate(parts[:-1]):
            if part == "samples" and index + 1 < len(parts):
                channel = parts[index + 1]
                if channel.startswith("CAM_"):
                    cameras[channel] += 1
                break
    if tables:
        classification = "METADATA"
    elif cameras:
        classification = "KEYFRAME_BLOBS"
    else:
        classification = "UNKNOWN"
    return classification, sorted(tables), dict(sorted(cameras.items()))


def _legacy_z_hint(path: Path, exc: Exception) -> RuntimeError:
    tools = [name for name in ("7z", "7za", "bsdtar", "tar") if shutil.which(name)]
    suffix = f" Available local fallback tools: {', '.join(tools)}." if tools else " No suitable local fallback tool was found."
    return RuntimeError(f"Cannot read archive {path.resolve()}: {type(exc).__name__}: {exc}. File has legacy UNIX .Z signature.{suffix}")


def inspect_archive(path: Path, sample_limit: int = 20) -> ArchiveInfo:
    signature = archive_signature(path)
    names: list[str] = []
    sample: list[str] = []
    file_count = 0
    try:
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                file_count += 1
                names.append(member.name)
                if len(sample) < sample_limit:
                    sample.append(member.name)
    except (tarfile.TarError, OSError, EOFError) as exc:
        if signature.startswith("legacy UNIX"):
            raise _legacy_z_hint(path, exc) from exc
        raise RuntimeError(f"Cannot read archive {path.resolve()}: {type(exc).__name__}: {exc}") from exc
    classification, tables, cameras = classify_member_names(names)
    return ArchiveInfo(str(path.resolve()), signature, classification, file_count, sample, tables, cameras)


def safe_destination(root: Path, member_name: str) -> Path:
    # Tar paths are POSIX paths, even on Windows.
    pure = PurePosixPath(member_name.replace("\\", "/"))
    if pure.is_absolute() or pure.drive or ".." in pure.parts:
        raise ValueError(f"Unsafe archive member path: {member_name!r}")
    destination = root.joinpath(*pure.parts).resolve()
    resolved_root = root.resolve()
    try:
        destination.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Archive member escapes extraction root: {member_name!r}") from exc
    return destination


def extract_archive(path: Path, extract_root: Path, force: bool = False) -> dict[str, int]:
    counts = Counter(extracted=0, skipped=0, directories=0)
    extract_root.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive:
                destination = safe_destination(extract_root, member.name)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    counts["directories"] += 1
                    continue
                if not member.isfile():
                    raise ValueError(f"Refusing non-regular archive member {member.name!r} ({member.type!r})")
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read archive member {member.name!r} from {path}")
                if destination.exists() and not force:
                    identical = destination.is_file() and destination.stat().st_size == member.size
                    if identical:
                        with destination.open("rb") as existing:
                            while True:
                                old_chunk = existing.read(1024 * 1024)
                                new_chunk = source.read(1024 * 1024)
                                if old_chunk != new_chunk:
                                    identical = False
                                    break
                                if not old_chunk:
                                    break
                    if identical:
                        counts["skipped"] += 1
                        continue
                    raise FileExistsError(f"Existing file differs from archive member: {destination}. Use --force-extract to replace it.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".partial")
                try:
                    with temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    if temporary.stat().st_size != member.size:
                        raise OSError(f"Short extraction for {member.name!r}")
                    temporary.replace(destination)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                counts["extracted"] += 1
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise RuntimeError(f"Failed extracting {path.resolve()}: {type(exc).__name__}: {exc}") from exc
    return dict(counts)


def find_table(extract_root: Path, table: str) -> Path:
    matches = sorted(extract_root.rglob(table))
    if not matches:
        raise FileNotFoundError(f"Required nuScenes table not found under {extract_root}: {table}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple {table} files found under {extract_root}: {matches}")
    return matches[0]


def read_table(extract_root: Path, table: str) -> list[dict[str, Any]]:
    path = find_table(extract_root, table)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a list of objects in {path}")
    return value


def by_token(rows: Iterable[dict[str, Any]], table: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        token = row.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError(f"Missing token in {table}: {row}")
        if token in result:
            raise ValueError(f"Duplicate token {token!r} in {table}")
        result[token] = row
    return result


def simplify_category(name: str) -> str:
    return normalize_object(name)


def derive_tags(description: str) -> tuple[str, str]:
    return normalize_weather(description), normalize_timeofday(description)


def join_cam_front_records(extract_root: Path, camera: str = "CAM_FRONT", keyframes_only: bool = True) -> list[dict[str, Any]]:
    samples = by_token(read_table(extract_root, "sample.json"), "sample.json")
    scenes = by_token(read_table(extract_root, "scene.json"), "scene.json")
    sensors = by_token(read_table(extract_root, "sensor.json"), "sensor.json")
    calibrated = by_token(read_table(extract_root, "calibrated_sensor.json"), "calibrated_sensor.json")
    logs = by_token(read_table(extract_root, "log.json"), "log.json")
    annotations = read_table(extract_root, "sample_annotation.json")
    categories = by_token(read_table(extract_root, "category.json"), "category.json")
    instances = by_token(read_table(extract_root, "instance.json"), "instance.json")

    annotations_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_sample[str(annotation.get("sample_token", ""))].append(annotation)

    result: list[dict[str, Any]] = []
    for sample_data in read_table(extract_root, "sample_data.json"):
        calibration = calibrated.get(str(sample_data.get("calibrated_sensor_token", "")))
        sensor = sensors.get(str(calibration.get("sensor_token", ""))) if calibration else None
        if not sensor or sensor.get("channel") != camera:
            continue
        is_key_frame = bool(sample_data.get("is_key_frame"))
        if keyframes_only and not is_key_frame:
            continue
        sample_token = str(sample_data.get("sample_token", ""))
        sample = samples.get(sample_token)
        if sample is None:
            continue
        scene_token = str(sample.get("scene_token", ""))
        scene = scenes.get(scene_token)
        if scene is None:
            continue
        log = logs.get(str(scene.get("log_token", "")), {})
        raw: list[str] = []
        instance_tokens: set[str] = set()
        for annotation in annotations_by_sample.get(sample_token, []):
            instance_token = str(annotation.get("instance_token", ""))
            instance = instances.get(instance_token)
            category = categories.get(str(instance.get("category_token", ""))) if instance else None
            category_name = category.get("name") if category else annotation.get("category_name")
            raw.append(str(category_name) if category_name else "unknown")
            if instance_token:
                instance_tokens.add(instance_token)
        raw_unique = sorted(set(raw))
        description = str(scene.get("description", ""))
        weather, timeofday = derive_tags(description)
        filename = str(sample_data.get("filename", ""))
        result.append({
            "image_id": str(sample_data.get("token", "")),
            "image_path": str(safe_destination(extract_root, filename)),
            "filename": filename,
            "sample_token": sample_token,
            "sample_data_token": str(sample_data.get("token", "")),
            "scene_token": scene_token,
            "scene_name": str(scene.get("name", "")),
            "scene_description": description,
            "timestamp": sample_data.get("timestamp", sample.get("timestamp", "")),
            "camera_channel": str(sensor.get("channel", "")),
            "is_key_frame": is_key_frame,
            "location": str(log.get("location", "")),
            "objects": sorted({simplify_category(name) for name in raw_unique}),
            "objects_raw": raw_unique,
            "num_annotations": len(annotations_by_sample.get(sample_token, [])),
            "num_unique_objects": len(instance_tokens),
            "weather_tag": weather,
            "timeofday_tag": timeofday,
        })
    return result


def image_error(path: Path) -> str | None:
    if not path.is_file():
        return "missing"
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except Exception as exc:
        return f"corrupt: {type(exc).__name__}: {exc}"


def validate_records(records: list[dict[str, Any]], workers: int = 8) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    paths = [Path(row["image_path"]) for row in records]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        errors = list(pool.map(image_error, paths))
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    data_tokens: set[str] = set()
    image_paths: set[str] = set()
    for row, error in zip(records, errors):
        token = str(row.get("sample_data_token", ""))
        canonical_path = str(Path(row["image_path"]).resolve()).casefold()
        reason = error
        if not token: reason = reason or "missing sample_data_token"
        elif token in data_tokens: reason = reason or "duplicate sample_data_token"
        if canonical_path in image_paths: reason = reason or "duplicate image_path"
        if not row.get("sample_token"): reason = reason or "missing sample_token"
        if not row.get("scene_token"): reason = reason or "missing scene_token"
        if row.get("camera_channel") != "CAM_FRONT": reason = reason or "not CAM_FRONT"
        if row.get("is_key_frame") is not True: reason = reason or "not a keyframe"
        if reason:
            rejected.append({"sample_data_token": token, "image_path": row["image_path"], "reason": reason})
            continue
        data_tokens.add(token)
        image_paths.add(canonical_path)
        valid.append(row)
    return valid, rejected


def selection_target(value: Any, available: int) -> int:
    """Resolve an integer cap or the literal 'all' without silently truncating."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "all"):
        return available
    try:
        target = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("selection.target_size must be a positive integer or 'all'") from exc
    if target <= 0:
        raise ValueError("selection.target_size must be positive")
    if target > available:
        raise ValueError(f"Requested {target:,} images but only {available:,} are valid")
    return target


def deterministic_select(records: list[dict[str, Any]], target: int | str | None, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: str(row["sample_data_token"]))
    resolved = selection_target(target, len(ordered))
    if resolved == len(ordered):
        return ordered
    random.Random(seed).shuffle(ordered)
    return ordered[:resolved]


def scene_level_split(records: list[dict[str, Any]], dev_target: int, test_target: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records or dev_target <= 0 or test_target <= 0:
        raise ValueError("Scene split requires records and positive dev/test targets")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row["scene_token"])].append(row)
    scenes = sorted(groups)
    random.Random(seed).shuffle(scenes)
    # Bounded subset-sum selects test scenes closest to the requested size.
    maximum = len(records)
    predecessor: list[tuple[int, str] | None] = [None] * (maximum + 1)
    reachable = [False] * (maximum + 1)
    reachable[0] = True
    for scene in scenes:
        size = len(groups[scene])
        for total in range(maximum - size, -1, -1):
            if reachable[total] and not reachable[total + size]:
                reachable[total + size] = True
                predecessor[total + size] = (total, scene)
    preferred = min(test_target, maximum)
    best = min(
        (total for total, ok in enumerate(reachable) if ok),
        key=lambda total: (
            abs((maximum - total) - dev_target) + abs(total - preferred),
            abs(total - preferred),
            total,
        ),
    )
    test_scenes: set[str] = set()
    cursor = best
    while cursor:
        previous, scene = predecessor[cursor]  # type: ignore[misc]
        test_scenes.add(scene)
        cursor = previous
    dev = [row for row in records if str(row["scene_token"]) not in test_scenes]
    test = [row for row in records if str(row["scene_token"]) in test_scenes]
    return dev, test


def csv_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    output = {field: row.get(field, "") for field in CSV_FIELDS}
    output["objects"] = "|".join(row.get("objects", []))
    output["objects_raw"] = "|".join(row.get("objects_raw", []))
    output["weather"] = normalize_weather(row.get("scene_description"), row.get("weather_tag"))
    output["timeofday"] = normalize_timeofday(row.get("scene_description"), row.get("timeofday_tag"))
    output["scene"] = normalize_scene(row.get("scene_description"))
    output["num_objects"] = int(row.get("num_annotations", 0))
    output["source_split"] = split
    output["split"] = split
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def normalize_csv(path: Path) -> int:
    """Upgrade an existing nuScenes CSV to the canonical retrieval schema."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        description = row.get("scene_description", "")
        objects = [normalize_object(value) for value in row.get("objects", "").split("|")]
        objects = list(dict.fromkeys(value for value in objects if value))
        split = str(row.get("source_split") or row.get("split") or "").strip().lower()
        row.update(
            {
                "objects": "|".join(objects),
                "weather": normalize_weather(description, row.get("weather") or row.get("weather_tag")),
                "timeofday": normalize_timeofday(description, row.get("timeofday") or row.get("timeofday_tag")),
                "scene": row.get("scene") or normalize_scene(description),
                "num_objects": row.get("num_objects") or row.get("num_annotations") or len(objects),
                "source_split": split,
                "split": split,
            }
        )
        normalized_rows.append(row)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_csv(temporary, normalized_rows)
    temporary.replace(path)
    return len(normalized_rows)


def normalize_existing_outputs(output_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in ("metadata.csv", "dev.csv", "test.csv"):
        path = output_dir / name
        if path.is_file():
            counts[name] = normalize_csv(path)
    if not counts:
        raise FileNotFoundError(f"No nuScenes CSV outputs found in {output_dir}")
    return counts


def export_outputs(output_dir: Path, selected: list[dict[str, Any]], dev: list[dict[str, Any]], test: list[dict[str, Any]], stats: dict[str, Any], seed: int, requested_target: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_rows = [csv_row(row, "dev") for row in dev]
    test_rows = [csv_row(row, "test") for row in test]
    write_csv(output_dir / "metadata.csv", dev_rows + test_rows)
    write_csv(output_dir / "dev.csv", dev_rows)
    write_csv(output_dir / "test.csv", test_rows)
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    selection = {
        "seed": seed,
        "requested_target_size": requested_target,
        "algorithm": ("sort by sample_data_token, keep all valid records; bounded subset-sum scene split"
                      if len(selected) == stats["valid_images"] else
                      "sort by sample_data_token, shuffle with random.Random(seed), take target_size; bounded subset-sum scene split"),
        "sample_data_tokens": [row["sample_data_token"] for row in selected],
        "sha256": hashlib.sha256("\n".join(row["sample_data_token"] for row in selected).encode()).hexdigest(),
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")


def print_archive(info: ArchiveInfo) -> None:
    print(f"Archive: {info.path}")
    print(f"Format: {info.format}")
    print(f"Type: {info.classification}")
    print(f"Files: {info.file_count:,}")
    if info.metadata_tables:
        print("Detected metadata tables:")
        for table in info.metadata_tables: print(f"- {table}")
    for camera, count in info.camera_counts.items(): print(f"{camera} images found: {count:,}")
    print("Sample paths:")
    for member in info.sample_paths: print(f"- {member}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/nuscenes.yaml"))
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--normalize-only", action="store_true",
        help="Normalize existing output CSVs without reading or extracting archives.",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = load_config(configured_path(args.config))
    archive_dir = configured_path(config["dataset"]["archive_dir"])
    pattern = str(config["dataset"].get("archive_pattern", "*.taz"))
    extract_root = configured_path(config["dataset"]["extract_dir"])
    output_dir = configured_path(config["dataset"]["output_dir"])
    if args.normalize_only:
        counts = normalize_existing_outputs(output_dir)
        print(json.dumps({"normalized": counts, "output_dir": str(output_dir.resolve())}, indent=2))
        return
    archives = discover_archives(archive_dir, pattern)
    if not archives:
        raise SystemExit(no_archive_message(archive_dir, pattern))
    print("Discovered archives:")
    for path in archives: print(f"- {path.resolve()}")
    infos = []
    for path in archives:
        info = inspect_archive(path)
        infos.append(info)
        print_archive(info)
    force = args.force_extract or bool(config.get("extraction", {}).get("force", False))
    for path in archives:
        print(f"Extraction {path.name}: {extract_archive(path, extract_root, force)}")
    selection = config["selection"]
    candidates = join_cam_front_records(extract_root, str(selection["camera"]), bool(selection["keyframes_only"]))
    valid, rejected = validate_records(candidates, args.workers)
    print(f"Found {len(valid):,} valid CAM_FRONT keyframes.")
    requested_target = selection.get("target_size", "all")
    try:
        selected = deterministic_select(valid, requested_target, int(selection["seed"]))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if "test_fraction" in selection:
        test_fraction = float(selection["test_fraction"])
        if not 0 < test_fraction < 1:
            raise ValueError("selection.test_fraction must be between 0 and 1")
        test_target = max(1, round(len(selected) * test_fraction))
        dev_target = len(selected) - test_target
    else:
        dev_target = int(selection["dev_target"])
        test_target = int(selection["test_target"])
    dev, test = scene_level_split(selected, dev_target, test_target, int(selection["seed"]))
    dev_scenes = {row["scene_token"] for row in dev}
    test_scenes = {row["scene_token"] for row in test}
    if dev_scenes & test_scenes:
        raise AssertionError("Scene leakage detected")
    rejected_counts = Counter("missing" if row["reason"] == "missing" else "corrupt" if row["reason"].startswith("corrupt:") else "other" for row in rejected)
    stats = {
        "archives": [asdict(info) for info in infos],
        "cam_front_keyframe_candidates": len(candidates),
        "valid_images": len(valid),
        "selected": len(selected),
        "selection_target": requested_target,
        "all_valid_images_selected": len(selected) == len(valid),
        "dev_count": len(dev), "test_count": len(test),
        "dev_scenes": len(dev_scenes), "test_scenes": len(test_scenes),
        "scene_leakage": 0,
        "missing_images": rejected_counts["missing"],
        "corrupted_images": rejected_counts["corrupt"],
        "other_rejections": rejected_counts["other"],
        "derived_fields": ["weather_tag", "timeofday_tag"],
    }
    export_outputs(output_dir, selected, dev, test, stats, int(selection["seed"]), requested_target)
    print(json.dumps(stats, indent=2))
    print(f"Wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
