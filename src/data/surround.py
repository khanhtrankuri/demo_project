"""All-camera nuScenes ingestion with per-camera geometric annotation filtering."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.nuscenes import normalize_object

CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")


def iter_json_array(path: Path):
    """Stream top-level JSON arrays without loading multi-GB metadata into RAM."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer, position, eof, started = "", 0, False, False
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started and position < len(buffer):
                if buffer[position] != "[":
                    raise ValueError(f"Expected JSON array: {path}")
                position += 1
                started = True
                continue
            if position < len(buffer) and buffer[position] in ",]":
                if buffer[position] == "]":
                    return
                position += 1
                continue
            if position < len(buffer):
                try:
                    item, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"Invalid or truncated JSON: {path}")
                else:
                    position = end
                    yield item
                    continue
            if eof:
                raise ValueError(f"Truncated JSON array: {path}")
            chunk = handle.read(1024 * 1024)
            eof = not chunk
            buffer = buffer[position:] + chunk
            position = 0


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid wxyz rotation quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def box_corners(annotation):
    # nuScenes size is width, length, height; local x is length.
    width, length, height = annotation["size"]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = signs * np.array([length, width, height]) / 2
    return corners @ rotation(annotation["rotation"]).T + annotation["translation"]


def boxes_in_camera(corners, ego_pose, calibration, width, height):
    """nuScenes ANY-corner visibility; geometric FOV only, not occlusion labels."""
    if len(corners) == 0:
        return np.zeros(0, dtype=bool)
    points = (np.asarray(corners) - ego_pose["translation"]) @ rotation(ego_pose["rotation"])
    points = (points - calibration["translation"]) @ rotation(calibration["rotation"])
    projected = points @ np.asarray(calibration["camera_intrinsic"]).T
    depth = points[..., 2]
    uv = projected[..., :2] / np.where(abs(projected[..., 2:]) > 1e-8, projected[..., 2:], 1e-8)
    inside = ((uv[..., 0] > 0) & (uv[..., 0] < width) & (uv[..., 1] > 0)
              & (uv[..., 1] < height) & (depth > 1))
    return inside.any(axis=1) & (depth > 0.1).all(axis=1)


def scene_splits(scene_tokens, seed, validation_fraction, test_fraction, locked_test=()):
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one")
    all_scenes = set(scene_tokens)
    test = all_scenes & set(locked_test)
    remaining = sorted(all_scenes - test)
    random.Random(seed).shuffle(remaining)
    if not test:
        count = max(1, round(len(all_scenes) * test_fraction))
        test.update(remaining[:count])
        remaining = remaining[count:]
    count = max(1, round(len(all_scenes) * validation_fraction))
    if len(remaining) <= count:
        raise ValueError("Need enough scenes for nonempty train, validation and test")
    validation = set(remaining[:count])
    return {s: "test" if s in test else "val" if s in validation else "train" for s in all_scenes}


def write_jsonl(path, rows):
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_manifest(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_dataset(root, output, *, version="v1.0-trainval", seed=42,
                    validation_fraction=0.1, test_fraction=0.1,
                    previous_test=None, verify_images=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    tables = root / version
    def table(name):
        return iter_json_array(tables / f"{name}.json")
    def keyed(name):
        return {row["token"]: row for row in table(name)}
    sensors, calibrations = keyed("sensor"), keyed("calibrated_sensor")
    scenes, samples = keyed("scene"), keyed("sample")
    categories, instances = keyed("category"), keyed("instance")
    local_files = {str(p.relative_to(root)).replace("\\", "/") for camera in CAMERAS
                   for p in (root / "samples" / camera).glob("*") if p.is_file()}
    if not local_files:
        raise ValueError(f"No camera images in {root / 'samples'}")
    print(f"Found {len(local_files):,} local camera files. Streaming sample_data...", flush=True)
    data, referenced, failures = [], set(), []
    for row in table("sample_data"):
        filename = row["filename"].replace("\\", "/")
        if filename not in local_files:
            continue
        referenced.add(filename)
        calibration = calibrations[row["calibrated_sensor_token"]]
        camera = sensors[calibration["sensor_token"]]["channel"]
        if camera not in CAMERAS or not row["is_key_frame"]:
            raise ValueError(f"Expected camera keyframe, got {filename}")
        data.append(dict(row, camera=camera))
    if local_files - referenced:
        raise ValueError(f"{len(local_files - referenced)} local images lack metadata")
    needed_samples = {r["sample_token"] for r in data}
    needed_poses = {r["ego_pose_token"] for r in data}
    print("Streaming ego poses and annotations for local images...", flush=True)
    poses = {r["token"]: r for r in table("ego_pose") if r["token"] in needed_poses}
    annotations = defaultdict(list)
    for ann in table("sample_annotation"):
        if ann["sample_token"] in needed_samples:
            name = categories[instances[ann["instance_token"]]["category_token"]]["name"]
            annotations[ann["sample_token"]].append((normalize_object(name), box_corners(ann)))
    locked_test = set()
    if previous_test and Path(previous_test).is_file():
        with Path(previous_test).open(encoding="utf-8-sig", newline="") as handle:
            locked_test = {r["scene_token"] for r in csv.DictReader(handle)}
    splits = scene_splits({samples[t]["scene_token"] for t in needed_samples}, seed,
                          validation_fraction, test_fraction, locked_test)
    rows, groups = [], {}
    for index, row in enumerate(sorted(data, key=lambda r: r["token"])):
        path = root / row["filename"]
        if verify_images:
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.size != (row["width"], row["height"]):
                        raise ValueError("Image dimensions differ from metadata")
            except Exception as exc:
                failures.append({"image_path": str(path), "error": str(exc)})
                continue
        sample_token = row["sample_token"]
        scene_token = samples[sample_token]["scene_token"]
        ann = annotations[sample_token]
        visible = boxes_in_camera([a[1] for a in ann], poses[row["ego_pose_token"]],
                                  calibrations[row["calibrated_sensor_token"]], row["width"], row["height"])
        counts = Counter(a[0] for a, present in zip(ann, visible) if present)
        objects = sorted(counts)
        # Do not claim scene-wide actions are visible in each camera.
        caption = "A road view" + (" containing " + ", ".join(objects) if objects else "") + "."
        record = {"image_id": row["token"], "image_path": str(path),
                  "sample_token": sample_token, "scene_token": scene_token,
                  "camera": row["camera"], "timestamp": row["timestamp"],
                  "objects": objects, "object_counts": dict(counts),
                  "caption": caption, "split": splits[scene_token]}
        rows.append(record)
        group = groups.setdefault(sample_token, {
            "sample_token": sample_token, "scene_token": scene_token, "split": splits[scene_token],
            "caption": " ".join(scenes[scene_token]["description"].split()).rstrip(".") + ".",
            "views": {},
        })
        if row["camera"] in group["views"]:
            raise ValueError(f"Duplicate sample/camera: {sample_token}/{row['camera']}")
        group["views"][row["camera"]] = record
        if (index + 1) % 3000 == 0:
            print(f"Validated/projected {index + 1:,}/{len(data):,} images", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    grouped = sorted(groups.values(), key=lambda r: r["sample_token"])
    if not grouped or any(not any(r["split"] == s for r in grouped) for s in ("train", "val", "test")):
        raise ValueError("A split has no valid images")
    write_jsonl(output / "images.jsonl", rows)
    for split in ("train", "val", "test"):
        write_jsonl(output / f"{split}.jsonl", [r for r in grouped if r["split"] == split])
    write_jsonl(output / "rejected.jsonl", failures)
    labels = sorted({normalize_object(c["name"]) for c in categories.values()})
    stats = {"schema_version": 1, "root": str(root), "seed": seed,
             "camera_order": CAMERAS, "classes": labels, "local_camera_files": len(local_files),
             "images": len(rows), "samples": len(grouped), "scenes": len(splits),
             "per_camera": dict(Counter(r["camera"] for r in rows)),
             "split_images": dict(Counter(r["split"] for r in rows)),
             "split_samples": dict(Counter(r["split"] for r in grouped)),
             "split_scenes": dict(Counter(splits.values())), "rejected": len(failures),
             "complete_samples": sum(len(r["views"]) == 6 for r in grouped),
             "preserved_old_test_scenes": len(set(splits) & locked_test),
             "labels": "3D boxes in camera FOV, no occlusion guarantee; scene descriptions are weak supervision",
             "image_verification": "full_decode" if verify_images else "disabled",
             "split_sha256": hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()}
    (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats
