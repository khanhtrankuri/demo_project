"""Deterministic, annotation-grounded captions for surround retrieval."""
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict

COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
CAMERA_NAMES = {
    "CAM_FRONT": "front",
    "CAM_FRONT_LEFT": "front-left",
    "CAM_FRONT_RIGHT": "front-right",
    "CAM_BACK": "rear",
    "CAM_BACK_LEFT": "rear-left",
    "CAM_BACK_RIGHT": "rear-right",
}


def count_phrase(count: int, name: str) -> str:
    amount = COUNT_WORDS.get(count, str(count)) if count < 5 else "at least five"
    suffix = name if count == 1 else f"{name}s"
    return f"{amount} {suffix}"


def _seeded(items: list[str], key: str, limit: int) -> list[str]:
    """Return a stable varied ordering without using process-randomized hash()."""
    values = list(dict.fromkeys(item.strip() for item in items if item and item.strip()))
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    random.Random(seed).shuffle(values)
    return values[:limit]


def _objects(view: dict) -> list[dict]:
    if view.get("object_instances"):
        return list(view["object_instances"])
    # Backwards-compatible fallback for v1 manifests. Unknown relations remain
    # unknown rather than being fabricated.
    return [
        {"class": name, "spatial": None, "distance_bin": None, "attribute": None}
        for name, count in view.get("object_counts", {}).items()
        for _ in range(int(count))
    ]


def generate_view_captions(view: dict, min_captions: int = 3, max_captions: int = 5) -> list[str]:
    if not 1 <= min_captions <= max_captions:
        raise ValueError("caption limits must satisfy 1 <= min <= max")
    camera = CAMERA_NAMES.get(view.get("camera"), str(view.get("camera", "camera")).lower())
    objects = _objects(view)
    counts = Counter(obj["class"] for obj in objects)
    inventory = ", ".join(count_phrase(n, cls) for cls, n in sorted(counts.items())) or "no labeled objects"
    full = f"{camera.capitalize()} camera view with {inventory}."
    compact = f"{camera} view, {inventory}"
    candidates = [full, compact]
    for cls, group in sorted(_group_by_class(objects).items()):
        sides = Counter(obj.get("spatial") for obj in group if obj.get("spatial"))
        distances = Counter(obj.get("distance_bin") for obj in group if obj.get("distance_bin"))
        attributes = Counter(obj.get("attribute") for obj in group if obj.get("attribute"))
        if sides:
            side = sides.most_common(1)[0][0]
            candidates.append(f"{count_phrase(len(group), cls).capitalize()} visible on the {side} in the {camera} camera.")
        if distances:
            distance = distances.most_common(1)[0][0]
            candidates.append(f"{count_phrase(len(group), cls).capitalize()} {distance} in the {camera} view.")
        if attributes:
            attribute = attributes.most_common(1)[0][0]
            candidates.append(f"{attribute.capitalize()} {cls} in the {camera} camera.")
    selected = _seeded(candidates, str(view.get("image_id", view.get("image_path", camera))), max_captions)
    while len(selected) < min_captions:
        selected.append(f"Road scene from the {camera} camera showing {inventory}.")
        selected = list(dict.fromkeys(selected))
        if len(selected) < min_captions:
            selected.append(f"{inventory} in the {camera} direction.")
    return selected[:max_captions]


def _group_by_class(objects: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for obj in objects:
        grouped[obj["class"]].append(obj)
    return grouped


def scene_objects(record: dict) -> list[dict]:
    """Deduplicate boxes visible in overlapping cameras by instance token."""
    result, positions = [], {}
    for camera, view in record.get("views", {}).items():
        for index, obj in enumerate(_objects(view)):
            key = obj.get("instance_token") or (camera, obj.get("class"), index)
            if key in positions:
                cameras = result[positions[key]].setdefault("cameras", [result[positions[key]]["camera"]])
                if camera not in cameras:
                    cameras.append(camera)
                continue
            positions[key] = len(result)
            selected_camera = obj.get("camera", camera)
            result.append(dict(obj, camera=selected_camera, cameras=[selected_camera]))
    return result


def generate_scene_captions(record: dict, min_captions: int = 3, max_captions: int = 5) -> list[str]:
    if not 1 <= min_captions <= max_captions:
        raise ValueError("caption limits must satisfy 1 <= min <= max")
    objects = scene_objects(record)
    counts = Counter(obj["class"] for obj in objects)
    inventory = ", ".join(count_phrase(n, cls) for cls, n in sorted(counts.items())) or "no labeled objects"
    candidates = [f"Surround scene containing {inventory}.", f"surround view, {inventory}"]
    by_camera = defaultdict(Counter)
    for obj in objects:
        by_camera[obj.get("camera")][obj["class"]] += 1
    for camera, camera_counts in sorted(by_camera.items(), key=lambda item: str(item[0])):
        direction = CAMERA_NAMES.get(camera, str(camera).lower())
        summary = ", ".join(count_phrase(n, cls) for cls, n in sorted(camera_counts.items()))
        candidates.append(f"The {direction} camera shows {summary}.")
    relations = []
    for obj in objects:
        if obj.get("spatial") or obj.get("distance_bin"):
            relation = " ".join(x for x in (obj.get("attribute"), obj["class"], obj.get("spatial"), obj.get("distance_bin")) if x)
            relations.append(relation)
    if relations:
        candidates.append("Driving scene with " + "; ".join(relations[:3]) + ".")
    selected = _seeded(candidates, str(record.get("sample_token", "scene")), max_captions)
    while len(selected) < min_captions:
        selected.append(f"Road surround view showing {inventory}.")
        selected = list(dict.fromkeys(selected))
        if len(selected) < min_captions:
            selected.append(f"Multi-camera driving sample with {inventory}.")
    return selected[:max_captions]


def enrich_record_captions(record: dict, min_captions: int = 3, max_captions: int = 5) -> dict:
    """Mutate a newly-built manifest record while preserving legacy captions."""
    for view in record.get("views", {}).values():
        view["captions"] = generate_view_captions(view, min_captions, max_captions)
        view.setdefault("caption", view["captions"][0])
    record["captions"] = generate_scene_captions(record, min_captions, max_captions)
    record.setdefault("caption", record["captions"][0])
    return record
