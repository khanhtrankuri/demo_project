from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from src.utils.common import clean_text, is_meaningful, unique_preserving_order


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_ATTRIBUTES = ("weather", "timeofday", "scene")


@dataclass
class BDDRecord:
    image_id: str
    image_path: str
    source_split: str
    weather: str
    timeofday: str
    scene: str
    objects: list[str]
    num_objects: int
    annotation_path: str

    @property
    def has_complete_metadata(self) -> bool:
        return all(clean_text(getattr(self, field)) for field in REQUIRED_ATTRIBUTES)

    @property
    def has_meaningful_metadata(self) -> bool:
        return all(is_meaningful(getattr(self, field)) for field in REQUIRED_ATTRIBUTES)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_dataset_root(start: str | Path) -> Path:
    start = Path(start).resolve()
    candidates = [start] if start.is_dir() else []
    candidates.extend(p for p in start.rglob("*") if p.is_dir())
    valid: list[Path] = []
    for candidate in candidates:
        split_count = sum((candidate / split / "img").is_dir() for split in ("train", "val", "test"))
        if split_count:
            valid.append(candidate)
    if not valid:
        raise FileNotFoundError(f"Could not locate BDD100K split/img folders below {start}")
    return max(valid, key=lambda p: sum((p / s / "img").is_dir() for s in ("train", "val", "test")))


def image_is_decodable(path: str | Path, fully_decode: bool = False) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as image:
            image.load() if fully_decode else image.verify()
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _parse_datasetninja(record: dict[str, Any]) -> tuple[dict[str, str], list[str], int]:
    tags = record.get("tags", [])
    objects = record.get("objects", [])
    if not isinstance(tags, list) or not isinstance(objects, list):
        raise ValueError("DatasetNinja annotation requires list-valued tags and objects")
    attrs = {field: "" for field in REQUIRED_ATTRIBUTES}
    for tag in tags:
        if isinstance(tag, dict) and tag.get("name") in attrs:
            attrs[str(tag["name"])] = clean_text(tag.get("value"))
    classes = [clean_text(obj.get("classTitle")) for obj in objects if isinstance(obj, dict)]
    classes = [value for value in classes if value]
    return attrs, unique_preserving_order(classes), len(classes)


def _parse_canonical(record: dict[str, Any]) -> tuple[dict[str, str], list[str], int]:
    attributes = record.get("attributes", {})
    labels = record.get("labels", [])
    if not isinstance(attributes, dict) or not isinstance(labels, list):
        raise ValueError("Canonical annotation requires attributes object and labels list")
    attrs = {field: clean_text(attributes.get(field)) for field in REQUIRED_ATTRIBUTES}
    classes = [clean_text(label.get("category")) for label in labels if isinstance(label, dict)]
    classes = [value for value in classes if value]
    return attrs, unique_preserving_order(classes), len(classes)


def parse_annotation(annotation_path: str | Path, image_path: str | Path, split: str) -> BDDRecord:
    annotation_path = Path(annotation_path).resolve()
    image_path = Path(image_path).resolve()
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Annotation root must be a JSON object")
    if "tags" in data or "objects" in data:
        attrs, objects, num_objects = _parse_datasetninja(data)
    elif "attributes" in data or "labels" in data:
        attrs, objects, num_objects = _parse_canonical(data)
    else:
        raise ValueError("Unrecognized BDD100K annotation format")
    embedded_name = clean_text(data.get("name"))
    if embedded_name and embedded_name.lower() != image_path.name.lower():
        raise ValueError(f"Annotation name {embedded_name!r} does not match {image_path.name!r}")
    return BDDRecord(
        image_id=image_path.name,
        image_path=str(image_path),
        source_split=split,
        weather=attrs["weather"],
        timeofday=attrs["timeofday"],
        scene=attrs["scene"],
        objects=objects,
        num_objects=num_objects,
        annotation_path=str(annotation_path),
    )


def iter_split_records(
    dataset_root: str | Path, split: str, errors: list[dict[str, str]] | None = None
) -> Iterator[BDDRecord]:
    root = Path(dataset_root).resolve()
    image_dir = root / split / "img"
    annotation_dir = root / split / "ann"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        return
    for image_path in sorted(image_dir.rglob("*"), key=lambda p: str(p).lower()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = image_path.relative_to(image_dir)
        annotation_path = annotation_dir / relative.parent / f"{image_path.name}.json"
        if not annotation_path.is_file():
            if errors is not None:
                errors.append({"image_path": str(image_path), "annotation_path": str(annotation_path), "reason": "missing annotation"})
            continue
        try:
            yield parse_annotation(annotation_path, image_path, split)
        except Exception as exc:
            if errors is None:
                raise
            errors.append({"image_path": str(image_path), "annotation_path": str(annotation_path), "reason": f"{type(exc).__name__}: {exc}"})
