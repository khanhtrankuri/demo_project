from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from src.utils.common import clean_text, is_meaningful, unique_preserving_order


OBJECT_ALIASES = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.construction": "construction vehicle",
    "vehicle.trailer": "trailer",
    "vehicle.emergency.police": "police vehicle",
    "vehicle.emergency.ambulance": "ambulance",
    "movable_object.trafficcone": "traffic cone",
    "movable_object.barrier": "barrier",
    "movable_object.pushable_pullable": "pushable object",
    "movable_object.debris": "debris",
    "static_object.bicycle_rack": "bicycle rack",
}


def normalize_object(value: Any) -> str:
    """Map nuScenes taxonomy values to short, human-readable CLIP labels."""
    raw = clean_text(value).lower()
    if not raw:
        return ""
    if raw.startswith("human.pedestrian."):
        return "person"
    return OBJECT_ALIASES.get(raw, raw.replace("_", " "))


def split_objects(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        parts = value
    else:
        parts = clean_text(value).split("|")
    return unique_preserving_order(
        normalized for item in parts if (normalized := normalize_object(item))
    )


def normalize_scene(description: Any) -> str:
    """Derive only scene types explicitly supported by the source description."""
    text = clean_text(description).lower()
    rules = (
        ("parking lot", ("parking lot", "car park")),
        ("tunnel", ("tunnel",)),
        ("highway", ("highway", "freeway")),
        ("residential", ("residential",)),
        ("gas station", ("gas station",)),
        ("intersection", ("intersection", "crossroad")),
        ("construction zone", ("construction",)),
        ("bridge", ("bridge",)),
    )
    return next((label for label, terms in rules if any(term in text for term in terms)), "unknown")


def normalize_weather(description: Any, existing: Any = None) -> str:
    current = clean_text(existing).lower()
    if is_meaningful(current):
        return current
    text = clean_text(description).lower()
    rules = (
        ("rainy", ("rain", "raining", "wet weather")),
        ("foggy", ("fog", "foggy")),
        ("snowy", ("snow", "snowing")),
    )
    return next((label for label, terms in rules if any(term in text for term in terms)), "unknown")


def normalize_timeofday(description: Any, existing: Any = None) -> str:
    current = clean_text(existing).lower()
    if is_meaningful(current):
        return current
    text = clean_text(description).lower()
    rules = (
        ("nighttime", ("night", "after dark")),
        ("dawn/dusk", ("dawn", "dusk", "sunrise", "sunset")),
        ("daytime", ("daytime", "day time")),
    )
    return next((label for label, terms in rules if any(term in text for term in terms)), "unknown")


@dataclass
class NuScenesRecord:
    image_id: str
    image_path: str
    source_split: str
    weather: str
    timeofday: str
    scene: str
    objects: list[str]
    num_objects: int
    annotation_path: str = ""
    scene_description: str = ""
    location: str = ""
    scene_token: str = ""
    sample_token: str = ""

    @property
    def has_meaningful_metadata(self) -> bool:
        return any(
            (
                is_meaningful(self.scene_description),
                is_meaningful(self.scene),
                is_meaningful(self.location),
                bool(self.objects),
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_from_row(row: dict[str, Any], metadata_path: Path) -> NuScenesRecord:
    image_path = Path(clean_text(row.get("image_path")))
    if not image_path.is_absolute():
        image_path = (metadata_path.parent / image_path).resolve()
    description = clean_text(row.get("scene_description"))
    objects = split_objects(row.get("objects") or row.get("objects_raw"))
    split = clean_text(row.get("source_split") or row.get("split")).lower()
    return NuScenesRecord(
        image_id=clean_text(row.get("image_id") or row.get("sample_data_token")),
        image_path=str(image_path),
        source_split=split,
        weather=normalize_weather(description, row.get("weather") or row.get("weather_tag")),
        timeofday=normalize_timeofday(description, row.get("timeofday") or row.get("timeofday_tag")),
        scene=clean_text(row.get("scene")) or normalize_scene(description),
        objects=objects,
        num_objects=int(row.get("num_objects") or row.get("num_annotations") or len(objects)),
        annotation_path=str(metadata_path),
        scene_description=description,
        location=clean_text(row.get("location")),
        scene_token=clean_text(row.get("scene_token")),
        sample_token=clean_text(row.get("sample_token")),
    )


def iter_processed_records(
    processed_dir: str | Path,
    split: str,
    errors: list[dict[str, str]] | None = None,
) -> Iterator[NuScenesRecord]:
    processed_dir = Path(processed_dir).resolve()
    metadata_path = processed_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"nuScenes metadata file does not exist: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            row_split = clean_text(row.get("source_split") or row.get("split")).lower()
            if split != "all" and row_split != split.lower():
                continue
            try:
                record = record_from_row(row, metadata_path)
                if not record.image_id:
                    raise ValueError("missing image_id/sample_data_token")
                if not Path(record.image_path).is_file():
                    raise FileNotFoundError(record.image_path)
                yield record
            except Exception as exc:
                if errors is None:
                    raise
                errors.append(
                    {
                        "metadata_path": str(metadata_path),
                        "line": str(line_number),
                        "image_path": clean_text(row.get("image_path")),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
