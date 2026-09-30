"""Generate semantic retrieval queries and ground truth from annotations."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from src.data.rich_caption import CAMERA_NAMES, scene_objects


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    category: str
    predicate: dict
    relevant_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["relevant_ids"] = list(self.relevant_ids)
        return value


def _sample_id(record: dict) -> str:
    return str(record.get("sample_token", record.get("image_id")))


def _matches(record: dict, predicate: dict) -> bool:
    objects = scene_objects(record)
    cls = predicate.get("class")
    selected = [obj for obj in objects if cls is None or obj.get("class") == cls]
    if predicate.get("camera"):
        selected = [obj for obj in selected if predicate["camera"] in obj.get("cameras", [obj.get("camera")])]
    if predicate.get("spatial"):
        selected = [obj for obj in selected if obj.get("spatial") == predicate["spatial"]]
    if predicate.get("distance_bin"):
        selected = [obj for obj in selected if obj.get("distance_bin") == predicate["distance_bin"]]
    if predicate.get("attribute"):
        selected = [obj for obj in selected if obj.get("attribute") == predicate["attribute"]]
    if predicate.get("max_distance_m") is not None:
        selected = [obj for obj in selected if obj.get("distance_m") is not None and obj["distance_m"] < predicate["max_distance_m"]]
    if len(selected) < int(predicate.get("min_count", 1)):
        return False
    for required in predicate.get("classes", []):
        if not any(obj.get("class") == required for obj in objects):
            return False
    return True


def _text(predicate: dict, category: str) -> str:
    cls = predicate.get("class", "object")
    if category == "object":
        return f"Find scenes containing a {cls}."
    if category == "count":
        return f"Find scenes with at least {predicate['min_count']} {cls}s."
    if category == "spatial":
        return f"Find scenes where a {cls} is visible on the {predicate['spatial']}."
    if category == "distance":
        if "max_distance_m" in predicate:
            return f"Find a {cls} less than {predicate['max_distance_m']:g} meters away."
        return f"Find scenes with a {cls} {predicate['distance_bin']}."
    if category == "attribute":
        return f"Find scenes containing a {predicate['attribute']} {cls}."
    if category == "camera-specific":
        camera = CAMERA_NAMES.get(predicate["camera"], predicate["camera"].lower())
        return f"Find a {cls} in the {camera} camera."
    if category == "multi-object":
        return "Find scenes containing both " + " and ".join(predicate["classes"]) + "."
    return "Find driving scenes matching the annotated scene conditions."


def generate_queries(records: Iterable[dict], min_relevant: int = 1) -> list[dict]:
    records = list(records)
    predicates: set[tuple[str, tuple]] = set()
    for record in records:
        objects = scene_objects(record)
        counts = Counter(obj["class"] for obj in objects)
        for cls, count in counts.items():
            predicates.add(("object", (("class", cls),)))
            if count >= 2:
                predicates.add(("count", (("class", cls), ("min_count", min(count, 4)))))
        for obj in objects:
            cls = obj["class"]
            for category, key in (("spatial", "spatial"), ("distance", "distance_bin"), ("attribute", "attribute")):
                if obj.get(key):
                    predicates.add((category, tuple(sorted({"class": cls, key: obj[key]}.items()))))
            if obj.get("distance_m") is not None and obj["distance_m"] < 20:
                predicates.add(("distance", tuple(sorted({"class": cls, "max_distance_m": 20}.items()))))
            for camera in obj.get("cameras", [obj.get("camera")]):
                if camera:
                    predicates.add(("camera-specific", tuple(sorted({"class": cls, "camera": camera}.items()))))
        for first, second in zip(sorted(counts), sorted(counts)[1:]):
            predicates.add(("multi-object", (("classes", (first, second)),)))
    queries = []
    for category, frozen in sorted(predicates, key=str):
        predicate = {key: list(value) if key == "classes" else value for key, value in frozen}
        relevant = tuple(_sample_id(r) for r in records if _matches(r, predicate))
        if len(relevant) < min_relevant:
            continue
        query_id = f"{category}:{len(queries):06d}"
        queries.append(Query(query_id, _text(predicate, category), category, predicate, relevant).to_dict())
    return queries


def relevance_by_query(queries: Iterable[dict]) -> dict[str, set[str]]:
    return {query["query_id"]: set(query["relevant_ids"]) for query in queries}
