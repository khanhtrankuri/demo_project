from __future__ import annotations

from typing import Any


VALUE_ALIASES = {
    "nighttime": "night",
    "night": "night",
    "motorcycle": "motor",
    "motorbike": "motor",
    "bicycle": "bike",
    "people": "person",
    "pedestrian": "person",
    "traffic_cone": "traffic cone",
    "gas stations": "gas station",
}


def canonical_value(value: Any) -> str:
    normalized = "" if value is None else str(value).strip().lower()
    return VALUE_ALIASES.get(normalized, normalized)


def requested_conditions(parsed_query: dict[str, Any]) -> list[tuple[str, str]]:
    conditions: list[tuple[str, str]] = []
    for field in ("weather", "timeofday", "scene"):
        value = parsed_query.get(field)
        if value:
            conditions.append((field, canonical_value(value)))
    for value in parsed_query.get("objects") or []:
        conditions.append(("objects", canonical_value(value)))
    return conditions


def metadata_score(candidate: dict[str, Any], parsed_query: dict[str, Any]) -> float:
    conditions = requested_conditions(parsed_query)
    if not conditions:
        return 0.0
    candidate_objects = {canonical_value(value) for value in candidate.get("objects", [])}
    matched = 0
    for field, expected in conditions:
        if field == "objects":
            matched += expected in candidate_objects
        else:
            matched += canonical_value(candidate.get(field)) == expected
    return matched / len(conditions)


def metadata_matches(candidate: dict[str, Any], parsed_query: dict[str, Any]) -> bool:
    conditions = requested_conditions(parsed_query)
    return not conditions or metadata_score(candidate, parsed_query) == 1.0


def apply_metadata(candidates: list[dict[str, Any]], parsed_query: dict[str, Any], hard: bool = False) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        enriched = dict(candidate)
        enriched["metadata_score"] = metadata_score(candidate, parsed_query)
        if not hard or metadata_matches(candidate, parsed_query):
            results.append(enriched)
    return results

