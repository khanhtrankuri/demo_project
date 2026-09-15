from __future__ import annotations

import re
from typing import Any


OBJECT_TERMS = {
    "pedestrian": "person", "pedestrians": "person", "people": "person", "person": "person",
    "cars": "car", "car": "car", "bike": "bike", "bikes": "bike", "bicycle": "bike",
    "bicycles": "bike", "motorbike": "motorcycle", "motorbikes": "motorcycle",
    "motorcycle": "motorcycle", "motorcycles": "motorcycle", "truck": "truck",
    "trucks": "truck", "bus": "bus", "buses": "bus", "traffic light": "traffic light",
    "traffic lights": "traffic light", "traffic sign": "traffic sign", "traffic signs": "traffic sign",
}
WEATHER_TERMS = {"rain": "rainy", "raining": "rainy", "rainy": "rainy", "snow": "snowy", "snowing": "snowy", "snowy": "snowy", "fog": "foggy", "foggy": "foggy", "clear": "clear", "overcast": "overcast", "partly cloudy": "partly cloudy"}
TIME_TERMS = {"night": "nighttime", "nighttime": "nighttime", "day": "daytime", "daytime": "daytime", "dawn": "dawn/dusk", "dusk": "dawn/dusk"}
SCENE_TERMS = {"city street": "city street", "highway": "highway", "residential": "residential", "parking lot": "parking lot", "tunnel": "tunnel", "gas station": "gas stations", "gas stations": "gas stations"}


def _contains(query: str, term: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", query) is not None


def _first_match(query: str, terms: dict[str, str]) -> str | None:
    for term in sorted(terms, key=len, reverse=True):
        if _contains(query, term):
            return terms[term]
    return None


def parse_query(query: str) -> dict[str, Any]:
    raw = query.strip()
    lowered = raw.lower()
    objects: list[str] = []
    for term in sorted(OBJECT_TERMS, key=len, reverse=True):
        canonical = OBJECT_TERMS[term]
        if _contains(lowered, term) and canonical not in objects:
            objects.append(canonical)
    return {
        "raw_query": raw,
        "semantic_query": raw,
        "weather": _first_match(lowered, WEATHER_TERMS),
        "timeofday": _first_match(lowered, TIME_TERMS),
        "scene": _first_match(lowered, SCENE_TERMS),
        "objects": objects,
    }

