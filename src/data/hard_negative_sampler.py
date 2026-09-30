"""Typed hard-negative mining from structured nuScenes labels."""
from __future__ import annotations

import random

from src.data.rich_caption import scene_objects
from src.models.retrieval.hard_negative import NegativePair

KINDS = ("semantic", "count", "distance", "temporal", "cross_camera", "random")


def _facts(record: dict) -> dict:
    objects = scene_objects(record)
    return {
        "classes": {obj["class"] for obj in objects},
        "count": {cls: sum(obj["class"] == cls for obj in objects) for cls in {obj["class"] for obj in objects}},
        "spatial": {(obj["class"], obj.get("spatial")) for obj in objects if obj.get("spatial")},
        "distance": {(obj["class"], obj.get("distance_bin")) for obj in objects if obj.get("distance_bin")},
        "camera": {(obj["class"], camera) for obj in objects
                   for camera in obj.get("cameras", [obj.get("camera")]) if camera},
    }


def classify_negative(anchor: dict, candidate: dict) -> str:
    return _classify_facts(anchor, candidate, _facts(anchor), _facts(candidate))


def _classify_facts(anchor, candidate, left, right):
    same_classes = left["classes"] & right["classes"]
    if anchor.get("scene_token") == candidate.get("scene_token") and anchor.get("sample_token") != candidate.get("sample_token"):
        return "temporal"
    if same_classes and any(left["count"].get(cls) != right["count"].get(cls) for cls in same_classes):
        return "count"
    if same_classes and left["spatial"] != right["spatial"]:
        return "semantic"
    if same_classes and left["distance"] != right["distance"]:
        return "distance"
    if same_classes and left["camera"] != right["camera"]:
        return "cross_camera"
    return "random" if not same_classes else "equivalent"


class HardNegativeSampler:
    def __init__(self, records: list[dict], seed: int = 42):
        self.records = records
        self.seed = seed
        self.facts = [_facts(record) for record in records]

    def sample(self, anchor_indices: list[int], per_anchor: int = 1, kinds=KINDS) -> list[NegativePair]:
        if per_anchor < 1:
            raise ValueError("per_anchor must be positive")
        allowed = set(kinds)
        output = []
        for anchor_index in anchor_indices:
            # Lazy Fisher-Yates permutation: uniform sampling without replacement,
            # stopping as soon as enough eligible negatives have been found.
            rng = random.Random(self.seed + anchor_index)
            swaps = {}
            found = 0
            for remaining in range(len(self.records), 0, -1):
                slot = rng.randrange(remaining)
                index = swaps.get(slot, slot)
                swaps[slot] = swaps.get(remaining - 1, remaining - 1)
                if index == anchor_index:
                    continue
                kind = _classify_facts(self.records[anchor_index], self.records[index],
                                       self.facts[anchor_index], self.facts[index])
                if kind in allowed:
                    output.append(NegativePair(anchor_index, index, kind))
                    found += 1
                    if found == per_anchor:
                        break
        return output
