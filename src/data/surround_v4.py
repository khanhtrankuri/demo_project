"""Dataset and collation for SurroundSearch-v4."""
from __future__ import annotations

from collections import Counter

import torch
from PIL import Image

from src.data.rich_caption import generate_scene_captions, generate_view_captions
from src.data.surround import CAMERAS
from src.models.multicamera import MultiScaleTransform

SPATIAL_BINS = {"left": 0, "center": 1, "right": 2}
DISTANCE_BINS = {"near": 0, "medium": 1, "far": 2}


class SurroundV4Dataset:
    def __init__(self, records, classes, image_size, mean, std, num_spatial_crops=3, training=False):
        self.records = records
        self.class_ids = {name: index for index, name in enumerate(classes)}
        self.transform = MultiScaleTransform(image_size, mean, std, num_spatial_crops, training)
        self.crops = 1 + num_spatial_crops
        self.size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        class_count = len(self.class_ids)
        images = torch.zeros(6, self.crops, 3, self.size, self.size)
        mask = torch.zeros(6, dtype=torch.bool)
        geometry = torch.zeros(6, 4)
        objects = torch.zeros(6, class_count)
        counts = torch.zeros(6, class_count, dtype=torch.long)
        spatial = torch.full((6, class_count), -100, dtype=torch.long)
        distance = torch.full((6, class_count), -100, dtype=torch.long)
        view_captions, view_ids = [[] for _ in CAMERAS], [None] * len(CAMERAS)
        for camera_index, camera in enumerate(CAMERAS):
            view = record.get("views", {}).get(camera)
            if view is None:
                continue
            with Image.open(view["image_path"]) as source:
                images[camera_index] = self.transform(source)
            mask[camera_index] = True
            yaw = float(view.get("geometry", {}).get("yaw", 0.0))
            translation = list(view.get("geometry", {}).get("translation", [0.0, 0.0, 0.0]))
            geometry[camera_index] = torch.tensor([yaw, *translation])
            grouped = {}
            for item in view.get("object_instances", []):
                if item.get("class") in self.class_ids:
                    grouped.setdefault(item["class"], []).append(item)
            if not grouped:
                grouped = {name: [{} for _ in range(int(value))]
                           for name, value in view.get("object_counts", {}).items() if name in self.class_ids}
            for name, instances in grouped.items():
                class_index = self.class_ids[name]
                objects[camera_index, class_index] = 1
                counts[camera_index, class_index] = min(4, len(instances))
                sides = Counter(item.get("spatial") for item in instances if item.get("spatial") in SPATIAL_BINS)
                ranges = Counter(item.get("distance_bin") for item in instances if item.get("distance_bin") in DISTANCE_BINS)
                if sides:
                    spatial[camera_index, class_index] = SPATIAL_BINS[sides.most_common(1)[0][0]]
                if ranges:
                    distance[camera_index, class_index] = DISTANCE_BINS[ranges.most_common(1)[0][0]]
            view_captions[camera_index] = view.get("captions") or generate_view_captions(view)
            view_ids[camera_index] = str(view.get("image_id", f"{record['sample_token']}:{camera}"))
        scene_captions = record.get("captions") or generate_scene_captions(record)
        return {"images": images, "camera_mask": mask, "geometry": geometry, "objects": objects,
                "counts": counts, "spatial": spatial, "distance": distance,
                "view_captions": view_captions, "view_ids": view_ids,
                "scene_captions": scene_captions, "scene_id": str(record["sample_token"]), "index": index}


def collate_v4(rows):
    tensors = {key: torch.stack([row[key] for row in rows])
               for key in ("images", "camera_mask", "geometry", "objects", "counts", "spatial", "distance")}
    tensors.update({key: [row[key] for row in rows]
                    for key in ("view_captions", "view_ids", "scene_captions", "scene_id", "index")})
    return tensors
