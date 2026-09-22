"""Torch loading separated from lightweight metadata preparation."""
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.surround import CAMERAS


class SurroundDataset:
    def __init__(self, records, classes, config, training=False):
        self.records, self.classes = records, {name: i for i, name in enumerate(classes)}
        self.height, self.width = config["image_height"], config["image_width"]
        operations = [transforms.Resize((self.height, self.width), interpolation=InterpolationMode.BILINEAR)]
        if training:
            operations += [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)]
        # Keep the full FOV and camera orientation consistent with projection labels.
        operations += [transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        images = torch.zeros(6, 3, self.height, self.width)
        mask = torch.zeros(6, dtype=torch.bool)
        labels = torch.zeros(6, len(self.classes))
        captions = [""] * 6
        for camera_index, camera in enumerate(CAMERAS):
            view = record["views"].get(camera)
            if view is None:
                continue
            # Fail with the path if data changed after preparation; no silent skips.
            with Image.open(view["image_path"]) as image:
                images[camera_index] = self.transform(image.convert("RGB"))
            mask[camera_index] = True
            for name in view["objects"]:
                labels[camera_index, self.classes[name]] = 1
            captions[camera_index] = view["caption"]
        return images, mask, labels, captions, record["caption"], index


def collate_surround(batch):
    images, masks, labels, view_captions, scene_captions, indices = zip(*batch)
    return (torch.stack(images), torch.stack(masks), torch.stack(labels),
            [c for views in view_captions for c in views], list(scene_captions), list(indices))
