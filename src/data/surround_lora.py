"""Full-FOV CLIP preprocessing; never horizontally flip camera orientation."""
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.surround import CAMERAS


class LoRASurroundDataset:
    def __init__(self, records, classes, image_processor, training=False):
        self.records = records
        self.classes = {name: i for i, name in enumerate(classes)}
        self.size = image_processor.crop_size["height"]
        if self.size != image_processor.crop_size["width"]:
            raise ValueError("CLIP requires square input")
        self.color = tuple(round(value * 255) for value in image_processor.image_mean)
        self.jitter = transforms.ColorJitter(.1, .1, .1) if training else None
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(), transforms.Normalize(image_processor.image_mean, image_processor.image_std),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        images = torch.zeros(6, 3, self.size, self.size)
        mask = torch.zeros(6, dtype=torch.bool)
        labels = torch.zeros(6, len(self.classes))
        captions = [""] * 6
        for j, camera in enumerate(CAMERAS):
            view = record["views"].get(camera)
            if view is None:
                continue
            with Image.open(view["image_path"]) as source:
                image = source.convert("RGB")
                if self.jitter is not None:
                    image = self.jitter(image)
                side = max(image.size)
                images[j] = self.transform(ImageOps.pad(image, (side, side), color=self.color))
            mask[j] = True
            for name in view["objects"]:
                labels[j, self.classes[name]] = 1
            captions[j] = view["caption"]
        return images, mask, labels, captions, record["caption"], index
