"""Aspect-preserving global/crop encoding with intra-camera attention."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms


class MultiScaleTransform:
    """Produce a letterboxed global view plus overlapping horizontal crops."""

    def __init__(self, image_size: int, mean, std, num_spatial_crops: int = 3, training: bool = False):
        if image_size < 1 or num_spatial_crops < 0:
            raise ValueError("image_size must be positive and num_spatial_crops nonnegative")
        self.image_size = image_size
        self.num_spatial_crops = num_spatial_crops
        self.fill = tuple(round(float(value) * 255) for value in mean)
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.jitter = transforms.ColorJitter(.1, .1, .1) if training else None

    def _resize(self, image: Image.Image) -> Image.Image:
        return image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.jitter is not None:
            image = self.jitter(image)
        width, height = image.size
        side = max(width, height)
        global_view = self._resize(ImageOps.pad(image, (side, side), color=self.fill))
        crop_width = min(width, height)
        max_left = max(0, width - crop_width)
        positions = [round(i * max_left / max(1, self.num_spatial_crops - 1)) for i in range(self.num_spatial_crops)]
        crops = [self._resize(image.crop((left, 0, left + crop_width, height))) for left in positions]
        return torch.stack([self.to_tensor(global_view)] + [self.to_tensor(crop) for crop in crops])


@dataclass
class SpatialOutput:
    camera_tokens: torch.Tensor
    crop_tokens: torch.Tensor
    patch_tokens: torch.Tensor | None = None


class SpatialEncoder(nn.Module):
    """Shared backbone over crops followed by a learned intra-camera CLS token."""

    def __init__(self, backbone: nn.Module, width: int, num_spatial_crops: int = 3,
                 layers: int = 1, heads: int = 8, dropout: float = 0.1,
                 backbone_batch_size: int = 1):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by attention heads")
        self.backbone = backbone
        self.backbone_batch_size = int(backbone_batch_size)
        if self.backbone_batch_size < 1:
            raise ValueError("backbone_batch_size must be positive")
        self.num_tokens = 1 + num_spatial_crops
        self.input_projection = (nn.Identity() if backbone.projection_dim == width
                                 else nn.Linear(backbone.projection_dim, width))
        self.crop_embedding = nn.Parameter(torch.empty(self.num_tokens, width))
        self.camera_cls = nn.Parameter(torch.empty(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        nn.init.normal_(self.crop_embedding, std=0.02)
        nn.init.normal_(self.camera_cls, std=0.02)

    def forward(self, images: torch.Tensor, camera_mask: torch.Tensor,
                return_patches: bool = False) -> SpatialOutput:
        if images.ndim != 6 or images.shape[1] != camera_mask.shape[1] or images.shape[2] != self.num_tokens:
            raise ValueError("images must be [B,C,1+num_spatial_crops,3,H,W]")
        if not camera_mask.any(dim=1).all():
            raise ValueError("Every scene needs at least one camera")
        batch, cameras, crops = images.shape[:3]
        valid = camera_mask.to(images.device)
        pixels = images[valid].reshape(-1, *images.shape[3:])
        encoded = torch.cat([self.backbone.encode_image(pixels[start:start + self.backbone_batch_size])
                             for start in range(0, len(pixels), self.backbone_batch_size)])
        encoded = encoded.reshape(-1, crops, self.backbone.projection_dim)
        encoded = self.input_projection(encoded)
        sequence = encoded + self.crop_embedding[None]
        cls = self.camera_cls.expand(len(sequence), -1, -1)
        sequence = self.encoder(torch.cat([cls, sequence], dim=1))
        camera = self.norm(sequence[:, 0])
        grouped_camera = camera.new_zeros(batch, cameras, camera.shape[-1])
        grouped_crops = camera.new_zeros(batch, cameras, crops, camera.shape[-1])
        grouped_camera[valid] = camera
        grouped_crops[valid] = sequence[:, 1:]
        patches = None
        if return_patches:
            patch = torch.cat([self.backbone.get_patch_tokens(pixels[start:start + self.backbone_batch_size])
                               for start in range(0, len(pixels), self.backbone_batch_size)])
            patch = self.input_projection(patch)
            patches = patch.new_zeros(batch, cameras, crops, patch.shape[-2], patch.shape[-1])
            patches[valid] = patch.reshape(-1, crops, patch.shape[-2], patch.shape[-1])
        return SpatialOutput(grouped_camera, grouped_crops, patches)
