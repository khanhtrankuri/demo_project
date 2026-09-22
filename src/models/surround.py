"""SurroundSearch-v1: compact, from-scratch, six-camera dual encoder."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def tokenize(texts, max_bytes=192):
    # PAD=0, UTF-8 bytes=1..256, BOS=257, EOS=258. No learned vocabulary leak.
    if max_bytes < 4:
        raise ValueError("max_bytes must be >= 4")
    tokens = torch.zeros(len(texts), max_bytes, dtype=torch.long)
    for i, text in enumerate(texts):
        values = [257] + [b + 1 for b in text.lower().encode("utf-8")[:max_bytes - 2]] + [258]
        tokens[i, :len(values)] = torch.tensor(values)
    return tokens


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride, 1, bias=False),
            nn.GroupNorm(8, output_channels), nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, output_channels),
        )
        self.skip = (nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
                     if input_channels != output_channels or stride != 1 else nn.Identity())

    def forward(self, x):
        return F.gelu(self.body(x) + self.skip(x))


class SurroundSearch(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        width, heads = int(config["width"]), int(config["heads"])
        dropout = float(config["dropout"])
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.max_text_bytes = int(config["max_text_bytes"])
        self.visual = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
            ResidualBlock(32, 64, 2), ResidualBlock(64, 128, 2),
            ResidualBlock(128, 192, 2), ResidualBlock(192, 256, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, width), nn.LayerNorm(width),
        )
        self.image_projection = nn.Linear(width, width, bias=False)
        self.object_head = nn.Linear(width, num_classes)
        self.camera_embedding = nn.Parameter(torch.randn(6, width) * 0.02)
        self.scene_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        # Construct independently: cloned Transformer layers otherwise start identical.
        self.fusion = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, width * 4, dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(int(config["fusion_layers"]))
        ])
        self.scene_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width, bias=False))
        self.embedding = nn.Embedding(259, width, padding_idx=0)
        self.text_position = nn.Parameter(torch.randn(self.max_text_bytes, width) * 0.01)
        self.text_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, width * 4, dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(int(config["text_layers"]))
        ])
        self.text_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width, bias=False))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_text(self, tokens):
        mask = tokens.eq(0)
        x = self.embedding(tokens) + self.text_position[:tokens.shape[1]]
        for layer in self.text_layers:
            x = layer(x, src_key_padding_mask=mask)
        # BOS attends to every non-padding token.
        return F.normalize(self.text_projection(x[:, 0]).float(), dim=-1)

    def encode_image(self, images):
        return F.normalize(self.image_projection(self.visual(images)).float(), dim=-1)

    def forward(self, images, mask, camera_dropout=0.0):
        if images.ndim != 5 or images.shape[1] != 6 or mask.shape != images.shape[:2]:
            raise ValueError("Expected images [B,6,3,H,W] and mask [B,6]")
        if not mask.any(dim=1).all():
            raise ValueError("Each sample needs at least one camera")
        batch = images.shape[0]
        flat_mask = mask.flatten()
        valid = self.visual(images.flatten(0, 1)[flat_mask])
        features = valid.new_zeros(batch * 6, valid.shape[-1])
        features[flat_mask] = valid
        features = features.reshape(batch, 6, -1)
        view = F.normalize(self.image_projection(features).float(), dim=-1)
        fusion_mask = mask.clone()
        if self.training and camera_dropout:
            fusion_mask &= torch.rand_like(mask, dtype=torch.float32) >= camera_dropout
            empty = ~fusion_mask.any(dim=1)
            fusion_mask[empty] = mask[empty]
        x = torch.cat([self.scene_token.expand(batch, -1, -1), features + self.camera_embedding], dim=1)
        padding = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=mask.device), ~fusion_mask], dim=1)
        for layer in self.fusion:
            x = layer(x, src_key_padding_mask=padding)
        return {"view": view, "scene": F.normalize(self.scene_projection(x[:, 0]).float(), dim=-1),
                "object_logits": self.object_head(features)}


def contrastive_loss(images, texts, captions, scale):
    """Uniform multi-positive target: every equal caption is a positive."""
    if len(captions) != len(images) or len(texts) != len(images):
        raise ValueError("Features and captions must align")
    ids = {}
    groups = torch.tensor([ids.setdefault(c, len(ids)) for c in captions], device=images.device)
    positives = groups[:, None].eq(groups[None, :]).float()
    logits = scale.float() * images.float() @ texts.float().T
    targets = positives / positives.sum(dim=1, keepdim=True)
    return -0.5 * ((targets * F.log_softmax(logits, dim=1)).sum(1).mean()
                   + (targets * F.log_softmax(logits.T, dim=1)).sum(1).mean())
