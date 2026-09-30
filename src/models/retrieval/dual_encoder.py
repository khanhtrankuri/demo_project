"""SurroundSearch-v4 dual encoder and auxiliary training outputs."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from src.models.heads import CountHead, DistanceHead, ObjectHead, SpatialHead
from src.models.multicamera import CameraTransformerFusion, GeometryEncoder, SpatialEncoder, WeightedCameraFusion
from src.models.retrieval.reranker import CrossModalReranker, QueryAwareCameraFusion


class SurroundDualEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, classes: int, config: dict):
        super().__init__()
        width = int(config.get("width", backbone.projection_dim))
        spatial = config.get("spatial", {})
        fusion = config.get("fusion", {})
        self.spatial_encoder = SpatialEncoder(
            backbone, width, int(spatial.get("num_spatial_crops", config.get("num_spatial_crops", 3))),
            int(spatial.get("layers", 1)),
            int(spatial.get("heads", 8)), float(spatial.get("dropout", .1)),
            int(spatial.get("backbone_batch_size", 1)))
        self.geometry_encoder = GeometryEncoder(width)
        self.camera_fusion = (WeightedCameraFusion(width, cameras=6) if fusion.get("type") == "weighted"
                              else CameraTransformerFusion(
                                  width, cameras=6, layers=int(fusion.get("layers", 2)),
                                  heads=int(fusion.get("heads", 8)), dropout=float(fusion.get("dropout", .1))))
        self.use_geometry = bool(config.get("geometry", {}).get("enabled", True))
        self.view_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, backbone.projection_dim))
        self.scene_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, backbone.projection_dim))
        self.object_head = ObjectHead(width, classes)
        self.count_head = CountHead(width, classes)
        self.spatial_head = SpatialHead(width, classes)
        self.distance_head = DistanceHead(width, classes)
        reranker = config.get("reranker", {})
        self.reranker = (CrossModalReranker(width, int(reranker.get("layers", 2)),
                                           int(reranker.get("heads", 8)), float(reranker.get("dropout", .1)))
                         if reranker.get("enabled", False) else None)
        self.query_fusion = (QueryAwareCameraFusion(width, int(reranker.get("heads", 8)))
                             if reranker.get("enabled", False) else None)
        self.logit_scale = nn.Parameter(torch.tensor(2.65926))

    @property
    def backbone(self):
        return self.spatial_encoder.backbone

    def encode_visual(self, images: torch.Tensor, camera_mask: torch.Tensor,
                      geometry: torch.Tensor | None = None, return_tokens: bool = False,
                      camera_dropout: float = 0.0, return_patches: bool = False) -> dict:
        # The reranker consumes fused scene/camera tokens, not backbone patches.
        spatial = self.spatial_encoder(images, camera_mask, return_patches=return_patches)
        geometry_embedding = self.geometry_encoder(geometry) if geometry is not None and self.use_geometry else None
        fused = self.camera_fusion(spatial.camera_tokens, camera_mask, geometry_embedding, camera_dropout)
        views = functional.normalize(self.view_projection(fused["cameras"]).float(), dim=-1)
        views = views.masked_fill(~fused["mask"][..., None], 0)
        scene = functional.normalize(self.scene_projection(fused["scene"]).float(), dim=-1)
        output = {
            "view": views, "scene": scene, "camera_tokens": fused["cameras"],
            "object_logits": self.object_head(fused["cameras"]),
            "count_logits": self.count_head(fused["cameras"]),
            "spatial_logits": self.spatial_head(fused["cameras"]),
            "distance_logits": self.distance_head(fused["cameras"]),
            "mask": fused["mask"],
        }
        if return_tokens:
            output["scene_tokens"] = fused["tokens"]
            output["crop_tokens"] = spatial.crop_tokens
            output["patch_tokens"] = spatial.patch_tokens
        return output

    def encode_text(self, tokens, return_tokens: bool = False):
        output = {"embedding": self.backbone.encode_text(tokens)}
        if return_tokens:
            output["tokens"] = self.backbone.get_text_tokens(tokens)
        return output

    def parameter_groups(self, vision_lr: float, text_lr: float, head_lr: float):
        vision, text, heads = [], [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if "backbone.model.vision_model" in name:
                vision.append(parameter)
            elif "backbone.model.text_model" in name:
                text.append(parameter)
            else:
                heads.append(parameter)
        return [group for group in (
            {"params": vision, "lr": vision_lr, "name": "vision_lora"},
            {"params": text, "lr": text_lr, "name": "text_lora"},
            {"params": heads, "lr": head_lr, "name": "v4_heads"},
        ) if group["params"]]

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}
