"""Masked camera-aware Transformer fusion."""
import torch
from torch import nn
from torch.nn import functional


class CameraTransformerFusion(nn.Module):
    def __init__(self, width: int, cameras: int = 6, layers: int = 2, heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by attention heads")
        self.cameras = cameras
        self.scene_cls = nn.Parameter(torch.empty(1, 1, width))
        self.camera_embedding = nn.Parameter(torch.empty(cameras, width))
        block = nn.TransformerEncoderLayer(width, heads, width * 4, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        nn.init.normal_(self.scene_cls, std=.02)
        nn.init.normal_(self.camera_embedding, std=.02)

    def forward(self, camera_tokens: torch.Tensor, camera_mask: torch.Tensor,
                geometry_embedding: torch.Tensor | None = None, camera_dropout: float = 0.0) -> dict:
        if camera_tokens.ndim != 3 or camera_tokens.shape[1] != self.cameras:
            raise ValueError(f"camera_tokens must be [B,{self.cameras},D]")
        if not camera_mask.any(dim=1).all():
            raise ValueError("Every scene needs at least one camera")
        mask = camera_mask.to(camera_tokens.device).clone()
        if self.training and camera_dropout:
            mask &= torch.rand_like(mask, dtype=torch.float32) >= camera_dropout
            empty = ~mask.any(dim=1)
            mask[empty] = camera_mask.to(mask.device)[empty]
        tokens = camera_tokens + self.camera_embedding[None]
        if geometry_embedding is not None:
            tokens = tokens + geometry_embedding.to(tokens.dtype)
        cls = self.scene_cls.expand(len(tokens), -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        padding = torch.cat([torch.zeros(len(mask), 1, dtype=torch.bool, device=mask.device), ~mask], dim=1)
        contextual = self.norm(self.encoder(sequence, src_key_padding_mask=padding))
        # Zero missing positions so downstream auxiliary heads cannot learn from
        # the camera-id embedding of an absent view.
        cameras = contextual[:, 1:].masked_fill(~mask[..., None], 0)
        return {"scene": functional.normalize(contextual[:, 0].float(), dim=-1),
                "cameras": cameras, "mask": mask, "tokens": contextual}


class WeightedCameraFusion(nn.Module):
    """Legacy-compatible scalar pooling used only by early ablation rows."""
    def __init__(self, width: int, cameras: int = 6):
        super().__init__()
        self.cameras = cameras
        self.attention = nn.Linear(width, 1)
        self.camera_bias = nn.Parameter(torch.zeros(cameras))

    def forward(self, camera_tokens, camera_mask, geometry_embedding=None, camera_dropout=0.0):
        mask = camera_mask.to(camera_tokens.device).clone()
        if self.training and camera_dropout:
            mask &= torch.rand_like(mask, dtype=torch.float32) >= camera_dropout
            empty = ~mask.any(1)
            mask[empty] = camera_mask.to(mask.device)[empty]
        tokens = camera_tokens if geometry_embedding is None else camera_tokens + geometry_embedding.to(camera_tokens.dtype)
        scores = (self.attention(tokens).squeeze(-1) + self.camera_bias).masked_fill(~mask, -torch.inf)
        scene = functional.normalize((tokens * scores.softmax(1)[..., None]).sum(1).float(), dim=-1)
        cameras = tokens.masked_fill(~mask[..., None], 0)
        return {"scene": scene, "cameras": cameras, "mask": mask,
                "tokens": torch.cat([scene[:, None].to(tokens.dtype), cameras], 1)}
