"""Camera extrinsic geometry projection."""
import torch
from torch import nn


class GeometryEncoder(nn.Module):
    """Encode yaw and translation as [sin(yaw), cos(yaw), tx, ty, tz]."""

    def __init__(self, width: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(32, width // 2)
        self.net = nn.Sequential(nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, width), nn.LayerNorm(width))

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        if geometry.shape[-1] == 4:
            yaw, translation = geometry[..., :1], geometry[..., 1:]
            geometry = torch.cat([yaw.sin(), yaw.cos(), translation], dim=-1)
        if geometry.shape[-1] != 5:
            raise ValueError("geometry must contain yaw+xyz or sin/cos(yaw)+xyz")
        return self.net(geometry.float())
