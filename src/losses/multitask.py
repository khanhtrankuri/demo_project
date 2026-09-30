"""Named weighted loss aggregation for auditable training logs."""
from torch import nn


class MultiTaskLoss(nn.Module):
    KEYS = ("view", "scene", "hard_negative", "object", "count", "spatial", "distance", "regularization")

    def __init__(self, weights: dict[str, float]):
        super().__init__()
        unknown = set(weights) - set(self.KEYS)
        if unknown:
            raise ValueError(f"Unknown loss weights: {sorted(unknown)}")
        self.weights = {key: float(weights.get(key, 0.0)) for key in self.KEYS}

    def forward(self, losses: dict):
        missing = [key for key, weight in self.weights.items() if weight and key not in losses]
        if missing:
            raise ValueError(f"Missing enabled loss components: {missing}")
        weighted = {key: losses[key] * weight for key, weight in self.weights.items() if key in losses}
        total = sum(weighted.values())
        return {"total": total, "raw": losses, "weighted": weighted}
