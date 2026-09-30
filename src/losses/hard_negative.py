"""Explicit ranking loss for typed hard-negative pairs."""
import torch
from torch.nn import functional


def hard_negative_loss(anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor,
                       margin: float = 0.2) -> torch.Tensor:
    if anchors.shape != positives.shape or anchors.shape != negatives.shape:
        raise ValueError("anchors, positives and negatives must have the same shape")
    positive_score = functional.cosine_similarity(anchors.float(), positives.float())
    negative_score = functional.cosine_similarity(anchors.float(), negatives.float())
    return functional.softplus((negative_score - positive_score + margin) * 5).mean() / 5
