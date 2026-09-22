"""Small residual adaptation of frozen pretrained image/text features."""
import torch
from torch import nn
from torch.nn import functional as F


def balanced_contrastive_loss(images, texts, captions, scale):
    """Deduplicate text candidates and give each caption equal loss weight."""
    lookup, first, groups = {}, [], []
    for index, caption in enumerate(captions):
        if caption not in lookup:
            lookup[caption] = len(first)
            first.append(index)
        groups.append(lookup[caption])
    targets = torch.tensor(groups, device=images.device)
    logits = scale.float() * images.float() @ texts[first].float().T
    counts = torch.bincount(targets, minlength=len(first)).float()
    image_loss = (F.cross_entropy(logits, targets, reduction="none") / counts[targets]).sum() / len(first)
    positives = targets[:, None].eq(torch.arange(len(first), device=images.device)[None, :]).T.float()
    positives /= positives.sum(dim=1, keepdim=True)
    text_loss = -(positives * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return (image_loss + text_loss) / 2


class ResidualAdapter(nn.Module):
    def __init__(self, width, bottleneck, dropout, scale):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, bottleneck),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(bottleneck, width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = scale

    def forward(self, x):
        return F.normalize(x + self.scale * self.net(x), dim=-1)


class SurroundTransfer(nn.Module):
    def __init__(self, config, classes):
        super().__init__()
        width = config["width"]
        args = (width, config["bottleneck"], config["dropout"], config["residual_scale"])
        self.view_adapter = ResidualAdapter(*args)
        self.scene_adapter = ResidualAdapter(*args)
        self.attention = nn.Linear(width, 1)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
        self.camera_bias = nn.Parameter(torch.zeros(6))
        self.object_head = nn.Linear(width, classes)
        self.logit_scale = nn.Parameter(torch.tensor(2.65926))

    def forward(self, features, mask, camera_dropout=0.):
        if not mask.any(dim=1).all():
            raise ValueError("Every sample needs a camera")
        # Mask before processing: absent camera content must not influence fusion.
        features = features.masked_fill(~mask[..., None], 0)
        views = self.view_adapter(features)
        fusion_mask = mask.clone()
        if self.training and camera_dropout:
            fusion_mask &= torch.rand_like(mask, dtype=torch.float32) >= camera_dropout
            empty = ~fusion_mask.any(dim=1)
            fusion_mask[empty] = mask[empty]
        attention = (self.attention(features).squeeze(-1) + self.camera_bias).masked_fill(~fusion_mask, -torch.inf)
        pooled = F.normalize((features * attention.softmax(dim=1)[..., None]).sum(1), dim=-1)
        base_scene = F.normalize((features * mask[..., None]).sum(1), dim=-1)
        return {"view": views, "scene": self.scene_adapter(pooled),
                "base_scene": base_scene, "object_logits": self.object_head(views)}
