"""Vision LoRA on frozen Hugging Face CLIP, with six-camera retrieval heads."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.surround_transfer import SurroundTransfer


class LoRALinear(nn.Module):
    """W(x) + (alpha / rank) B A dropout(x); W stays frozen."""
    def __init__(self, base, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA requires Linear, positive rank/alpha and dropout in [0, 1)")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        # FP32 trainable weights / optimizer; frozen weights may be BF16.
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        result = self.base(x)
        delta = F.linear(F.linear(self.dropout(x.float()), self.lora_A), self.lora_B)
        return result + (self.scale * delta).to(result.dtype)


def inject_vision_lora(clip, settings):
    targets = set(settings["lora_targets"])
    if not targets or not targets <= {"q_proj", "k_proj", "v_proj", "out_proj"}:
        raise ValueError("LoRA targets must be CLIP attention projections")
    clip.requires_grad_(False)
    replaced = []
    for name, module in list(clip.vision_model.named_modules()):
        if ".self_attn." in name and name.rsplit(".", 1)[-1] in targets and isinstance(module, nn.Linear):
            parent_name, child = name.rsplit(".", 1)
            parent = clip.vision_model.get_submodule(parent_name)
            setattr(parent, child, LoRALinear(module, settings["lora_rank"], settings["lora_alpha"], settings["lora_dropout"]))
            replaced.append(f"vision_model.{name}")
    expected = len(clip.vision_model.encoder.layers) * len(targets)
    if len(replaced) != expected:
        raise ValueError(f"Expected {expected} LoRA projections; found {len(replaced)}")
    return replaced


class SurroundLoRA(nn.Module):
    def __init__(self, clip, settings, classes):
        super().__init__()
        self.clip = clip
        self.targets = inject_vision_lora(clip, settings)
        if settings["gradient_checkpointing"]:
            # Non-reentrant checkpointing propagates gradients even with frozen inputs.
            clip.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        width = clip.config.projection_dim
        self.heads = SurroundTransfer(dict(width=width, bottleneck=settings["bottleneck"],
                                          dropout=settings["dropout"], residual_scale=settings["residual_scale"]), classes)
        with torch.no_grad():
            self.heads.logit_scale.copy_(clip.logit_scale.float().clamp(0, math.log(100)))

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        self.clip.vision_model.train(mode)
        return self

    def encode_views(self, images, mask):
        device = self.clip.visual_projection.weight.device
        mask = mask.to(device)
        if not mask.any(dim=1).all():
            raise ValueError("Every sample needs a camera")
        pixels = images.to(device, non_blocking=True)[mask]
        result = self.clip.vision_model(pixel_values=pixels).pooler_output
        features = F.normalize(self.clip.visual_projection(result).float(), dim=-1)
        grouped = features.new_zeros(*mask.shape, features.shape[-1])
        grouped[mask] = features
        return grouped

    @torch.no_grad()
    def encode_text(self, tokens):
        self.clip.text_model.eval()
        features = self.clip.get_text_features(**tokens)
        return F.normalize(features.float(), dim=-1)

    def adapter_state(self):
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def load_adapter(self, state):
        parameters = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if parameters.keys() != state.keys():
            raise ValueError("Adapter checkpoint parameter names differ")
        with torch.no_grad():
            for name, parameter in parameters.items():
                if parameter.shape != state[name].shape:
                    raise ValueError(f"Adapter shape differs: {name}")
                parameter.copy_(state[name])

    def parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    lora=sum(p.numel() for n, p in self.named_parameters() if "lora_" in n),
                    target_projections=len(self.targets))
