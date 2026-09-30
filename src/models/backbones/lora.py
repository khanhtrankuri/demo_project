"""LoRA injection shared by Hugging Face CLIP and SigLIP towers."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid LoRA settings")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs):
        delta = functional.linear(functional.linear(self.dropout(inputs.float()), self.lora_A), self.lora_B)
        return self.base(inputs) + (delta * self.scale).to(inputs.dtype)


def inject_attention_lora(tower: nn.Module, settings: dict, *, last_layers: int | None = None) -> list[str]:
    targets = tuple(settings.get("targets", ("q_proj", "v_proj")))
    if not targets or not set(targets) <= {"q_proj", "k_proj", "v_proj", "out_proj"}:
        raise ValueError("LoRA targets must be attention projections")
    layers = tower.encoder.layers
    start = 0 if last_layers is None else max(0, len(layers) - int(last_layers))
    replaced = []
    for index, layer in enumerate(layers[start:], start):
        attention = layer.self_attn
        for target in targets:
            base = getattr(attention, target)
            setattr(attention, target, LoRALinear(base, int(settings["rank"]), float(settings["alpha"]),
                                                  float(settings.get("dropout", 0.0))))
            replaced.append(f"encoder.layers.{index}.self_attn.{target}")
    return replaced
