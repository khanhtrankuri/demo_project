"""Detached ring buffer for cross-batch contrastive candidates."""
import torch
from torch import nn


class CrossBatchMemoryQueue(nn.Module):
    def __init__(self, capacity: int, dimension: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        if capacity < 1 or dimension < 1 or dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("queue needs positive dimensions and FP16/BF16 storage")
        self.capacity = capacity
        self.register_buffer("embeddings", torch.zeros(capacity, dimension, dtype=dtype), persistent=True)
        self.register_buffer("ids", torch.full((capacity,), -1, dtype=torch.long), persistent=True)
        self.register_buffer("pointer", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("size", torch.zeros((), dtype=torch.long), persistent=True)

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, ids: torch.Tensor):
        embeddings, ids = embeddings.detach(), ids.detach().long()
        if len(embeddings) != len(ids) or embeddings.shape[1] != self.embeddings.shape[1]:
            raise ValueError("queue embedding/id shape mismatch")
        if len(embeddings) >= self.capacity:
            embeddings, ids = embeddings[-self.capacity:], ids[-self.capacity:]
        count = len(embeddings)
        pointer = int(self.pointer)
        first = min(count, self.capacity - pointer)
        self.embeddings[pointer:pointer + first].copy_(embeddings[:first].to(self.embeddings.dtype))
        self.ids[pointer:pointer + first].copy_(ids[:first].to(self.ids.device))
        rest = count - first
        if rest:
            self.embeddings[:rest].copy_(embeddings[first:].to(self.embeddings.dtype))
            self.ids[:rest].copy_(ids[first:].to(self.ids.device))
        self.pointer.fill_((pointer + count) % self.capacity)
        self.size.fill_(min(self.capacity, int(self.size) + count))

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        size = int(self.size)
        if size < self.capacity:
            return self.embeddings[:size].detach(), self.ids[:size].detach()
        pointer = int(self.pointer)
        order = torch.cat([torch.arange(pointer, self.capacity, device=self.ids.device),
                           torch.arange(0, pointer, device=self.ids.device)])
        return self.embeddings[order].detach(), self.ids[order].detach()
