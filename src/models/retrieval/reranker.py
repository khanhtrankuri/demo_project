"""Top-K cross-modal reranking and query-conditioned camera fusion."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class CrossAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float = .1):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 4, width))

    def forward(self, query, memory, memory_padding_mask=None):
        attended = self.attention(self.query_norm(query), self.memory_norm(memory), self.memory_norm(memory),
                                  key_padding_mask=memory_padding_mask, need_weights=False)[0]
        query = query + attended
        return query + self.mlp(self.output_norm(query))


class QueryAwareCameraFusion(nn.Module):
    """Create a query-conditioned scene vector for shortlisted candidates."""
    def __init__(self, width: int, heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, query_embedding: torch.Tensor, camera_tokens: torch.Tensor,
                camera_mask: torch.Tensor) -> torch.Tensor:
        query = query_embedding[:, None]
        attended = self.attention(query, camera_tokens, camera_tokens,
                                  key_padding_mask=~camera_mask, need_weights=False)[0][:, 0]
        return functional.normalize(self.norm(query_embedding + attended).float(), dim=-1)


class CrossModalReranker(nn.Module):
    """Score `[B,K]` candidates; never used while building the database index."""
    def __init__(self, width: int, layers: int = 2, heads: int = 8, dropout: float = .1):
        super().__init__()
        self.blocks = nn.ModuleList([CrossAttentionBlock(width, heads, dropout) for _ in range(layers)])
        self.score = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, text_tokens: torch.Tensor, visual_tokens: torch.Tensor,
                text_mask: torch.Tensor | None = None, visual_mask: torch.Tensor | None = None) -> torch.Tensor:
        if visual_tokens.ndim != 4:
            raise ValueError("visual_tokens must be [B,K,V,D]")
        batch, candidates, visual_length, width = visual_tokens.shape
        if text_tokens.shape[0] != batch or text_tokens.shape[-1] != width:
            raise ValueError("text and visual token shapes are incompatible")
        text = text_tokens[:, None].expand(-1, candidates, -1, -1).reshape(batch * candidates, -1, width)
        visual = visual_tokens.reshape(batch * candidates, visual_length, width)
        padding = None
        if visual_mask is not None:
            padding = ~visual_mask.reshape(batch * candidates, visual_length)
        for block in self.blocks:
            text = block(text, visual, padding)
        if text_mask is None:
            pooled = text.mean(1)
        else:
            mask = text_mask[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1).to(text.dtype)
            pooled = (text * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.score(pooled).reshape(batch, candidates)
