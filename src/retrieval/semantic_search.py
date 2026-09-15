from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval.vector_index import read_index


class SemanticSearchEngine:
    def __init__(self, encoder: Any, index_path: str | Path, mapping_path: str | Path) -> None:
        self.encoder = encoder
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.index = read_index(self.index_path)
        self.mapping: list[dict[str, Any]] = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        if self.index.ntotal != len(self.mapping):
            raise ValueError(f"Index has {self.index.ntotal} vectors but mapping has {len(self.mapping)} rows")

    def search(self, query: str, top_k: int = 10) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        query_embedding = np.ascontiguousarray(self.encoder.encode_text(query), dtype="float32")
        count = min(max(0, top_k), self.index.ntotal)
        scores, positions = self.index.search(query_embedding, count)
        results: list[dict[str, Any]] = []
        for rank, (position, score) in enumerate(zip(positions[0], scores[0]), start=1):
            if position < 0:
                continue
            result = dict(self.mapping[int(position)])
            result.update(rank=rank, faiss_position=int(position), clip_score=float(score))
            results.append(result)
        return results, (time.perf_counter() - started) * 1000
