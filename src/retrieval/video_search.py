from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.models.clip_encoder import CLIPEncoder
from src.retrieval.vector_index import create_flat_ip
from src.video.frame_extractor import ExtractedFrame, extract_frames


class VideoSearchEngine:
    """Build a temporary CLIP index for one uploaded video and search it by text."""

    def __init__(self, encoder: CLIPEncoder) -> None:
        self.encoder = encoder
        self.index: Any | None = None
        self.frames: list[ExtractedFrame] = []

    def build_from_video(
        self,
        video_path: str | Path,
        frame_dir: str | Path,
        sample_fps: float = 2.0,
        batch_size: int = 64,
    ) -> dict[str, Any]:
        self.frames = extract_frames(video_path, frame_dir, sample_fps=sample_fps)
        image_paths = [frame.image_path for frame in self.frames]
        embeddings, valid_indices, errors = self.encoder.encode_images(
            image_paths, batch_size=batch_size, show_progress=False
        )
        self.frames = [self.frames[index] for index in valid_indices]
        self.index = create_flat_ip(int(embeddings.shape[1]))
        self.index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        return {
            "sampled_frames": len(image_paths),
            "indexed_frames": len(self.frames),
            "skipped_frames": len(errors),
            "errors": errors,
        }

    def search(self, english_query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Video index is not built yet")
        query_embedding = np.ascontiguousarray(self.encoder.encode_text(english_query), dtype="float32")
        count = min(max(0, int(top_k)), self.index.ntotal)
        scores, positions = self.index.search(query_embedding, count)
        results: list[dict[str, Any]] = []
        for rank, (position, score) in enumerate(zip(positions[0], scores[0]), start=1):
            if position < 0:
                continue
            frame = self.frames[int(position)]
            item = asdict(frame)
            item.update(rank=rank, clip_score=float(score))
            results.append(item)
        return results
