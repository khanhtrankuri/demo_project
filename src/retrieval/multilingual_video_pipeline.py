from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.clip_encoder import CLIPEncoder
from src.models.language_normalizer import LanguageQueryNormalizer
from src.retrieval.video_search import VideoSearchEngine


class MultilingualVideoRetrievalPipeline:
    """End-to-end online pipeline for one uploaded video plus one multilingual prompt."""

    def __init__(
        self,
        clip_encoder: CLIPEncoder,
        language_normalizer: LanguageQueryNormalizer,
    ) -> None:
        self.clip_encoder = clip_encoder
        self.language_normalizer = language_normalizer
        self.video_search = VideoSearchEngine(clip_encoder)

    def index_video(
        self,
        video_path: str | Path,
        frame_dir: str | Path,
        sample_fps: float = 2.0,
        batch_size: int = 64,
    ) -> dict[str, Any]:
        return self.video_search.build_from_video(
            video_path=video_path,
            frame_dir=frame_dir,
            sample_fps=sample_fps,
            batch_size=batch_size,
        )

    def search(self, multilingual_query: str, top_k: int = 10) -> dict[str, Any]:
        normalized = self.language_normalizer.normalize(multilingual_query)
        results = self.video_search.search(normalized.english_query, top_k=top_k)
        return {
            "raw_query": multilingual_query,
            "normalized_query": normalized.to_dict(),
            "results": results,
        }

    def run(
        self,
        video_path: str | Path,
        multilingual_query: str,
        frame_dir: str | Path,
        sample_fps: float = 2.0,
        batch_size: int = 64,
        top_k: int = 10,
    ) -> dict[str, Any]:
        index_info = self.index_video(
            video_path=video_path,
            frame_dir=frame_dir,
            sample_fps=sample_fps,
            batch_size=batch_size,
        )
        search_output = self.search(multilingual_query, top_k=top_k)
        search_output["index_info"] = index_info
        return search_output
