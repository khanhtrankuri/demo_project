from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm


class CLIPEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        mixed_precision: bool = True,
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch and open_clip_torch are required; install requirements.txt") from exc
        self.torch = torch
        self.open_clip = open_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.mixed_precision = mixed_precision and self.device == "cuda"
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def _autocast(self):
        return self.torch.autocast(device_type="cuda", dtype=self.torch.float16) if self.mixed_precision else nullcontext()

    def encode_images(
        self, image_paths: Sequence[str | Path], batch_size: int = 64, show_progress: bool = True
    ) -> tuple[np.ndarray, list[int], list[dict[str, str]]]:
        batches: list[np.ndarray] = []
        valid_indices: list[int] = []
        errors: list[dict[str, str]] = []
        iterator = range(0, len(image_paths), batch_size)
        iterator = tqdm(iterator, desc="Encoding images", unit="batch", disable=not show_progress)
        for start in iterator:
            tensors = []
            indices = []
            for index in range(start, min(start + batch_size, len(image_paths))):
                path = Path(image_paths[index])
                try:
                    with Image.open(path) as image:
                        tensors.append(self.preprocess(image.convert("RGB")))
                    indices.append(index)
                except Exception as exc:
                    errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            if not tensors:
                continue
            tensor = self.torch.stack(tensors).to(self.device, non_blocking=True)
            try:
                with self.torch.no_grad(), self._autocast():
                    features = self.model.encode_image(tensor)
                    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            except self.torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(
                    f"CUDA out of memory at batch_size={batch_size}. Retry with --batch-size 32 or 16."
                ) from exc
            batches.append(features.float().cpu().numpy())
            valid_indices.extend(indices)
        dimension = int(getattr(self.model.visual, "output_dim", 512))
        embeddings = np.concatenate(batches).astype("float32", copy=False) if batches else np.empty((0, dimension), dtype="float32")
        return embeddings, valid_indices, errors

    def encode_text(self, texts: str | Sequence[str]) -> np.ndarray:
        values = [texts] if isinstance(texts, str) else list(texts)
        tokens = self.tokenizer(values).to(self.device)
        with self.torch.no_grad(), self._autocast():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.float().cpu().numpy().astype("float32", copy=False)

