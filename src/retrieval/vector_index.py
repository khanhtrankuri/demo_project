from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


FAISS_IMPORT_ERROR: str | None = None
try:
    import faiss as _faiss
except Exception as exc:  # Native loading can fail under Windows Application Control.
    _faiss = None
    FAISS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

FAISS_AVAILABLE = _faiss is not None


class NumpyFlatIP:
    """Exact IndexFlatIP-compatible fallback for policy-restricted machines."""

    def __init__(self, dimension: int) -> None:
        self.d = int(dimension)
        self.vectors = np.empty((0, self.d), dtype="float32")

    @property
    def ntotal(self) -> int:
        return int(len(self.vectors))

    def add(self, vectors: np.ndarray) -> None:
        values = np.ascontiguousarray(vectors, dtype="float32")
        if values.ndim != 2 or values.shape[1] != self.d:
            raise ValueError(f"Expected vectors shaped (n, {self.d}), got {values.shape}")
        self.vectors = np.concatenate((self.vectors, values), axis=0)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.ascontiguousarray(queries, dtype="float32")
        if values.ndim != 2 or values.shape[1] != self.d:
            raise ValueError(f"Expected queries shaped (n, {self.d}), got {values.shape}")
        k = min(max(0, int(k)), self.ntotal)
        if k == 0:
            return np.empty((len(values), 0), dtype="float32"), np.empty((len(values), 0), dtype="int64")
        similarities = values @ self.vectors.T
        positions = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        scores = np.take_along_axis(similarities, positions, axis=1)
        return scores.astype("float32"), positions.astype("int64")


def backend_name() -> str:
    return "faiss.IndexFlatIP" if FAISS_AVAILABLE else "numpy exact inner-product fallback"


def create_flat_ip(dimension: int) -> Any:
    return _faiss.IndexFlatIP(int(dimension)) if FAISS_AVAILABLE else NumpyFlatIP(dimension)


def write_index(index: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if FAISS_AVAILABLE and not isinstance(index, NumpyFlatIP):
        _faiss.write_index(index, str(path))
        return
    with path.open("wb") as handle:
        np.savez(handle, vectors=index.vectors, dimension=np.asarray([index.d], dtype="int64"))
    path.with_suffix(path.suffix + ".backend.json").write_text(
        json.dumps({"backend": backend_name(), "faiss_import_error": FAISS_IMPORT_ERROR}, indent=2) + "\n",
        encoding="utf-8",
    )


def read_index(path: str | Path) -> Any:
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".backend.json")
    if sidecar.is_file():
        with np.load(path) as saved:
            index = NumpyFlatIP(int(saved["dimension"][0]))
            index.add(saved["vectors"])
            return index
    if not FAISS_AVAILABLE:
        raise RuntimeError(f"This is a genuine FAISS index, but FAISS cannot load: {FAISS_IMPORT_ERROR}")
    return _faiss.read_index(str(path))

