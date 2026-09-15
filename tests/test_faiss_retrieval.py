import json
from pathlib import Path

import numpy as np

from src.retrieval.semantic_search import SemanticSearchEngine
from src.retrieval.vector_index import create_flat_ip, write_index


class MockEncoder:
    def encode_text(self, query):
        return np.array([[1.0, 0.0]], dtype="float32")


def test_faiss_retrieval_with_mock_vectors(tmp_path: Path):
    index = create_flat_ip(2)
    index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    index_path = tmp_path / "mock.faiss"; write_index(index, index_path)
    mapping_path = tmp_path / "mock.json"
    mapping_path.write_text(json.dumps([{"image_id": "best"}, {"image_id": "other"}]))
    engine = SemanticSearchEngine(MockEncoder(), index_path, mapping_path)
    results, latency = engine.search("anything", 2)
    assert results[0]["image_id"] == "best"
    assert results[0]["clip_score"] == 1.0
    assert latency >= 0
