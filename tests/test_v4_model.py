import torch
from torch import nn

from src.models.retrieval import CrossModalReranker, QueryAwareCameraFusion, SurroundDualEncoder


class TinyBackbone(nn.Module):
    projection_dim = 16

    def __init__(self):
        super().__init__()
        self.image = nn.Linear(3, 16)
        self.text = nn.Embedding(32, 16)

    def encode_image(self, pixels):
        return self.image(pixels.mean((-1, -2)))

    def get_patch_tokens(self, pixels):
        return self.image(pixels.mean((-1, -2)))[:, None]

    def encode_text(self, tokens):
        return self.text(tokens["input_ids"]).mean(1)

    def get_text_tokens(self, tokens):
        return self.text(tokens["input_ids"])


def test_dual_encoder_outputs_all_heads_and_query_independent_index_embedding():
    config = {"width": 16, "spatial": {"num_spatial_crops": 3, "heads": 4},
              "fusion": {"layers": 2, "heads": 4}}
    model = SurroundDualEncoder(TinyBackbone(), classes=3, config=config).eval()
    images = torch.randn(2, 6, 4, 3, 8, 8)
    mask = torch.ones(2, 6, dtype=torch.bool)
    geometry = torch.zeros(2, 6, 4)
    first = model.encode_visual(images, mask, geometry, return_tokens=True)
    second = model.encode_visual(images, mask, geometry)
    assert torch.allclose(first["scene"], second["scene"])
    assert first["view"].shape == (2, 6, 16)
    assert first["object_logits"].shape == (2, 6, 3)
    assert first["count_logits"].shape == (2, 6, 3, 5)
    assert first["spatial_logits"].shape == first["distance_logits"].shape == (2, 6, 3, 3)


def test_reranker_and_query_aware_fusion_only_score_supplied_candidates():
    text = torch.randn(2, 5, 16, requires_grad=True)
    visual = torch.randn(2, 4, 7, 16, requires_grad=True)
    scores = CrossModalReranker(16, layers=2, heads=4)(text, visual)
    assert scores.shape == (2, 4)
    scores.sum().backward()
    assert text.grad is not None and visual.grad is not None
    camera = torch.randn(2, 6, 16)
    mask = torch.tensor([[True] * 6, [True, False, False, False, False, False]])
    conditioned = QueryAwareCameraFusion(16, 4)(text.detach().mean(1), camera, mask)
    assert conditioned.shape == (2, 16)
    assert torch.allclose(conditioned.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_reranker_tokens_do_not_run_unused_patch_encoder(monkeypatch):
    backbone = TinyBackbone()
    def unexpected_patches(*args):
        raise AssertionError('Scene-token reranking must not encode patches again')
    monkeypatch.setattr(backbone, 'get_patch_tokens', unexpected_patches)
    model = SurroundDualEncoder(backbone, 3, {
        'width': 16, 'spatial': {'heads': 4}, 'fusion': {'layers': 1, 'heads': 4}})
    result = model.encode_visual(torch.randn(1, 6, 4, 3, 8, 8),
                                 torch.ones(1, 6, dtype=torch.bool), return_tokens=True)
    assert result['scene_tokens'].shape == (1, 7, 16)
    assert result['patch_tokens'] is None
    result['scene'].square().sum().backward()
    assert backbone.image.weight.grad is not None
