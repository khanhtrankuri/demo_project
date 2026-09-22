import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F
from src.models.surround_transfer import SurroundTransfer
from src.models.surround_transfer import balanced_contrastive_loss


def test_initial_adapters_preserve_pretrained_space_and_mask_missing_camera():
    model = SurroundTransfer(dict(width=16, bottleneck=8, dropout=.2, residual_scale=.2), 3).eval()
    features = F.normalize(torch.randn(3, 6, 16), dim=-1)
    mask = torch.tensor([[True]*6, [True, False, False, False, False, False], [True]*6])
    features[1, 1:] = float("nan")
    result = model(features, mask)
    expected = F.normalize(features.masked_fill(~mask[..., None], 0).sum(1), dim=-1)
    assert torch.allclose(result["view"][mask], features[mask], atol=1e-6)
    assert torch.allclose(result["scene"], expected, atol=1e-6)
    assert torch.isfinite(result["scene"]).all()


def test_adapter_can_learn_without_changing_input_features():
    model = SurroundTransfer(dict(width=16, bottleneck=8, dropout=.2, residual_scale=.2), 3)
    features = F.normalize(torch.randn(3, 6, 16), dim=-1)
    original = features.clone()
    result = model(features, torch.ones(3, 6, dtype=torch.bool), .5)
    loss = result["view"].sum() + result["scene"].sum() + result["object_logits"].square().mean()
    loss.backward()
    assert torch.equal(features, original)
    assert model.view_adapter.net[-1].weight.grad.abs().sum() > 0
    assert model.scene_adapter.net[-1].weight.grad.abs().sum() > 0


def test_balanced_loss_handles_repeated_text_without_false_negatives():
    images = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], requires_grad=True)
    texts = images.detach().clone()
    loss = balanced_contrastive_loss(images, texts, ["car", "car", "bus"], torch.tensor(20.))
    # Text-to-image distributes mass over two car images: irreducible log(2).
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.)).item() / 4, abs=1e-6)
    loss.backward()
    assert torch.isfinite(images.grad).all()


def test_all_dropped_cameras_fall_back_to_available_views():
    model = SurroundTransfer(dict(width=16, bottleneck=8, dropout=0., residual_scale=.2), 3).train()
    features = F.normalize(torch.randn(2, 6, 16), dim=-1)
    mask = torch.tensor([[True] * 6, [False, True, False, False, False, False]])
    expected = model(features, mask, 0.)
    actual = model(features, mask, 1.)
    assert torch.allclose(actual["scene"], expected["scene"], atol=1e-6)
    with pytest.raises(ValueError, match="needs a camera"):
        model(features, torch.zeros_like(mask))


@pytest.mark.parametrize("field", ["weights_hash", "backbone", "preprocess", "manifest_hash", "classes"])
def test_test_cache_must_match_selected_checkpoint(field):
    from surround_transfer import validate_checkpoint_cache
    identity = dict(weights_hash="weights", backbone="clip", preprocess="padding", manifest_hash="train")
    checkpoint = dict(signature=dict(train=identity, test_manifest="test"), classes=["car", "bus"])
    data = dict(identity=dict(identity, manifest_hash="test"), classes=["car", "bus"])
    validate_checkpoint_cache(checkpoint, data, "test")
    if field == "classes":
        data[field] = ["bus", "car"]
    else:
        data["identity"][field] = "changed"
    with pytest.raises(ValueError):
        validate_checkpoint_cache(checkpoint, data, "test")


def test_evaluation_keeps_full_gallery_and_duplicate_positives():
    from surround_transfer import evaluate
    model = SurroundTransfer(dict(width=2, bottleneck=2, dropout=0., residual_scale=.2), 2).eval()
    features = torch.eye(2)[:, None, :].expand(-1, 6, -1).clone()
    data = dict(features=features, mask=torch.ones(2, 6, dtype=torch.bool), view_text=features,
                scene_text=torch.eye(2), captions=[["car"] * 6, ["bus"] * 6], scene_captions=["car", "bus"])
    metrics = evaluate(model, data, "cpu")
    assert metrics["selection_score"] == 1.
    assert metrics["view"]["images"] == 12
    assert metrics["view"]["unique_captions"] == 2
    assert metrics["scene"]["images"] == 2
