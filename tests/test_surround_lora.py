import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

from src.models.gradient_cache import cached_backward
from src.models.surround_lora import LoRALinear, SurroundLoRA


def settings():
    return dict(lora_rank=2, lora_alpha=4, lora_dropout=.1, lora_targets=["q_proj", "v_proj"],
                gradient_checkpointing=True, bottleneck=8, dropout=.1, residual_scale=.2)


def tiny_model():
    transformers = pytest.importorskip("transformers")
    config = transformers.CLIPConfig(
        projection_dim=16,
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, max_position_embeddings=8, eos_token_id=2, bos_token_id=1, pad_token_id=0),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                           num_attention_heads=2, image_size=8, patch_size=4))
    return SurroundLoRA(transformers.CLIPModel(config), settings(), 2)


def test_lora_starts_identical_then_learns_without_updating_base():
    base = nn.Linear(5, 4)
    original = copy.deepcopy(base.state_dict())
    layer = LoRALinear(base, rank=2, alpha=4, dropout=0.)
    x = torch.randn(3, 5)
    assert torch.equal(layer(x), base(x))
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=.1)
    layer(x).square().mean().backward()
    assert layer.lora_B.grad.abs().sum() > 0
    assert base.weight.grad is None and base.bias.grad is None
    optimizer.step()
    assert not torch.allclose(layer(x), base(x))
    for name, value in original.items():
        assert torch.equal(value, base.state_dict()[name])


def test_gradient_cache_matches_logical_batch_with_dropout_and_uneven_tail():
    torch.manual_seed(7)
    encoder = nn.Sequential(nn.Linear(4, 8), nn.Dropout(.3), nn.GELU(), nn.Linear(8, 3))
    head = nn.Linear(3, 3)
    reference_encoder, reference_head = copy.deepcopy(encoder), copy.deepcopy(head)
    images, mask = torch.randn(5, 4), torch.ones(5, dtype=torch.bool)
    target = F.normalize(torch.randn(5, 3), dim=-1)
    def objective(features, h):
        return F.cross_entropy(F.normalize(h(features), dim=-1) @ target.T, torch.arange(5))
    rng = torch.get_rng_state()
    cached_loss = cached_backward(lambda x, m: encoder(x), images, mask,
                                 lambda f: objective(f, head), micro_batch_size=2)
    post_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    full_features = torch.cat([reference_encoder(images[i:i+2]) for i in range(0, 5, 2)])
    reference_loss = objective(full_features, reference_head)
    reference_loss.backward()
    assert cached_loss == pytest.approx(reference_loss.item(), abs=1e-6)
    assert torch.equal(post_rng, torch.get_rng_state())
    for model, reference in ((encoder, reference_encoder), (head, reference_head)):
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            assert torch.allclose(parameter.grad, expected.grad, atol=2e-6, rtol=1e-5)


def test_only_vision_lora_and_heads_are_trainable_with_checkpointing():
    model = tiny_model().train()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all(n.startswith("heads.") or (n.startswith("clip.vision_model.") and ".lora_" in n) for n in trainable)
    assert model.parameter_counts()["target_projections"] == 4
    images, mask = torch.randn(2, 6, 3, 8, 8), torch.ones(2, 6, dtype=torch.bool)
    mask[1, 1:] = False
    images[1, 1:] = float("nan")
    features = model.encode_views(images, mask)
    assert torch.isfinite(features).all()
    prediction = model.heads(features, mask)
    loss = prediction["view"][mask].sum() + prediction["scene"].sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if ".lora_B" in n)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert not model.clip.text_model.training


def test_compact_adapter_roundtrip_excludes_base_and_is_strict():
    model = tiny_model().eval()
    for name, p in model.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(p, std=.01)
    state = model.adapter_state()
    assert not any("base.weight" in n or "text_model" in n for n in state)
    images, mask = torch.randn(1, 6, 3, 8, 8), torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        before = model.encode_views(images, mask)
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.zero_()
        model.load_adapter(state)
        after = model.encode_views(images, mask)
    assert torch.equal(before, after)
    state.pop(next(iter(state)))
    with pytest.raises(ValueError, match="parameter names"):
        model.load_adapter(state)


def test_dataset_keeps_six_cameras_and_uses_clip_normalization(tmp_path):
    from PIL import Image
    from src.data.surround import CAMERAS
    from src.data.surround_lora import LoRASurroundDataset
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (32, 8), (255, 0, 0)).save(image_path)
    processor = SimpleNamespace(crop_size=dict(height=8, width=8), image_mean=[.5]*3, image_std=[.5]*3)
    row = dict(caption="scene", views={CAMERAS[0]: dict(image_path=str(image_path), objects=["car"], caption="car")})
    dataset = LoRASurroundDataset([row], ["car"], processor)
    images, mask, labels, captions, description, _ = dataset[0]
    assert images.shape == (6, 3, 8, 8)
    assert mask.tolist() == [True, False, False, False, False, False]
    assert labels[0, 0] == 1 and labels.sum() == 1
    assert images[0, :, 4, 4].tolist() == pytest.approx([1., -1., -1.], abs=.1)
    assert images[0, :, 0, 0].abs().max() < .02
    assert captions[0] == "car" and description == "scene"


def test_lora_rejects_unknown_attention_targets():
    model = tiny_model()
    from src.models.surround_lora import inject_vision_lora
    with pytest.raises(ValueError, match="attention projections"):
        inject_vision_lora(model.clip, dict(settings(), lora_targets=["fc1"]))


def test_training_saves_compact_checkpoint_preserves_baseline_and_resumes(tmp_path, monkeypatch):
    import surround_lora as cli
    from PIL import Image
    from src.data.surround import CAMERAS
    torch.set_num_threads(2)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (12, 8), (80, 120, 40)).save(image_path)
    rows = [dict(caption="scene", views={camera: dict(image_path=str(image_path), caption="car", objects=["car"])
                                        for camera in CAMERAS}) for _ in range(3)]
    train_settings = dict(output_dir=str(tmp_path), epochs=1, effective_batch_size=2, micro_batch_size=1,
                          learning_rate=1e-4, head_learning_rate=1e-3, weight_decay=.05, warmup_fraction=.05,
                          patience=3, min_delta=.001, scene_weight=1., object_weight=.2, anchor_weight=1., camera_dropout=.1)
    config = dict(seed=42, model=dict(settings(), precision="fp32"), training=train_settings,
                  runtime=dict(workers=0, eval_batch_size=2, text_batch_size=8))
    bank = {caption: F.normalize(torch.randn(16), dim=0) for caption in ("car", "scene")}
    monkeypatch.setattr(cli, "text_bank", lambda *args: bank)
    processor = SimpleNamespace(crop_size=dict(height=8, width=8), image_mean=[.5]*3, image_std=[.5]*3)
    args = SimpleNamespace(command="train", device="cpu", resume=False)
    records = dict(train=rows, val=rows[:1])
    model = tiny_model()
    signature = dict(experiment="unit-test")
    cli.run_training(config, args, tmp_path, records, ["car", "bus"], model, processor, None, signature)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    last = torch.load(tmp_path / "last.pt", weights_only=False)
    assert best["epoch"] == 0  # Perfect one-caption baseline must not be discarded.
    assert last["epoch"] == 1 and last["global_step"] == 2
    assert "optimizer" not in best and "optimizer" in last
    assert not any("base.weight" in key for key in best["adapter"])
    args.resume = True
    cli.run_training(config, args, tmp_path, records, ["car", "bus"], model, processor, None, signature)
    resumed = torch.load(tmp_path / "last.pt", weights_only=False)
    assert resumed["global_step"] == 2
    for key, value in last["adapter"].items():
        assert torch.equal(value, resumed["adapter"][key])
    with pytest.raises(ValueError, match="Resume requires identical"):
        cli.run_training(config, args, tmp_path, records, ["car", "bus"], model, processor, None, dict(experiment="different"))
