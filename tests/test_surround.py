import json
from pathlib import Path

import numpy as np
import pytest

from src.data.surround import box_corners, boxes_in_camera, iter_json_array, scene_splits


def test_prepare_all_cameras_preserves_rows_and_projects_labels(tmp_path):
    from PIL import Image
    from src.data.surround import CAMERAS, prepare_dataset, read_manifest
    root = tmp_path / "raw"
    tables = root / "v1.0-trainval"
    tables.mkdir(parents=True)
    def save(name, value):
        (tables / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    pose = {"translation": [0, 0, 0], "rotation": [1, 0, 0, 0]}
    save("sensor", [{"token": c, "channel": c} for c in CAMERAS])
    save("calibrated_sensor", [dict(pose, token=c, sensor_token=c,
        camera_intrinsic=[[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        rotation=[0, 0, 1, 0] if c == "CAM_BACK" else [1, 0, 0, 0]) for c in CAMERAS])
    save("scene", [{"token": f"scene-{i}", "description": "A car on the road"} for i in range(3)])
    save("sample", [{"token": str(i), "scene_token": f"scene-{i}"} for i in range(3)])
    save("category", [{"token": "cat", "name": "vehicle.car"}])
    save("instance", [{"token": "instance", "category_token": "cat"}])
    save("ego_pose", [dict(pose, token="pose")])
    save("sample_annotation", [dict(pose, sample_token=str(i), instance_token="instance",
                                    translation=[0, 0, 5], size=[1, 1, 1]) for i in range(3)])
    data = []
    for i in range(3):
        for camera in CAMERAS:
            filename = f"samples/{camera}/{i}.jpg"
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (100, 100)).save(path)
            data.append(dict(token=f"{camera}-{i}", filename=filename, sample_token=str(i),
                             calibrated_sensor_token=camera, ego_pose_token="pose", timestamp=i,
                             width=100, height=100, is_key_frame=True))
    save("sample_data", data)
    output = tmp_path / "processed"
    stats = prepare_dataset(root, output)
    assert stats["images"] == 18 and stats["samples"] == 3
    assert stats["complete_samples"] == 3 and stats["rejected"] == 0
    rows = read_manifest(output / "images.jsonl")
    assert all(r["objects"] == ([] if r["camera"] == "CAM_BACK" else ["car"]) for r in rows)
    splits = [read_manifest(output / f"{name}.jsonl") for name in ("train", "val", "test")]
    assert all(len(rows) == 1 for rows in splits)
    assert len({r["scene_token"] for rows in splits for r in rows}) == 3


def test_projection_uses_camera_pose_and_depth():
    corners = box_corners({"size": [1, 1, 1], "translation": [0, 0, 5], "rotation": [1, 0, 0, 0]})
    pose = {"translation": [0, 0, 0], "rotation": [1, 0, 0, 0]}
    calibration = dict(pose, camera_intrinsic=[[100, 0, 50], [0, 100, 50], [0, 0, 1]])
    assert boxes_in_camera([corners], pose, calibration, 100, 100).tolist() == [True]
    back = dict(calibration, rotation=[0, 0, 1, 0])
    assert boxes_in_camera([corners], pose, back, 100, 100).tolist() == [False]
    shifted = corners + [10, 0, 0]
    assert boxes_in_camera([shifted], pose, calibration, 100, 100).tolist() == [False]
    assert boxes_in_camera([shifted], dict(pose, translation=[10, 0, 0]), calibration, 100, 100).tolist() == [True]
    assert boxes_in_camera([], pose, calibration, 100, 100).shape == (0,)


def test_group_split_preserves_old_test_and_every_scene():
    scenes = [f"scene-{i}" for i in range(30)]
    splits = scene_splits(scenes, 42, .1, .1, ["scene-4", "scene-8"])
    assert splits == scene_splits(reversed(scenes), 42, .1, .1, ["scene-8", "scene-4"])
    assert set(splits) == set(scenes)
    assert {s for s, split in splits.items() if split == "test"} == {"scene-4", "scene-8"}
    assert set(splits.values()) == {"train", "val", "test"}


def test_streaming_json_across_chunk_boundary(tmp_path):
    data = [{"token": "x", "text": "a" * (1024 * 1024 + 10)}, {"token": "y"}]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert list(iter_json_array(path)) == data
    path.write_text('[{"broken":', encoding="utf-8")
    with pytest.raises(ValueError, match="truncated"):
        list(iter_json_array(path))


def small_model():
    pytest.importorskip("torch")
    from src.models.surround import SurroundSearch
    config = dict(width=32, heads=4, dropout=.1, max_text_bytes=24, text_layers=1, fusion_layers=1)
    return SurroundSearch(config, 3)


def test_missing_camera_mask_prevents_pixel_leak_and_nan():
    torch = pytest.importorskip("torch")
    model = small_model().eval()
    images = torch.randn(2, 6, 3, 32, 64)
    mask = torch.tensor([[True, False, False, False, False, False], [True]*6])
    with torch.no_grad():
        before = model(images, mask)
        images[0, 1:] = float("nan")
        after = model(images, mask)
    assert torch.isfinite(after["scene"]).all()
    assert torch.allclose(before["scene"], after["scene"], atol=1e-6)
    assert torch.allclose(after["scene"].norm(dim=-1), torch.ones(2), atol=1e-5)
    with pytest.raises(ValueError, match="at least one"):
        model(images, torch.zeros_like(mask))


def test_losses_backpropagate_all_branches_and_tokenizer_is_bounded():
    torch = pytest.importorskip("torch")
    from src.models.surround import tokenize, contrastive_loss
    model = small_model()
    images = torch.randn(2, 6, 3, 32, 64)
    mask = torch.ones(2, 6, dtype=torch.bool)
    result = model(images, mask, camera_dropout=1.0)
    tokens = tokenize(["car" * 100, "xe máy"], 24)
    assert tokens.shape == (2, 24) and tokens.max() <= 258
    text = model.encode_text(tokens)
    loss = contrastive_loss(result["scene"], text, ["car", "xe máy"], model.logit_scale.exp())
    loss += result["object_logits"].square().mean()
    loss += contrastive_loss(result["view"][:, 0], text, ["car", "xe máy"], model.logit_scale.exp())
    loss.backward()
    for name in ("camera_embedding", "embedding.weight", "image_projection.weight", "object_head.weight"):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_full_gallery_metrics_allow_duplicate_caption_positives():
    torch = pytest.importorskip("torch")
    from surround import retrieval_metrics
    features = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    metrics = retrieval_metrics(features, torch.eye(2), torch.tensor([0, 0, 1]))
    assert metrics["image_to_text_recall_at_1"] == 1
    assert metrics["text_to_image_recall_at_1"] == 1


def test_leakage_guard_rejects_other_camera_of_same_sample():
    pytest.importorskip("torch")
    from surround import assert_disjoint
    row = {"scene_token": "s", "sample_token": "a", "views": {"CAM_FRONT": {"image_id": "f"}}}
    other = dict(row, views={"CAM_BACK": {"image_id": "b"}})
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint({"train": [row], "test": [other]})
