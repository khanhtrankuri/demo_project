from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_nuscenes import (
    archive_signature,
    classify_member_names,
    deterministic_select,
    discover_archives,
    extract_archive,
    inspect_archive,
    join_cam_front_records,
    normalize_csv,
    scene_level_split,
    selection_target,
    simplify_category,
    validate_records,
)
from src.data.nuscenes import normalize_object, normalize_scene, record_from_row


def make_tar(path: Path, files: dict[str, bytes], mode: str = "w") -> None:
    with tarfile.open(path, mode) as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def write_json(root: Path, name: str, value: object) -> None:
    metadata = root / "v1.0-trainval"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / name).write_text(json.dumps(value), encoding="utf-8")


def jpeg_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(stream, "JPEG")
    return stream.getvalue()


def metadata_fixture(root: Path) -> None:
    write_json(root, "sensor.json", [
        {"token": "sensor-front", "channel": "CAM_FRONT", "modality": "camera"},
        {"token": "sensor-back", "channel": "CAM_BACK", "modality": "camera"},
    ])
    write_json(root, "calibrated_sensor.json", [
        {"token": "cal-front", "sensor_token": "sensor-front"},
        {"token": "cal-back", "sensor_token": "sensor-back"},
    ])
    write_json(root, "log.json", [{"token": "log", "location": "singapore-onenorth"}])
    write_json(root, "scene.json", [{"token": "scene", "name": "scene-0001", "description": "Rain at night", "log_token": "log"}])
    write_json(root, "sample.json", [{"token": "sample", "scene_token": "scene", "timestamp": 1}])
    write_json(root, "category.json", [{"token": "cat", "name": "vehicle.car"}])
    write_json(root, "instance.json", [{"token": "instance", "category_token": "cat"}])
    write_json(root, "sample_annotation.json", [{"token": "ann", "sample_token": "sample", "instance_token": "instance"}])
    write_json(root, "sample_data.json", [
        {"token": "front-key", "sample_token": "sample", "calibrated_sensor_token": "cal-front", "filename": "samples/CAM_FRONT/a.jpg", "is_key_frame": True, "timestamp": 2},
        {"token": "front-sweep", "sample_token": "sample", "calibrated_sensor_token": "cal-front", "filename": "sweeps/CAM_FRONT/b.jpg", "is_key_frame": False, "timestamp": 3},
        {"token": "back-key", "sample_token": "sample", "calibrated_sensor_token": "cal-back", "filename": "samples/CAM_BACK/c.jpg", "is_key_frame": True, "timestamp": 4},
    ])


def test_discover_nuscenes_taz(tmp_path: Path) -> None:
    directory = tmp_path / "nuScenes"
    directory.mkdir()
    (directory / "b.taz").touch()
    (directory / "a.taz").touch()
    (directory / "ignored.tgz").touch()
    assert [p.name for p in discover_archives(directory)] == ["a.taz", "b.taz"]


def test_archive_format_detection(tmp_path: Path) -> None:
    path = tmp_path / "data.taz"
    make_tar(path, {"scene.json": b"[]"}, "w:gz")
    assert archive_signature(path) == "tar.gz (gzip-compressed tar)"
    assert inspect_archive(path).format == "tar.gz (gzip-compressed tar)"


@pytest.mark.parametrize(("names", "expected"), [
    (["v1.0-trainval/scene.json", "v1.0-trainval/sample.json"], "METADATA"),
    (["samples/CAM_FRONT/a.jpg", "samples/CAM_BACK/b.jpg"], "KEYFRAME_BLOBS"),
    (["README"], "UNKNOWN"),
])
def test_archive_classification(names: list[str], expected: str) -> None:
    assert classify_member_names(names)[0] == expected


def test_safe_tar_extraction_and_idempotence(tmp_path: Path) -> None:
    good = tmp_path / "good.taz"
    target = tmp_path / "out"
    make_tar(good, {"samples/CAM_FRONT/a.jpg": b"abc"})
    assert extract_archive(good, target)["extracted"] == 1
    assert extract_archive(good, target)["skipped"] == 1
    bad = tmp_path / "bad.taz"
    make_tar(bad, {"../../escape.txt": b"bad"})
    with pytest.raises(ValueError, match="Unsafe"):
        extract_archive(bad, target)
    assert not (tmp_path / "escape.txt").exists()


def test_cam_front_keyframe_filter_and_token_joins(tmp_path: Path) -> None:
    metadata_fixture(tmp_path)
    image = tmp_path / "samples" / "CAM_FRONT" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(jpeg_bytes())
    rows = join_cam_front_records(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["sample_data_token"] == "front-key"
    assert row["camera_channel"] == "CAM_FRONT"
    assert row["is_key_frame"] is True
    assert row["scene_token"] == "scene"
    assert row["location"] == "singapore-onenorth"
    assert row["objects_raw"] == ["vehicle.car"]
    assert row["objects"] == ["car"]
    assert row["weather_tag"] == "rainy"
    assert row["timeofday_tag"] == "nighttime"


@pytest.mark.parametrize(("raw", "simple"), [
    ("vehicle.car", "car"), ("vehicle.bus.rigid", "bus"),
    ("human.pedestrian.adult", "person"), ("animal", "animal"),
])
def test_category_simplification(raw: str, simple: str) -> None:
    assert simplify_category(raw) == simple


def test_deterministic_10k_selection() -> None:
    rows = [{"sample_data_token": f"token-{index:05d}"} for index in range(10_010)]
    first = deterministic_select(rows, 10_000, 42)
    second = deterministic_select(list(reversed(rows)), 10_000, 42)
    assert len(first) == 10_000
    assert [r["sample_data_token"] for r in first] == [r["sample_data_token"] for r in second]


def test_full_selection_keeps_every_valid_image_in_stable_order() -> None:
    rows = [{"sample_data_token": token} for token in ("c", "a", "b")]
    assert selection_target("all", len(rows)) == 3
    assert selection_target(None, len(rows)) == 3
    assert [row["sample_data_token"] for row in deterministic_select(rows, "all", 42)] == ["a", "b", "c"]
    with pytest.raises(ValueError, match="only 3"):
        deterministic_select(rows, 4, 42)
    with pytest.raises(ValueError, match="positive integer"):
        deterministic_select(rows, "everything", 42)


def test_fractional_full_scene_split_has_no_leakage() -> None:
    rows = [{"scene_token": f"scene-{index // 4}", "sample_data_token": str(index)} for index in range(120)]
    test_target = round(len(rows) * .2)
    dev, test = scene_level_split(rows, len(rows) - test_target, test_target, 42)
    assert (len(dev), len(test)) == (96, 24)
    assert {r["scene_token"] for r in dev}.isdisjoint({r["scene_token"] for r in test})


def test_scene_level_split_has_no_leakage() -> None:
    rows = [{"scene_token": f"scene-{index // 10}", "sample_data_token": str(index)} for index in range(100)]
    dev, test = scene_level_split(rows, 80, 20, 42)
    assert (len(dev), len(test)) == (80, 20)
    assert {r["scene_token"] for r in dev}.isdisjoint({r["scene_token"] for r in test})


def test_missing_image_handling(tmp_path: Path) -> None:
    row = {
        "image_path": str(tmp_path / "missing.jpg"), "sample_token": "sample",
        "sample_data_token": "data", "scene_token": "scene",
        "camera_channel": "CAM_FRONT", "is_key_frame": True,
    }
    valid, rejected = validate_records([row], workers=1)
    assert not valid
    assert rejected[0]["reason"] == "missing"


def test_nuscenes_normalization_uses_canonical_schema(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(jpeg_bytes())
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "image_id,image_path,scene_description,objects,num_annotations,weather_tag,timeofday_tag,split\n"
        f"frame,{image},Parking lot at night,vehicle.car|movable_object.trafficcone,2,unknown,unknown,dev\n",
        encoding="utf-8",
    )
    assert normalize_csv(metadata) == 1
    with metadata.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["objects"] == "car|traffic cone"
    assert row["scene"] == "parking lot"
    assert row["timeofday"] == "nighttime"
    assert row["source_split"] == "dev"
    record = record_from_row(row, metadata)
    assert record.num_objects == 2


def test_normalization_does_not_invent_weather_or_time() -> None:
    assert normalize_object("human.pedestrian.adult") == "person"
    assert normalize_scene("Cars stopped at an intersection") == "intersection"
