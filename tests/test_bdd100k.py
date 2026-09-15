import json
from pathlib import Path

from PIL import Image

from src.data.bdd100k import iter_split_records, parse_annotation


def test_datasetninja_parser(tmp_path: Path):
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (8, 8), "red").save(image)
    annotation = tmp_path / "sample.jpg.json"
    annotation.write_text(json.dumps({"tags": [{"name": "weather", "value": "rainy"}, {"name": "scene", "value": "city street"}, {"name": "timeofday", "value": "night"}], "objects": [{"classTitle": "car"}, {"classTitle": "car"}, {"classTitle": "person"}]}))
    record = parse_annotation(annotation, image, "train")
    assert record.weather == "rainy"
    assert record.objects == ["car", "person"]
    assert record.num_objects == 3
    assert record.has_meaningful_metadata


def test_split_sidecar_matching(tmp_path: Path):
    image_dir, ann_dir = tmp_path / "train" / "img", tmp_path / "train" / "ann"
    image_dir.mkdir(parents=True); ann_dir.mkdir(parents=True)
    image = image_dir / "x.jpg"; Image.new("RGB", (4, 4)).save(image)
    (ann_dir / "x.jpg.json").write_text(json.dumps({"tags": [], "objects": []}))
    records = list(iter_split_records(tmp_path, "train"))
    assert len(records) == 1 and records[0].image_id == "x.jpg"

