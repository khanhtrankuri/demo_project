from train import caption_from_record, split_train_validation
from src.data.bdd100k import BDDRecord
from src.data.nuscenes import NuScenesRecord


def record(**overrides):
    values = {
        "image_id": "x.jpg",
        "image_path": "x.jpg",
        "source_split": "train",
        "weather": "clear",
        "timeofday": "daytime",
        "scene": "city street",
        "objects": ["car", "person"],
        "num_objects": 2,
        "annotation_path": "x.jpg.json",
    }
    values.update(overrides)
    return BDDRecord(**values)


def test_caption_uses_scene_attributes_and_objects():
    assert caption_from_record(record()) == (
        "a city street driving scene in clear weather during daytime with car, person."
    )


def test_caption_omits_placeholder_metadata():
    caption = caption_from_record(
        record(weather="undefined", timeofday="", scene="undefined", objects=[])
    )
    assert caption == "a driving scene."


def test_validation_split_is_deterministic_and_scene_separated():
    records = [
        NuScenesRecord(
            image_id=str(index), image_path=f"{index}.jpg", source_split="dev",
            weather="unknown", timeofday="unknown", scene="unknown", objects=[],
            num_objects=0, scene_token=f"scene-{index // 4}",
        )
        for index in range(40)
    ]
    train_a, validation_a = split_train_validation(records, 0.2, 42)
    train_b, validation_b = split_train_validation(list(reversed(records)), 0.2, 42)
    assert [record.image_id for record in train_a] == [record.image_id for record in train_b]
    assert [record.image_id for record in validation_a] == [record.image_id for record in validation_b]
    assert len(validation_a) == 8
    assert {record.scene_token for record in train_a}.isdisjoint(
        {record.scene_token for record in validation_a}
    )
