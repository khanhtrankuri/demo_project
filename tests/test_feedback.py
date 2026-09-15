import json

from src.utils.feedback import append_feedback


def test_feedback_is_appended_as_jsonl(tmp_path):
    path = tmp_path / "feedback.jsonl"
    append_feedback(path, {"image_id": "one.jpg", "relevant": True})
    append_feedback(path, {"image_id": "two.jpg", "relevant": False})
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {"image_id": "one.jpg", "relevant": True},
        {"image_id": "two.jpg", "relevant": False},
    ]
