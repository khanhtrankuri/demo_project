import torch

from src.data.hard_negative_sampler import HardNegativeSampler, classify_negative
from src.losses.contrastive import multi_positive_info_nce, positive_mask_from_ids
from src.models.retrieval.memory_queue import CrossBatchMemoryQueue


def _record(sample, side="left", count=1, distance="near", camera="CAM_FRONT", scene=None):
    objects = [{"class": "car", "instance_token": f"{sample}-{i}", "spatial": side,
                "distance_bin": distance, "camera": camera} for i in range(count)]
    return {"sample_token": sample, "scene_token": scene or sample,
            "views": {camera: {"camera": camera, "object_instances": objects}}}


def test_multi_positive_loss_does_not_treat_paraphrases_as_negatives():
    images = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    texts = torch.tensor([[1., 0.], [.99, .01], [0., 1.]])
    positives = positive_mask_from_ids([10, 20], [10, 10, 20])
    loss = multi_positive_info_nce(images, texts, positives, 20.0)
    assert loss < .01
    loss.backward()
    assert torch.isfinite(images.grad).all()


def test_memory_queue_wraps_and_never_keeps_graph():
    queue = CrossBatchMemoryQueue(3, 2, torch.float16)
    values = torch.arange(8, dtype=torch.float32).reshape(4, 2).requires_grad_()
    queue.enqueue(values, torch.arange(4))
    embeddings, ids = queue.get()
    assert ids.tolist() == [1, 2, 3]
    assert not embeddings.requires_grad and embeddings.dtype == torch.float16
    queue.enqueue(torch.tensor([[9., 9.]]), torch.tensor([9]))
    _, ids = queue.get()
    assert ids.tolist() == [2, 3, 9]


def test_hard_negative_types_are_annotation_driven():
    anchor = _record("a", side="left", count=3)
    assert classify_negative(anchor, _record("b", side="left", count=1)) == "count"
    assert classify_negative(anchor, _record("c", side="right", count=3)) == "semantic"
    assert classify_negative(anchor, _record("d", side="left", count=3, distance="far")) == "distance"
    temporal = _record("e", side="right", count=3, scene="shared")
    same_scene = _record("f", side="left", count=3, scene="shared")
    assert classify_negative(temporal, same_scene) == "temporal"
    pairs = HardNegativeSampler([anchor, _record("g", side="right", count=3)]).sample([0])
    assert pairs[0].kind == "semantic"
    assert classify_negative(anchor, _record("same", side="left", count=3)) == "equivalent"


def test_fast_sampler_exhausts_eligible_candidates_without_duplicates():
    records = [_record('a', count=3), _record('equivalent', count=3),
               _record('count', count=1), _record('spatial', count=3, side='right'),
               _record('distance', count=3, distance='far')]
    sampler = HardNegativeSampler(records)
    pairs = sampler.sample([0], per_anchor=20)
    assert {p.negative for p in pairs} == {2, 3, 4}
    assert len(pairs) == 3
    assert pairs == sampler.sample([0], per_anchor=20)
    assert all(p.kind == classify_negative(records[p.anchor], records[p.negative]) for p in pairs)
    assert [p.negative for p in sampler.sample([0], 20, kinds=('distance',))] == [4]
    assert HardNegativeSampler(records[:1]).sample([0]) == []
