"""Train, evaluate, index, query and benchmark SurroundSearch-v4."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.hard_negative_sampler import HardNegativeSampler
from src.data.query_generator import generate_queries
from src.data.rich_caption import generate_scene_captions
from src.data.surround import CAMERAS, read_manifest
from src.data.surround_v4 import SurroundV4Dataset, collate_v4
from src.evaluation.semantic_retrieval import evaluate_rankings
from src.losses import MultiTaskLoss, hard_negative_loss, multi_positive_info_nce, positive_mask_from_ids
from src.models.backbones import CLIPBackbone, SigLIP2Backbone, SigLIPBackbone
from src.models.retrieval import CrossBatchMemoryQueue, SurroundDualEncoder
from src.utils.config import load_config, resolve_path
from src.utils.wandb_tracker import start_wandb


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def stable_id(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)


def precision_context(device, precision):
    if not str(device).startswith("cuda"):
        return nullcontext()
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def load_components(config, classes, device):
    from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel, Siglip2Model, SiglipModel

    settings = config["model"]
    local = resolve_path(config, settings["local_dir"])
    source = str(local) if (local / "config.json").is_file() else settings["repo"]
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(settings["precision"])
    if dtype is None:
        raise ValueError("model.precision must be one of: bf16, fp16, fp32")
    if not str(device).startswith("cuda"):
        dtype = torch.float32
    common = {"torch_dtype": dtype, "attn_implementation": "sdpa"}
    if source == str(local):
        common["local_files_only"] = True
    kind = settings["backbone"].lower()
    if kind == "clip":
        base = CLIPModel.from_pretrained(source, **common)
        backbone = CLIPBackbone(base, settings["vision_lora"], settings.get("text_lora"),
                                settings.get("gradient_checkpointing", True))
    elif kind == "siglip":
        base = SiglipModel.from_pretrained(source, **common)
        backbone = SigLIPBackbone(base, settings["vision_lora"], settings.get("text_lora"),
                                  settings.get("gradient_checkpointing", True))
    elif kind == "siglip2":
        base = Siglip2Model.from_pretrained(source, **common)
        backbone = SigLIP2Backbone(base, settings["vision_lora"], settings.get("text_lora"),
                                   settings.get("gradient_checkpointing", True))
    else:
        raise ValueError(f"Unsupported backbone: {kind}")
    processor = AutoImageProcessor.from_pretrained(source, local_files_only=source == str(local), use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=source == str(local))
    pretrained_size = getattr(base.config.vision_config, "image_size", None)
    if pretrained_size is not None and int(pretrained_size) != int(settings["image_resolution"]):
        raise ValueError("image_resolution must match the pretrained checkpoint")
    model = SurroundDualEncoder(backbone, len(classes), settings).to(device)
    return model, processor, tokenizer


def tokenize(tokenizer, texts, device, kind):
    kwargs = dict(padding=True, truncation=True, return_tensors="pt")
    if kind in ("siglip", "siglip2"):
        kwargs.update(padding="max_length", max_length=min(64, tokenizer.model_max_length))
    return tokenizer(texts, **kwargs).to(device)


def dataset_for(records, classes, processor, config, training=False):
    settings = config["model"]
    return SurroundV4Dataset(records, classes, settings["image_resolution"], processor.image_mean,
                             processor.image_std, settings["num_spatial_crops"], training)


def loader_for(records, classes, processor, config, training=False):
    batch = config["training"]["micro_batch_size"] if training else config["runtime"]["eval_batch_size"]
    workers = config["runtime"]["workers"]
    return DataLoader(dataset_for(records, classes, processor, config, training), batch_size=batch,
                      shuffle=training, num_workers=workers, persistent_workers=bool(workers and training),
                      pin_memory=torch.cuda.is_available(), collate_fn=collate_v4)


def flatten_captions(batch, level, multi_positive=True):
    captions, ids = [], []
    if level == "scene":
        for texts, identity in zip(batch["scene_captions"], batch["scene_id"]):
            chosen = texts if multi_positive else texts[:1]
            captions.extend(chosen)
            ids.extend([identity] * len(chosen))
    else:
        for per_camera, per_ids, mask in zip(batch["view_captions"], batch["view_ids"], batch["camera_mask"]):
            for texts, identity, present in zip(per_camera, per_ids, mask.tolist()):
                if present:
                    chosen = texts if multi_positive else texts[:1]
                    captions.extend(chosen)
                    ids.extend([identity] * len(chosen))
    return captions, ids


def queue_loss(images, texts, image_ids, text_ids, scale, image_queue, text_queue):
    positives = positive_mask_from_ids(image_ids, text_ids, images.device)
    queued_images, queued_image_ids = image_queue.get()
    queued_texts, queued_text_ids = text_queue.get()
    queued_images, queued_image_ids = queued_images.to(images.device), queued_image_ids.to(images.device)
    queued_texts, queued_text_ids = queued_texts.to(images.device), queued_text_ids.to(images.device)
    image_numeric = torch.tensor([stable_id(value) for value in image_ids], device=images.device)
    text_numeric = torch.tensor([stable_id(value) for value in text_ids], device=images.device)
    qi = text_numeric[:, None].eq(queued_image_ids[None]) if len(queued_image_ids) else None
    qt = image_numeric[:, None].eq(queued_text_ids[None]) if len(queued_text_ids) else None
    loss = multi_positive_info_nce(images, texts, positives, scale,
                                   queued_images if len(queued_images) else None,
                                   queued_texts if len(queued_texts) else None, qi, qt)
    return loss, image_numeric, text_numeric


def auxiliary_losses(output, batch, device):
    valid = batch["camera_mask"].to(device)
    object_loss = functional.binary_cross_entropy_with_logits(output["object_logits"][valid], batch["objects"].to(device)[valid])
    count_loss = functional.cross_entropy(output["count_logits"][valid].flatten(0, 1), batch["counts"].to(device)[valid].flatten())
    spatial_target = batch["spatial"].to(device)[valid]
    distance_target = batch["distance"].to(device)[valid]
    def masked_classification(logits, target):
        flattened = target.flatten()
        return (functional.cross_entropy(logits.flatten(0, 1), flattened, ignore_index=-100)
                if flattened.ne(-100).any() else logits.sum() * 0)
    spatial_loss = masked_classification(output["spatial_logits"][valid], spatial_target)
    distance_loss = masked_classification(output["distance_logits"][valid], distance_target)
    mean_view = (output["view"] * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
    regularization = (1 - functional.cosine_similarity(output["scene"], mean_view.detach())).mean()
    return {"object": object_loss, "count": count_loss, "spatial": spatial_loss,
            "distance": distance_loss, "regularization": regularization}


def train_loop(config, records, classes, model, processor, tokenizer, device, max_steps=None, tracker=None):
    settings, kind = config["training"], config["model"]["backbone"].lower()
    optimizer = torch.optim.AdamW(model.parameter_groups(settings["vision_lora_lr"], settings["text_lora_lr"],
                                                         settings["head_lr"]), weight_decay=settings["weight_decay"])
    objective = MultiTaskLoss(settings["loss"])
    dimension = model.backbone.projection_dim
    queue_dtype = torch.bfloat16 if config["model"]["precision"] == "bf16" else torch.float16
    queues = {name: CrossBatchMemoryQueue(settings["queue_size"], dimension, queue_dtype).to(device)
              for name in ("view_image", "view_text", "scene_image", "scene_text")}
    sampler = HardNegativeSampler(records, config["seed"]) if settings["loss"].get("hard_negative", 0) else None
    accumulation = max(1, settings["effective_batch_size"] // settings["micro_batch_size"])
    logs, forward_seconds, backward_seconds, step = [], 0.0, 0.0, 0
    optimizer_step, samples_seen = 0, 0
    window, window_microbatches, window_samples = {}, 0, 0
    window_forward, window_backward = 0.0, 0.0
    window_started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loader = loader_for(records, classes, processor, config, training=True)

    def update_parameters(epoch):
        nonlocal optimizer_step, window, window_microbatches, window_samples
        nonlocal window_forward, window_backward, window_started
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.logit_scale.clamp_(0, np.log(100))
        optimizer_step += 1
        if tracker is not None:
            elapsed = max(time.perf_counter() - window_started, 1e-9)
            metrics = {f"train/loss_{key}": value / window_microbatches for key, value in window.items()}
            metrics.update({
                "progress/optimizer_step": optimizer_step,
                "progress/micro_step": step,
                "progress/epoch": epoch + 1,
                "progress/samples_seen": samples_seen,
                "system/microbatches_per_optimizer_step": window_microbatches,
                "system/forward_seconds_per_optimizer_step": window_forward,
                "system/backward_seconds_per_optimizer_step": window_backward,
                "system/samples_per_second": window_samples / elapsed,
                "model/logit_scale": float(model.logit_scale.exp().detach()),
            })
            for index, group in enumerate(optimizer.param_groups):
                metrics[f"lr/{group.get('name', index)}"] = float(group["lr"])
            if device == "cuda":
                metrics.update({
                    "system/gpu_memory_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "system/gpu_memory_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                    "system/gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                })
            tracker.log(metrics)
        window, window_microbatches, window_samples = {}, 0, 0
        window_forward, window_backward = 0.0, 0.0
        window_started = time.perf_counter()

    for epoch in range(settings["epochs"]):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            images, mask = batch["images"].to(device, non_blocking=True), batch["camera_mask"].to(device, non_blocking=True)
            train_reranker = model.reranker is not None and len(batch["scene_id"]) > 1
            with precision_context(device, config["model"]["precision"]):
                output = model.encode_visual(images, mask, batch["geometry"].to(device),
                                             return_tokens=train_reranker,
                                             camera_dropout=settings["camera_dropout"])
                view_captions, view_text_ids = flatten_captions(batch, "view", settings.get("multi_positive", True))
                scene_captions, scene_text_ids = flatten_captions(batch, "scene", settings.get("multi_positive", True))
                view_text = model.encode_text(tokenize(tokenizer, view_captions, device, kind))["embedding"]
                scene_token_batch = tokenize(tokenizer, scene_captions, device, kind)
                scene_text_output = model.encode_text(scene_token_batch, return_tokens=train_reranker)
                scene_text = scene_text_output["embedding"]
                view_images = output["view"][mask]
                view_image_ids = [identity for identities, row_mask in zip(batch["view_ids"], batch["camera_mask"])
                                  for identity, present in zip(identities, row_mask.tolist()) if present]
                view_loss, view_image_numeric, view_text_numeric = queue_loss(
                    view_images, view_text, view_image_ids, view_text_ids, model.logit_scale.exp(),
                    queues["view_image"], queues["view_text"])
                scene_loss, scene_image_numeric, scene_text_numeric = queue_loss(
                    output["scene"], scene_text, batch["scene_id"], scene_text_ids, model.logit_scale.exp(),
                    queues["scene_image"], queues["scene_text"])
                pairs = (sampler.sample(batch["index"], settings["hard_negatives_per_anchor"])
                         if sampler is not None else [])
                if pairs:
                    row_for_global = {value: row for row, value in enumerate(batch["index"])}
                    usable = [pair for pair in pairs if pair.anchor in row_for_global]
                    negative_captions = [(records[pair.negative].get("captions") or generate_scene_captions(records[pair.negative]))[0]
                                         for pair in usable]
                    negative_text = model.encode_text(tokenize(tokenizer, negative_captions, device, kind))["embedding"]
                    anchor = torch.stack([output["scene"][row_for_global[pair.anchor]] for pair in usable])
                    positive = torch.stack([scene_text[scene_text_ids.index(batch["scene_id"][row_for_global[pair.anchor]])]
                                            for pair in usable])
                    hard = hard_negative_loss(anchor, positive, negative_text, settings["hard_negative_margin"])
                else:
                    hard = output["scene"].sum() * 0
                if train_reranker:
                    first_indices, cursor = [], 0
                    for identity in batch["scene_id"]:
                        first_indices.append(cursor)
                        cursor += scene_text_ids.count(identity)
                    query_tokens = scene_text_output["tokens"][first_indices]
                    candidates = output["scene_tokens"][None].expand(len(query_tokens), -1, -1, -1)
                    candidate_mask = torch.cat([torch.ones(len(mask), 1, dtype=torch.bool, device=device),
                                                output["mask"]], dim=1)
                    candidate_mask = candidate_mask[None].expand(len(query_tokens), -1, -1)
                    query_mask = scene_token_batch["attention_mask"][first_indices]
                    reranker_scores = model.reranker(query_tokens, candidates, query_mask, candidate_mask)
                    hard = hard + functional.cross_entropy(reranker_scores, torch.arange(len(query_tokens), device=device))
                losses = {"view": view_loss, "scene": scene_loss, "hard_negative": hard,
                          **auxiliary_losses(output, batch, device)}
                result = objective(losses)
            if device == "cuda":
                torch.cuda.synchronize()
            forward_delta = time.perf_counter() - started
            forward_seconds += forward_delta
            started = time.perf_counter()
            (result["total"] / accumulation).backward()
            if device == "cuda":
                torch.cuda.synchronize()
            backward_delta = time.perf_counter() - started
            backward_seconds += backward_delta
            if settings.get("queue_enabled", True):
                queues["view_image"].enqueue(view_images, view_image_numeric)
                queues["view_text"].enqueue(view_text, view_text_numeric)
                queues["scene_image"].enqueue(output["scene"], scene_image_numeric)
                queues["scene_text"].enqueue(scene_text, scene_text_numeric)
            step += 1
            samples_seen += len(batch["scene_id"])
            current = {key: float(value.detach()) for key, value in losses.items()} | {
                "total": float(result["total"].detach())}
            logs.append(current)
            for key, value in current.items():
                window[key] = window.get(key, 0.0) + value
            window_microbatches += 1
            window_samples += len(batch["scene_id"])
            window_forward += forward_delta
            window_backward += backward_delta
            pbar.set_postfix(loss=f"{current['total']:.3f}", opt_step=optimizer_step)
            if step % accumulation == 0:
                update_parameters(epoch)
            if max_steps and step >= max_steps:
                if step % accumulation:
                    update_parameters(epoch)
                return logs, forward_seconds, backward_seconds, optimizer
    if step % accumulation:
        update_parameters(settings["epochs"] - 1)
    return logs, forward_seconds, backward_seconds, optimizer


def adapter_state(model):
    return {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad}


def load_adapter(model, state):
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError("v4 checkpoint trainable parameter names do not match")
    with torch.no_grad():
        for name, value in parameters.items():
            value.copy_(state[name])


@torch.no_grad()
def encode_records(config, records, classes, model, processor, device, return_tokens=False):
    model.eval()
    scenes, views, scene_ids, view_rows, scene_tokens, scene_masks = [], [], [], [], [], []
    for batch in loader_for(records, classes, processor, config):
        with precision_context(device, config["model"]["precision"]):
            output = model.encode_visual(batch["images"].to(device), batch["camera_mask"].to(device),
                                         batch["geometry"].to(device), return_tokens=return_tokens)
        scenes.append(output["scene"].cpu())
        views.append(output["view"][batch["camera_mask"].to(device)].cpu())
        scene_ids.extend(batch["scene_id"])
        for index in batch["index"]:
            view_rows.extend(records[index]["views"][camera] for camera in CAMERAS if camera in records[index]["views"])
        if return_tokens:
            scene_tokens.append(output["scene_tokens"].cpu())
            scene_masks.append(output["mask"].cpu())
    return {"scene": torch.cat(scenes), "view": torch.cat(views), "scene_ids": scene_ids,
            "scene_mapping": records, "view_mapping": view_rows,
            "scene_tokens": torch.cat(scene_tokens) if scene_tokens else None,
            "scene_masks": torch.cat(scene_masks) if scene_masks else None}


def rank_queries(config, queries, encoded, model, tokenizer, device, records=None, classes=None, processor=None):
    rankings = {}
    kind = config["model"]["backbone"].lower()
    ids = encoded["scene_ids"]
    for query in queries:
        tokens = tokenize(tokenizer, [query["text"]], device, kind)
        text = model.encode_text(tokens)["embedding"][0].cpu()
        order = torch.argsort(encoded["scene"] @ text, descending=True)
        if records is not None and model.reranker is not None:
            candidate_count = int(config["model"]["reranker"].get("candidates", 50))
            reranked, _ = rerank_query(config, query["text"], order[:candidate_count].tolist(), records,
                                       classes, model, processor, tokenizer, device)
            seen = set(reranked)
            order = torch.tensor(reranked + [index for index in order.tolist() if index not in seen])
        rankings[query["query_id"]] = [ids[index] for index in order.tolist()]
    return rankings


@torch.no_grad()
def rerank_query(config, query_text, stage1_indices, records, classes, model, processor, tokenizer, device):
    """Rerank only supplied Top-K candidates; database embeddings stay fixed."""
    if model.reranker is None or not len(stage1_indices):
        return stage1_indices, torch.zeros(len(stage1_indices))
    candidates = [records[int(index)] for index in stage1_indices]
    encoded = encode_records(config, candidates, classes, model, processor, device, return_tokens=True)
    kind = config["model"]["backbone"].lower()
    token_batch = tokenize(tokenizer, [query_text], device, kind)
    text = model.encode_text(token_batch, return_tokens=True)
    visual_tokens = encoded["scene_tokens"].to(device)[None]
    camera_mask = encoded["scene_masks"].to(device)
    token_mask = torch.cat([torch.ones(len(candidates), 1, dtype=torch.bool, device=device), camera_mask], 1)
    visual_mask = token_mask[None]
    cross_scores = model.reranker(text["tokens"], visual_tokens, token_batch.get("attention_mask"), visual_mask)[0]
    camera_tokens = visual_tokens[0, :, 1:]
    conditioned = model.query_fusion(text["embedding"].expand(len(candidates), -1), camera_tokens, camera_mask)
    aware_scores = conditioned @ text["embedding"][0]
    combined = cross_scores + aware_scores
    order = torch.argsort(combined, descending=True).cpu()
    original = torch.as_tensor(stage1_indices, dtype=torch.long)
    return original[order].tolist(), combined[order.to(combined.device)].cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "train", "eval", "index", "query", "benchmark"))
    parser.add_argument("--config", default="configs/surround_v4.yaml")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--query")
    parser.add_argument("--level", choices=("view", "scene"), default="scene")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--smoke-steps", type=int, default=2)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    processed = resolve_path(config, config["dataset"]["processed_dir"])
    output = resolve_path(config, config["training"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    records = {name: read_manifest(processed / f"{name}.jsonl") for name in ("train", "val", "test")}
    classes = json.loads((processed / "stats.json").read_text(encoding="utf-8"))["classes"]
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, processor, tokenizer = load_components(config, classes, args.device)
    tracker = start_wandb(config, output, args.command)
    checkpoint_path = output / "best.pt"
    if args.command in ("smoke", "train"):
        try:
            selected = (records["train"][:max(config["training"]["micro_batch_size"] * args.smoke_steps, 2)]
                        if args.command == "smoke" else records["train"])
            logs, forward_time, backward_time, optimizer = train_loop(
                config, selected, classes, model, processor, tokenizer, args.device,
                args.smoke_steps if args.command == "smoke" else None, tracker)
            accumulation = max(1, config["training"]["effective_batch_size"]
                               // config["training"]["micro_batch_size"])
            report = {**model.parameter_counts(), "steps": len(logs),
                      "optimizer_steps": (len(logs) + accumulation - 1) // accumulation, "losses": logs,
                      "forward_seconds": forward_time, "backward_seconds": backward_time,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else 0,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if args.device == "cuda" else 0}
            report_path = output / ("smoke_report.json" if args.command == "smoke" else "train_report.json")
            checkpoint_path = output / ("smoke.pt" if args.command == "smoke" else "best.pt")
            save_json(report_path, report)
            payload = {"architecture": "SurroundSearch-v4", "adapter": adapter_state(model), "classes": classes,
                       "config": {key: value for key, value in config.items() if not key.startswith("_")}}
            torch.save(payload, checkpoint_path)
            if tracker is not None:
                tracker.complete(report, checkpoint_path)
            print(json.dumps(report, indent=2))
        except BaseException:
            if tracker is not None:
                tracker.finish(1)
            raise
        if tracker is not None:
            tracker.finish(0)
        return
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Train v4 first: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["classes"] != classes:
        raise ValueError("checkpoint classes differ from dataset")
    load_adapter(model, checkpoint["adapter"])
    selected = records[args.split]
    if args.command == "index":
        encoded = encode_records(config, selected, classes, model, processor, args.device)
        index_dir = output / f"index_{args.split}"
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "scene.npy", encoded["scene"].numpy())
        np.save(index_dir / "view.npy", encoded["view"].numpy())
        save_json(index_dir / "scene.json", [{"sample_token": value} for value in encoded["scene_ids"]])
        save_json(index_dir / "view.json", encoded["view_mapping"])
        print(index_dir)
        return
    if args.command == "query":
        if not args.query:
            parser.error("--query is required")
        index_dir = output / f"index_{args.split}"
        matrix = torch.from_numpy(np.load(index_dir / f"{args.level}.npy"))
        mapping = json.loads((index_dir / f"{args.level}.json").read_text())
        with torch.no_grad():
            query = model.encode_text(tokenize(tokenizer, [args.query], args.device, config["model"]["backbone"]))["embedding"][0].cpu()
        candidate_count = int(config["model"].get("reranker", {}).get("candidates", 50))
        stage1 = torch.argsort(matrix @ query, descending=True)[:candidate_count]
        if args.level == "scene":
            order, rerank_scores = rerank_query(config, args.query, stage1.tolist(), selected, classes,
                                                model, processor, tokenizer, args.device)
        else:
            order, rerank_scores = stage1.tolist(), torch.empty(0)
        result = []
        for rank, index in enumerate(order[:args.top_k]):
            score = float(rerank_scores[rank]) if len(rerank_scores) else float(matrix[index] @ query)
            result.append(mapping[index] | {"score": score})
        print(json.dumps(result, indent=2))
        return
    encoded = encode_records(config, selected, classes, model, processor, args.device)
    queries = generate_queries(selected)
    rankings = rank_queries(config, queries, encoded, model, tokenizer, args.device,
                            selected, classes, processor)
    metrics = evaluate_rankings(queries, rankings)
    save_json(output / f"semantic_{args.split}.json", metrics)
    if args.command == "benchmark":
        save_json(output / f"queries_{args.split}.json", queries)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
