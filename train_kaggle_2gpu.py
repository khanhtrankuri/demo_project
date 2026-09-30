"""Distributed SurroundSearch-v4 training for a Kaggle 2 x NVIDIA T4 runtime.

Launch from a Kaggle notebook cell with::

    !torchrun --standalone --nproc_per_node=2 train_kaggle_2gpu.py \
        --data-dir /kaggle/input/<dataset>/nuscenes_surround \
        --output-dir /kaggle/working/surround_v4

``micro_batch_size`` is per GPU; ``effective_batch_size`` is global across all
GPUs and gradient-accumulation steps.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.data.hard_negative_sampler import HardNegativeSampler
from src.data.rich_caption import generate_scene_captions
from src.data.surround import read_manifest
from src.data.surround_v4 import SurroundV4Dataset, collate_v4
from src.losses import MultiTaskLoss, hard_negative_loss
from src.models.retrieval import CrossBatchMemoryQueue
from src.utils.config import load_config, resolve_path
from src.utils.wandb_tracker import start_wandb
from surround_v4 import (
    adapter_state,
    auxiliary_losses,
    flatten_captions,
    load_adapter,
    load_components,
    precision_context,
    queue_loss,
    save_json,
    tokenize,
)


def accumulation_steps(effective_batch_size: int, micro_batch_size: int, world_size: int) -> int:
    """Return exact DDP accumulation count for a requested global batch."""
    physical_batch = micro_batch_size * world_size
    if min(effective_batch_size, micro_batch_size, world_size) < 1:
        raise ValueError("batch sizes and world size must be positive")
    if effective_batch_size % physical_batch:
        raise ValueError(
            "effective_batch_size must be divisible by micro_batch_size * world_size "
            f"({effective_batch_size} is not divisible by {physical_batch})"
        )
    return effective_batch_size // physical_batch


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI settings after YAML loading so Kaggle paths stay portable."""
    if args.data_dir:
        config["dataset"]["processed_dir"] = os.path.expanduser(args.data_dir)
    if args.output_dir:
        config["training"]["output_dir"] = os.path.expanduser(args.output_dir)
    for name in ("epochs", "micro_batch_size", "effective_batch_size"):
        value = getattr(args, name)
        if value is not None:
            config["training"][name] = value
    if args.workers is not None:
        config["runtime"]["workers"] = args.workers
    config["model"]["precision"] = args.precision
    if args.no_wandb:
        config.setdefault("wandb", {})["enabled"] = False
    return config


class DistributedTrainingStep(nn.Module):
    """Make every trainable branch participate in one DDP forward call."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        images,
        camera_mask,
        geometry,
        view_tokens,
        scene_tokens,
        negative_tokens=None,
        reranker_first_indices=None,
        *,
        camera_dropout: float = 0.0,
        train_reranker: bool = False,
    ):
        visual = self.model.encode_visual(
            images,
            camera_mask,
            geometry,
            return_tokens=train_reranker,
            camera_dropout=camera_dropout,
        )
        view_text = self.model.encode_text(view_tokens)["embedding"]
        scene_text = self.model.encode_text(scene_tokens, return_tokens=train_reranker)
        negative_text = (
            self.model.encode_text(negative_tokens)["embedding"]
            if negative_tokens is not None
            else None
        )
        reranker_scores = None
        if train_reranker:
            query_tokens = scene_text["tokens"][reranker_first_indices]
            candidates = visual["scene_tokens"][None].expand(
                len(query_tokens), -1, -1, -1
            )
            candidate_mask = torch.cat(
                [
                    torch.ones(
                        len(camera_mask), 1, dtype=torch.bool, device=camera_mask.device
                    ),
                    visual["mask"],
                ],
                dim=1,
            )
            candidate_mask = candidate_mask[None].expand(len(query_tokens), -1, -1)
            reranker_scores = self.model.reranker(
                query_tokens,
                candidates,
                scene_tokens["attention_mask"][reranker_first_indices],
                candidate_mask,
            )
        return (
            visual,
            view_text,
            scene_text,
            negative_text,
            reranker_scores,
            self.model.logit_scale.exp(),
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/surround_v4.yaml")
    parser.add_argument("--data-dir", help="Directory containing train.jsonl and stats.json")
    parser.add_argument("--output-dir", default="/kaggle/working/surround_v4")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers per GPU")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, help="Stop after this many micro-steps per rank")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume from a checkpoint path; without a value, use OUTPUT/last.pt",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--allow-non-t4",
        action="store_true",
        help="Allow two CUDA GPUs other than T4 (useful for local verification)",
    )
    return parser.parse_args(argv)


def setup_distributed(allow_non_t4: bool):
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "This trainer must be launched with torchrun. On Kaggle use: "
            "torchrun --standalone --nproc_per_node=2 train_kaggle_2gpu.py ..."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Kaggle 2-GPU training")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"Expected exactly 2 torchrun processes, got WORLD_SIZE={world_size}")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"WORLD_SIZE={world_size}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
        )
    torch.cuda.set_device(local_rank)
    names = [torch.cuda.get_device_name(index) for index in range(world_size)]
    if not allow_non_t4 and any("T4" not in name.upper() for name in names):
        raise RuntimeError(f"Expected two NVIDIA T4 GPUs, found: {names}; pass --allow-non-t4 to override")
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    return rank, local_rank, world_size, torch.device("cuda", local_rank), names


def seed_everything(seed: int, rank: int):
    rank_seed = seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)


def build_loader(records, classes, processor, config, sampler):
    model_settings = config["model"]
    dataset = SurroundV4Dataset(
        records,
        classes,
        model_settings["image_resolution"],
        processor.image_mean,
        processor.image_std,
        model_settings["num_spatial_crops"],
        training=True,
    )
    workers = int(config["runtime"]["workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["micro_batch_size"]),
        sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
        collate_fn=collate_v4,
        drop_last=False,
    )


def cosine_schedule(optimizer, total_steps: int, warmup_ratio: float):
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    warmup_steps = int(total_steps * warmup_ratio)

    def factor(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def average_metrics(values: dict[str, float], count: int, device) -> dict[str, float]:
    keys = sorted(values)
    tensor = torch.tensor([values[key] for key in keys] + [float(count)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    denominator = max(float(tensor[-1]), 1.0)
    return {key: float(tensor[index] / denominator) for index, key in enumerate(keys)}


def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, optimizer_step, config, classes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "architecture": "SurroundSearch-v4",
            "adapter": adapter_state(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "classes": classes,
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, scaler, scheduler, classes):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "SurroundSearch-v4":
        raise ValueError(f"Not a SurroundSearch-v4 checkpoint: {path}")
    if payload["classes"] != classes:
        raise ValueError("Checkpoint classes differ from dataset classes")
    load_adapter(model, payload["adapter"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload.get("scaler", {}))
    scheduler.load_state_dict(payload["scheduler"])
    return int(payload["epoch"]) + 1, int(payload["optimizer_step"])


def train(args):
    # Kaggle's two T4 workers do not have InfiniBand; these defaults avoid slow
    # transport probing and are still overridable from the notebook environment.
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    rank, local_rank, world_size, device, gpu_names = setup_distributed(args.allow_non_t4)
    tracker = None
    exit_code = 1
    try:
        config = apply_overrides(load_config(args.config), args)
        settings = config["training"]
        accumulation = accumulation_steps(
            int(settings["effective_batch_size"]),
            int(settings["micro_batch_size"]),
            world_size,
        )
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("These GPUs do not support BF16; use the default --precision fp16 on T4")
        seed_everything(int(config["seed"]), rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        processed = resolve_path(config, config["dataset"]["processed_dir"])
        output_dir = resolve_path(config, config["training"]["output_dir"])
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        required = [processed / "train.jsonl", processed / "stats.json"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing prepared v4 data files: " + ", ".join(missing))
        records = read_manifest(processed / "train.jsonl")
        classes = json.loads((processed / "stats.json").read_text(encoding="utf-8"))["classes"]
        if not records:
            raise ValueError("train.jsonl is empty")

        # Rank 0 populates the Hugging Face cache before rank 1 opens the same files.
        if rank == 0:
            core, processor, tokenizer = load_components(config, classes, "cuda")
        dist.barrier()
        if rank != 0:
            core, processor, tokenizer = load_components(config, classes, "cuda")
        dist.barrier()

        optimizer = torch.optim.AdamW(
            core.parameter_groups(
                settings["vision_lora_lr"], settings["text_lora_lr"], settings["head_lr"]
            ),
            weight_decay=settings["weight_decay"],
        )
        sampler = DistributedSampler(
            records,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(config["seed"]),
            drop_last=False,
        )
        loader = build_loader(records, classes, processor, config, sampler)
        optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation)
        total_optimizer_steps = max(1, int(settings["epochs"]) * optimizer_steps_per_epoch)
        scheduler = cosine_schedule(optimizer, total_optimizer_steps, args.warmup_ratio)
        scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
        objective = MultiTaskLoss(settings["loss"])
        queue_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
        queues = {
            name: CrossBatchMemoryQueue(settings["queue_size"], core.backbone.projection_dim, queue_dtype).to(device)
            for name in ("view_image", "view_text", "scene_image", "scene_text")
        }
        negative_sampler = (
            HardNegativeSampler(records, int(config["seed"]))
            if settings["loss"].get("hard_negative", 0)
            else None
        )
        training_module = DistributedTrainingStep(core)
        ddp = DDP(
            training_module,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

        start_epoch, optimizer_step = 0, 0
        checkpoint_path = output_dir / "last.pt"
        if args.resume:
            resume_path = checkpoint_path if args.resume == "auto" else Path(args.resume).expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            start_epoch, optimizer_step = load_checkpoint(
                resume_path, core, optimizer, scaler, scheduler, classes
            )
        if rank == 0:
            tracker = start_wandb(config, output_dir, "train")
            print(
                json.dumps(
                    {
                        "gpus": gpu_names,
                        "world_size": world_size,
                        "per_gpu_micro_batch": settings["micro_batch_size"],
                        "gradient_accumulation": accumulation,
                        "global_effective_batch": settings["effective_batch_size"],
                        "precision": args.precision,
                        "train_samples": len(records),
                        "start_epoch": start_epoch,
                    },
                    indent=2,
                ),
                flush=True,
            )

        optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        samples_seen = 0
        loss_window: dict[str, float] = {}
        window_count = 0
        started = time.perf_counter()
        core.train()

        for epoch in range(start_epoch, int(settings["epochs"])):
            sampler.set_epoch(epoch)
            iterator = tqdm(loader, desc=f"Epoch {epoch + 1}", disable=rank != 0)
            for batch_index, batch in enumerate(iterator):
                group_start = (batch_index // accumulation) * accumulation
                group_size = min(accumulation, len(loader) - group_start)
                should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
                images = batch["images"].to(device, non_blocking=True)
                mask = batch["camera_mask"].to(device, non_blocking=True)
                geometry = batch["geometry"].to(device, non_blocking=True)
                view_captions, view_text_ids = flatten_captions(
                    batch, "view", settings.get("multi_positive", True)
                )
                scene_captions, scene_text_ids = flatten_captions(
                    batch, "scene", settings.get("multi_positive", True)
                )
                kind = config["model"]["backbone"].lower()
                view_token_batch = tokenize(tokenizer, view_captions, device, kind)
                scene_token_batch = tokenize(tokenizer, scene_captions, device, kind)
                pairs = (
                    negative_sampler.sample(batch["index"], settings["hard_negatives_per_anchor"])
                    if negative_sampler is not None
                    else []
                )
                row_for_global = {value: row for row, value in enumerate(batch["index"])}
                usable_pairs = [pair for pair in pairs if pair.anchor in row_for_global]
                negative_captions = [
                    (
                        records[pair.negative].get("captions")
                        or generate_scene_captions(records[pair.negative])
                    )[0]
                    for pair in usable_pairs
                ]
                negative_token_batch = (
                    tokenize(tokenizer, negative_captions, device, kind) if negative_captions else None
                )
                train_reranker = core.reranker is not None and len(batch["scene_id"]) > 1
                first_indices, cursor = [], 0
                if train_reranker:
                    for identity in batch["scene_id"]:
                        first_indices.append(cursor)
                        cursor += scene_text_ids.count(identity)
                reranker_first_indices = (
                    torch.tensor(first_indices, dtype=torch.long, device=device)
                    if train_reranker
                    else None
                )

                sync_context = nullcontext() if should_step else ddp.no_sync()
                with sync_context:
                    with precision_context(device, args.precision):
                        (
                            output,
                            view_text,
                            scene_text_output,
                            negative_text,
                            reranker_scores,
                            logit_scale,
                        ) = ddp(
                            images,
                            mask,
                            geometry,
                            view_token_batch,
                            scene_token_batch,
                            negative_token_batch,
                            reranker_first_indices,
                            camera_dropout=settings["camera_dropout"],
                            train_reranker=train_reranker,
                        )
                        scene_text = scene_text_output["embedding"]
                        view_images = output["view"][mask]
                        view_image_ids = [
                            identity
                            for identities, row_mask in zip(batch["view_ids"], batch["camera_mask"])
                            for identity, present in zip(identities, row_mask.tolist())
                            if present
                        ]
                        view_loss, view_image_numeric, view_text_numeric = queue_loss(
                            view_images,
                            view_text,
                            view_image_ids,
                            view_text_ids,
                            logit_scale,
                            queues["view_image"],
                            queues["view_text"],
                        )
                        scene_loss, scene_image_numeric, scene_text_numeric = queue_loss(
                            output["scene"],
                            scene_text,
                            batch["scene_id"],
                            scene_text_ids,
                            logit_scale,
                            queues["scene_image"],
                            queues["scene_text"],
                        )
                        if usable_pairs:
                            anchor = torch.stack(
                                [output["scene"][row_for_global[pair.anchor]] for pair in usable_pairs]
                            )
                            positive = torch.stack(
                                [
                                    scene_text[
                                        scene_text_ids.index(
                                            batch["scene_id"][row_for_global[pair.anchor]]
                                        )
                                    ]
                                    for pair in usable_pairs
                                ]
                            )
                            hard = hard_negative_loss(
                                anchor,
                                positive,
                                negative_text,
                                settings["hard_negative_margin"],
                            )
                        else:
                            hard = output["scene"].sum() * 0
                        if train_reranker:
                            hard = hard + functional.cross_entropy(
                                reranker_scores,
                                torch.arange(len(query_tokens), device=device),
                            )
                        losses = {
                            "view": view_loss,
                            "scene": scene_loss,
                            "hard_negative": hard,
                            **auxiliary_losses(output, batch, device),
                        }
                        result = objective(losses)
                    scaler.scale(result["total"] / group_size).backward()

                if settings.get("queue_enabled", True):
                    queues["view_image"].enqueue(view_images, view_image_numeric)
                    queues["view_text"].enqueue(view_text, view_text_numeric)
                    queues["scene_image"].enqueue(output["scene"], scene_image_numeric)
                    queues["scene_text"].enqueue(scene_text, scene_text_numeric)

                current = {key: float(value.detach()) for key, value in losses.items()}
                current["total"] = float(result["total"].detach())
                for key, value in current.items():
                    loss_window[key] = loss_window.get(key, 0.0) + value
                window_count += 1
                micro_step += 1
                samples_seen += len(batch["scene_id"]) * world_size

                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in core.parameters() if parameter.requires_grad], 1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    with torch.no_grad():
                        core.logit_scale.clamp_(0, np.log(100))
                    optimizer_step += 1
                    reduced = average_metrics(loss_window, window_count, device)
                    if rank == 0:
                        iterator.set_postfix(loss=f"{reduced['total']:.3f}", step=optimizer_step)
                        metrics = {f"train/loss_{key}": value for key, value in reduced.items()}
                        metrics.update(
                            {
                                "progress/optimizer_step": optimizer_step,
                                "progress/micro_step": micro_step,
                                "progress/epoch": epoch + 1,
                                "progress/samples_seen": samples_seen,
                                "system/world_size": world_size,
                                "system/gradient_accumulation": accumulation,
                                "system/gpu_memory_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                                "system/gpu_memory_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                                "model/logit_scale": float(core.logit_scale.exp().detach()),
                            }
                        )
                        for index, group in enumerate(optimizer.param_groups):
                            metrics[f"lr/{group.get('name', index)}"] = float(group["lr"])
                        if tracker is not None:
                            tracker.log(metrics)
                    loss_window, window_count = {}, 0

                if args.max_steps is not None and micro_step >= args.max_steps:
                    break

            dist.barrier()
            if rank == 0:
                save_checkpoint(
                    checkpoint_path,
                    core,
                    optimizer,
                    scaler,
                    scheduler,
                    epoch,
                    optimizer_step,
                    config,
                    classes,
                )
            dist.barrier()
            if args.max_steps is not None and micro_step >= args.max_steps:
                break

        elapsed = time.perf_counter() - started
        peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30, device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        report = {
            **core.parameter_counts(),
            "steps": micro_step,
            "optimizer_steps": optimizer_step,
            "epochs_completed": epoch + 1 if "epoch" in locals() else start_epoch,
            "world_size": world_size,
            "global_effective_batch_size": settings["effective_batch_size"],
            "elapsed_seconds": elapsed,
            "forward_seconds": elapsed,
            "backward_seconds": 0.0,
            "peak_allocated_gib": float(peak),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "checkpoint": str(checkpoint_path),
        }
        if rank == 0:
            save_json(output_dir / "train_kaggle_report.json", report)
            if tracker is not None:
                tracker.complete(report, checkpoint_path)
            print(json.dumps(report, indent=2), flush=True)
        exit_code = 0
    finally:
        if tracker is not None:
            tracker.finish(exit_code)
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
