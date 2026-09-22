#!/usr/bin/env python3
"""Fine-tune OpenCLIP on a configured BDD100K or processed nuScenes dataset.

Supervision comes from grounded scene metadata and object categories. Optional
scene-grouped validation, validation-selected checkpoints, and early stopping
prevent adjacent frames from leaking into model selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bdd100k import BDDRecord, find_dataset_root, iter_split_records
from src.data.nuscenes import NuScenesRecord, iter_processed_records
from src.utils.common import clean_text, is_meaningful
from src.utils.config import load_config, resolve_path


SceneRecord = BDDRecord | NuScenesRecord


def caption_from_record(record: SceneRecord) -> str:
    """Turn normalized scene metadata into a grounded training caption."""
    description = clean_text(getattr(record, "scene_description", ""))
    location = clean_text(getattr(record, "location", ""))
    if is_meaningful(description):
        caption = description.rstrip(" .")
        if is_meaningful(location):
            caption += f" in {location.replace('-', ' ')}"
        objects = [clean_text(value) for value in record.objects]
        objects = [value for value in objects if is_meaningful(value)]
        if objects:
            caption += "; visible objects: " + ", ".join(objects)
        return caption + "."

    attributes: list[str] = []
    weather = clean_text(record.weather)
    scene = clean_text(record.scene)
    timeofday = clean_text(record.timeofday)
    if is_meaningful(weather):
        attributes.append(f"in {weather} weather")
    if is_meaningful(timeofday):
        attributes.append(f"during {timeofday}")

    subject = f"a {scene} driving scene" if is_meaningful(scene) else "a driving scene"
    caption = " ".join([subject, *attributes])
    objects = [clean_text(value) for value in record.objects]
    objects = [value for value in objects if is_meaningful(value)]
    if objects:
        caption += " with " + ", ".join(objects)
    return caption + "."


def load_records(
    dataset_root: Path,
    splits: Sequence[str],
    dataset_format: str = "bdd100k",
) -> tuple[list[SceneRecord], list[dict[str, str]]]:
    records: list[SceneRecord] = []
    errors: list[dict[str, str]] = []
    for split in splits:
        before = len(records)
        if dataset_format == "nuscenes":
            records.extend(iter_processed_records(dataset_root, split, errors))
        else:
            records.extend(iter_split_records(dataset_root, split, errors))
        print(f"split={split} records={len(records) - before:,}")
    # Image ids can theoretically repeat across native splits. The full path is
    # the identity here, so no valid image is accidentally removed.
    records.sort(key=lambda item: (item.source_split.lower(), item.image_path.lower()))
    return records, errors


def split_train_validation(
    records: Sequence[SceneRecord], validation_fraction: float, seed: int
) -> tuple[list[SceneRecord], list[SceneRecord]]:
    """Split whole scenes so adjacent frames never cross train/validation."""
    records = list(records)
    if validation_fraction <= 0 or len(records) < 2:
        return records, []
    groups: dict[str, list[SceneRecord]] = defaultdict(list)
    for record in records:
        scene_token = clean_text(getattr(record, "scene_token", ""))
        key = scene_token or f"image:{record.image_id}"
        groups[key].append(record)
    for group in groups.values():
        group.sort(key=lambda record: (record.image_path.lower(), record.image_id.lower()))
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, round(len(records) * validation_fraction))
    validation_keys: set[str] = set()
    validation_count = 0
    for key in keys:
        if len(records) - validation_count - len(groups[key]) < 1:
            continue
        validation_keys.add(key)
        validation_count += len(groups[key])
        if validation_count >= target:
            break
    train_records = [record for key in keys if key not in validation_keys for record in groups[key]]
    validation_records = [record for key in keys if key in validation_keys for record in groups[key]]
    return train_records, validation_records


def _import_training_dependencies():
    try:
        import open_clip
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc
    return open_clip, torch, DataLoader, tqdm


class BDD100KContrastiveDataset:
    """Pickle-safe dataset class for Windows DataLoader workers."""

    def __init__(self, records: Sequence[SceneRecord], transform: Any) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        from PIL import Image

        record = self.records[index]
        try:
            with Image.open(record.image_path) as image:
                tensor = self.transform(image.convert("RGB"))
            return tensor, caption_from_record(record), record.image_path, ""
        except Exception as exc:  # Keep a corrupt image from aborting a long run.
            error = f"{type(exc).__name__}: {exc}"
            return None, "", record.image_path, error


def collate_valid(samples: Sequence[tuple[Any, str, str, str]]):
    """Drop unreadable images while reporting their paths to the main process."""
    import torch

    valid = [sample for sample in samples if sample[0] is not None]
    failures = [
        {"image_path": path, "reason": error}
        for image, _caption, path, error in samples
        if image is None
    ]
    if not valid:
        return None, [], failures
    images = torch.stack([sample[0] for sample in valid])
    captions = [sample[1] for sample in valid]
    return images, captions, failures


def multi_positive_clip_loss(torch: Any, image_features: Any, text_features: Any, logit_scale: Any, captions: Sequence[str]):
    """Symmetric CLIP loss that does not treat duplicate captions as negatives."""
    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    logits = logit_scale * image_features @ text_features.T

    group_by_caption: dict[str, int] = {}
    group_ids = [group_by_caption.setdefault(text, len(group_by_caption)) for text in captions]
    groups = torch.tensor(group_ids, device=logits.device)
    positives = groups[:, None].eq(groups[None, :])
    negative_infinity = torch.finfo(logits.dtype).min

    image_numerator = torch.logsumexp(logits.masked_fill(~positives, negative_infinity), dim=1)
    image_loss = (torch.logsumexp(logits, dim=1) - image_numerator).mean()
    text_logits = logits.T
    text_positives = positives.T
    text_numerator = torch.logsumexp(
        text_logits.masked_fill(~text_positives, negative_infinity), dim=1
    )
    text_loss = (torch.logsumexp(text_logits, dim=1) - text_numerator).mean()
    return (image_loss + text_loss) / 2


def validation_loss(
    torch: Any, model: Any, tokenizer: Any, loader: Any, device: str, use_amp: bool
) -> tuple[float, int, list[dict[str, str]]]:
    model.eval()
    loss_sum = 0.0
    example_count = 0
    failures: list[dict[str, str]] = []
    with torch.no_grad():
        for images, captions, batch_failures in loader:
            failures.extend(batch_failures)
            if images is None:
                continue
            images = images.to(device, non_blocking=True)
            tokens = tokenizer(captions).to(device, non_blocking=True)
            autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with autocast:
                loss = multi_positive_clip_loss(
                    torch,
                    model.encode_image(images),
                    model.encode_text(tokens),
                    model.logit_scale.exp(),
                    captions,
                )
            loss_sum += float(loss.detach()) * len(captions)
            example_count += len(captions)
    return loss_sum / example_count if example_count else float("inf"), example_count, failures


def optimizer_for(torch: Any, model: Any, learning_rate: float, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or "logit_scale" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
    )


def scheduler_for(torch: Any, optimizer: Any, warmup_steps: int, total_steps: int):
    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def atomic_torch_save(torch: Any, value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    torch: Any,
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_loss: float,
    epoch_loss: float,
    args: argparse.Namespace,
    model_name: str,
    pretrained: str,
    record_count: int,
    validation_loss_value: float,
    epochs_without_improvement: int,
) -> None:
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "epoch_loss": epoch_loss,
        "validation_loss": validation_loss_value,
        "epochs_without_improvement": epochs_without_improvement,
        "model_name": model_name,
        "pretrained": pretrained,
        "record_count": record_count,
        "splits": list(args.splits),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    atomic_torch_save(torch, state, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-format", choices=("bdd100k", "nuscenes"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test", "dev", "all"),
        default=None, help="Dataset splits to train on (nuScenes default: dev).",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--validation-batch-size", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--min-delta", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--precision", choices=("amp", "fp32"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", nargs="?", const="auto", help="Checkpoint path, or auto for output-dir/last.pt.")
    parser.add_argument("--max-samples", type=int, help="Optional smoke-test limit after loading all metadata.")
    return parser.parse_args()


def configured_args(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    training = config.get("training", {})
    defaults = {
        "splits": training.get("splits", ["train", "val", "test"]),
        "epochs": training.get("epochs", 5),
        "batch_size": training.get("batch_size", 32),
        "learning_rate": training.get("learning_rate", 5e-6),
        "weight_decay": training.get("weight_decay", 0.2),
        "warmup_steps": training.get("warmup_steps", 500),
        "gradient_accumulation": training.get("gradient_accumulation", 1),
        "max_grad_norm": training.get("max_grad_norm", 1.0),
        "validation_fraction": training.get("validation_fraction", 0.0),
        "validation_batch_size": training.get("validation_batch_size", 64),
        "early_stopping_patience": training.get("early_stopping_patience", 0),
        "min_delta": training.get("min_delta", 0.0),
        "num_workers": training.get("num_workers", 4),
        "precision": training.get("precision", "amp"),
        "seed": config.get("seed", 42),
        "dataset_format": config.get("dataset", {}).get("format", "bdd100k"),
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.output_dir is None:
        configured = training.get("output_dir", "artifacts/checkpoints")
        args.output_dir = resolve_path(config, configured)
    else:
        args.output_dir = args.output_dir.resolve()
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "validation_batch_size": args.validation_batch_size,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise SystemExit("These arguments must be positive: " + ", ".join(invalid))
    if args.num_workers < 0 or args.warmup_steps < 0 or args.weight_decay < 0 or args.max_grad_norm < 0:
        raise SystemExit("num-workers, warmup-steps, weight-decay, and max-grad-norm cannot be negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if not 0 <= args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be in [0, 1)")
    if args.early_stopping_patience < 0 or args.min_delta < 0:
        raise SystemExit("--early-stopping-patience and --min-delta cannot be negative")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args = configured_args(args, config)
    validate_args(args)
    open_clip, torch, DataLoader, tqdm = _import_training_dependencies()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    configured_root = args.dataset_root or resolve_path(config, config["paths"]["dataset_root"])
    dataset_root = (
        Path(configured_root).resolve()
        if args.dataset_format == "nuscenes"
        else find_dataset_root(configured_root)
    )
    records, parse_errors = load_records(dataset_root, args.splits, args.dataset_format)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise SystemExit(f"No annotated images found in {dataset_root} for splits={args.splits}")
    train_records, validation_records = split_train_validation(
        records, args.validation_fraction, args.seed
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False")
    use_amp = args.precision == "amp" and device == "cuda"
    if args.precision == "amp" and device != "cuda":
        print("AMP requested on CPU; using fp32 instead")

    model_cfg = config["model"]
    model_name = str(model_cfg["name"])
    pretrained = str(model_cfg["pretrained"])
    model, preprocess_train, preprocess_eval = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.train()

    dataset = BDD100KContrastiveDataset(train_records, preprocess_train)
    generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_valid,
        generator=generator,
    )
    validation_loader = None
    if validation_records:
        validation_loader = DataLoader(
            BDD100KContrastiveDataset(validation_records, preprocess_eval),
            batch_size=args.validation_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device == "cuda",
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_valid,
        )
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    total_steps = steps_per_epoch * args.epochs
    optimizer = optimizer_for(torch, model, args.learning_rate, args.weight_decay)
    scheduler = scheduler_for(torch, optimizer, args.warmup_steps, total_steps)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # Compatibility with older supported torch builds.
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if parse_errors:
        (args.output_dir / "metadata_parse_errors.json").write_text(
            json.dumps(parse_errors, indent=2) + "\n", encoding="utf-8"
        )

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    epochs_without_improvement = 0
    if args.resume:
        resume_path = args.output_dir / "last.pt" if args.resume == "auto" else Path(args.resume).resolve()
        if not resume_path.is_file():
            raise SystemExit(f"Resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_loss = float(checkpoint.get("best_loss", float("inf")))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(f"resumed={resume_path} next_epoch={start_epoch + 1} global_step={global_step}")

    print(f"dataset_root={dataset_root}")
    print(f"dataset_format={args.dataset_format}")
    print(f"splits={','.join(args.splits)}")
    print(f"training_images={len(train_records):,}")
    print(f"validation_images={len(validation_records):,}")
    print(f"metadata_parse_errors={len(parse_errors):,}")
    print(f"model={model_name} pretrained={pretrained}")
    print(f"device={device} precision={'amp' if use_amp else 'fp32'}")
    print(f"epochs={args.epochs} batch_size={args.batch_size} steps={total_steps:,}")
    print(f"output_dir={args.output_dir}")

    run_started = time.perf_counter()
    skipped_images: dict[str, str] = {}
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs):
        generator.manual_seed(args.seed + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        example_count = 0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")

        for batch_index, (images, captions, failures) in enumerate(progress):
            for failure in failures:
                skipped_images[failure["image_path"]] = failure["reason"]
            if images is None:
                continue
            images = images.to(device, non_blocking=True)
            tokens = tokenizer(captions).to(device, non_blocking=True)
            autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            try:
                with autocast:
                    image_features = model.encode_image(images)
                    text_features = model.encode_text(tokens)
                    logit_scale = model.logit_scale.exp()
                    loss = multi_positive_clip_loss(
                        torch, image_features, text_features, logit_scale, captions
                    )
            except torch.cuda.OutOfMemoryError as exc:
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    f"CUDA out of memory at batch_size={args.batch_size}. "
                    "Retry with --batch-size 16 (or 8)."
                ) from exc

            group_start = (batch_index // args.gradient_accumulation) * args.gradient_accumulation
            accumulation_divisor = min(
                args.gradient_accumulation, len(loader) - group_start
            )
            scaler.scale(loss / accumulation_divisor).backward()
            should_step = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                if args.max_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    model.logit_scale.clamp_(0, math.log(100.0))
                scheduler.step()
                global_step += 1

            batch_examples = len(captions)
            loss_sum += float(loss.detach()) * batch_examples
            example_count += batch_examples
            progress.set_postfix(
                loss=f"{loss_sum / example_count:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                skipped=len(skipped_images),
            )

        if not example_count:
            raise RuntimeError("Every image failed to load; no optimizer update was performed")
        epoch_loss = loss_sum / example_count
        val_loss = epoch_loss
        if validation_loader is not None:
            val_loss, val_examples, val_failures = validation_loss(
                torch, model, tokenizer, validation_loader, device, use_amp
            )
            for failure in val_failures:
                skipped_images[failure["image_path"]] = failure["reason"]
            if not val_examples:
                raise RuntimeError("Every validation image failed to load")
        improved = val_loss < best_loss - args.min_delta
        if improved:
            best_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss, "validation_loss": val_loss})
        save_checkpoint(
            torch, args.output_dir / "last.pt", model, optimizer, scheduler, scaler,
            epoch, global_step, best_loss, epoch_loss, args, model_name, pretrained, len(train_records),
            val_loss, epochs_without_improvement,
        )
        # Weights-only files can be passed directly as OpenCLIP's `pretrained`
        # path when rebuilding the retrieval index.
        atomic_torch_save(torch, model.state_dict(), args.output_dir / "last_model.pt")
        if improved:
            atomic_torch_save(torch, model.state_dict(), args.output_dir / "best_model.pt")
        print(
            f"epoch={epoch + 1} train_loss={epoch_loss:.6f} "
            f"validation_loss={val_loss:.6f} best_validation_loss={best_loss:.6f}"
        )
        if validation_loader is not None and args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
            print(f"early_stopping=1 patience={args.early_stopping_patience}")
            break

    if skipped_images:
        failures = [{"image_path": path, "reason": reason} for path, reason in skipped_images.items()]
        (args.output_dir / "skipped_images.json").write_text(
            json.dumps(failures, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "dataset_root": str(dataset_root),
        "splits": list(args.splits),
        "training_images": len(train_records),
        "validation_images": len(validation_records),
        "skipped_images": len(skipped_images),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "global_steps": global_step,
        "best_validation_loss": best_loss,
        "history": history,
        "elapsed_seconds": time.perf_counter() - run_started,
        "best_model": str(args.output_dir / "best_model.pt"),
        "last_model": str(args.output_dir / "last_model.pt"),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
