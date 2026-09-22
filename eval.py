#!/usr/bin/env python3
"""Evaluate a fine-tuned OpenCLIP checkpoint on annotated BDD100K images.

The evaluator reports the same multi-positive contrastive loss used for
training and bidirectional image/text retrieval metrics. Captions that are
identical are treated as the same class, so another image with the same BDD100K
metadata caption is not incorrectly counted as a negative.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bdd100k import BDDRecord, find_dataset_root
from src.utils.config import load_config, resolve_path
from train import (
    BDD100KContrastiveDataset,
    caption_from_record,
    collate_valid,
    load_records,
    multi_positive_clip_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Weights-only model or resumable checkpoint (default: <training.output_dir>/best_model.pt).",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-format", choices=("bdd100k", "nuscenes"))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test", "dev", "all"),
        default=None,
        help="Splits to evaluate (default: test for nuScenes, val for BDD100K).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--similarity-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--precision", choices=("amp", "fp32"), default="amp")
    parser.add_argument("--recall-k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--max-samples", type=int, help="Evaluate a deterministic random subset.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, help="Output JSON path (default: reports/model_eval.json).")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch-size": args.batch_size,
        "text-batch-size": args.text_batch_size,
        "similarity-batch-size": args.similarity_batch_size,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise SystemExit("These arguments must be positive: " + ", ".join(invalid))
    if args.num_workers < 0:
        raise SystemExit("--num-workers cannot be negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if not args.recall_k or any(value <= 0 for value in args.recall_k):
        raise SystemExit("--recall-k values must be positive")
    args.recall_k = sorted(set(args.recall_k))


def import_dependencies():
    try:
        import open_clip
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            "Evaluation dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc
    return open_clip, torch, DataLoader, tqdm


def load_checkpoint(torch: Any, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a model state dict and non-tensor checkpoint metadata."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"Unsupported checkpoint format: {path}")

    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
        metadata = {
            key: payload[key]
            for key in (
                "epoch",
                "global_step",
                "best_loss",
                "epoch_loss",
                "model_name",
                "pretrained",
                "record_count",
                "splits",
            )
            if key in payload
        }
    elif payload and all(hasattr(value, "shape") for value in payload.values()):
        state = payload
        metadata = {}
    else:
        raise SystemExit(
            f"Checkpoint does not contain model_state_dict or a weights-only state dict: {path}"
        )
    return state, metadata


def select_records(
    records: Sequence[BDDRecord], max_samples: int | None, seed: int
) -> list[BDDRecord]:
    records = list(records)
    if max_samples is None or max_samples >= len(records):
        return records
    selected = random.Random(seed).sample(range(len(records)), max_samples)
    return [records[index] for index in sorted(selected)]


def encode_texts(
    torch: Any,
    model: Any,
    tokenizer: Any,
    captions: Sequence[str],
    device: str,
    batch_size: int,
    use_amp: bool,
) -> Any:
    batches = []
    for start in range(0, len(captions), batch_size):
        tokens = tokenizer(captions[start : start + batch_size]).to(device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        batches.append(features.float().cpu())
    return torch.cat(batches)


def retrieval_metrics(
    torch: Any,
    image_features: Any,
    text_features: Any,
    image_group_ids: Any,
    recall_k: Sequence[int],
    device: str,
    batch_size: int,
) -> dict[str, float]:
    """Compute exact retrieval metrics without materializing the full score matrix."""
    max_k = min(max(recall_k), max(len(image_features), len(text_features)))
    text_device = text_features.to(device)
    image_group_ids = image_group_ids.long()

    i2t_hits = {k: 0 for k in recall_k}
    i2t_rr_sum = 0.0
    for start in range(0, len(image_features), batch_size):
        stop = min(start + batch_size, len(image_features))
        queries = image_features[start:stop].to(device)
        scores = queries @ text_device.T
        targets = image_group_ids[start:stop].to(device)
        top_indices = scores.topk(min(max_k, scores.shape[1]), dim=1).indices
        for k in recall_k:
            effective_k = min(k, scores.shape[1])
            i2t_hits[k] += int(
                top_indices[:, :effective_k].eq(targets[:, None]).any(dim=1).sum().item()
            )
        target_scores = scores.gather(1, targets[:, None])
        ranks = scores.gt(target_scores).sum(dim=1) + 1
        i2t_rr_sum += float((1.0 / ranks.float()).sum().item())

    image_device = image_features.to(device)
    groups_device = image_group_ids.to(device)
    t2i_hits = {k: 0 for k in recall_k}
    t2i_rr_sum = 0.0
    for start in range(0, len(text_features), batch_size):
        stop = min(start + batch_size, len(text_features))
        queries = text_features[start:stop].to(device)
        scores = queries @ image_device.T
        query_groups = torch.arange(start, stop, device=device)
        top_indices = scores.topk(min(max_k, scores.shape[1]), dim=1).indices
        top_groups = groups_device[top_indices]
        for k in recall_k:
            effective_k = min(k, scores.shape[1])
            t2i_hits[k] += int(
                top_groups[:, :effective_k]
                .eq(query_groups[:, None])
                .any(dim=1)
                .sum()
                .item()
            )

        # The best-scoring matching image determines the first relevant rank.
        for row, group_id in enumerate(range(start, stop)):
            relevant = groups_device.eq(group_id)
            best_relevant = scores[row, relevant].max()
            rank = int(scores[row].gt(best_relevant).sum().item()) + 1
            t2i_rr_sum += 1.0 / rank

    image_count = len(image_features)
    text_count = len(text_features)
    result = {
        f"image_to_text_recall_at_{k}": i2t_hits[k] / image_count for k in recall_k
    }
    result["image_to_text_mrr"] = i2t_rr_sum / image_count
    result.update(
        {f"text_to_image_recall_at_{k}": t2i_hits[k] / text_count for k in recall_k}
    )
    result["text_to_image_mrr"] = t2i_rr_sum / text_count
    return result


def evaluate(
    torch: Any,
    DataLoader: Any,
    tqdm: Any,
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    records: Sequence[BDDRecord],
    device: str,
    batch_size: int,
    text_batch_size: int,
    similarity_batch_size: int,
    num_workers: int,
    use_amp: bool,
    recall_k: Sequence[int],
) -> tuple[dict[str, float | int], list[dict[str, str]]]:
    dataset = BDD100KContrastiveDataset(records, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_valid,
    )
    model.eval()
    image_batches = []
    valid_captions: list[str] = []
    failures: list[dict[str, str]] = []
    loss_sum = 0.0
    example_count = 0

    for images, captions, batch_failures in tqdm(loader, desc="Evaluating", unit="batch"):
        failures.extend(batch_failures)
        if images is None:
            continue
        images = images.to(device, non_blocking=True)
        tokens = tokenizer(captions).to(device, non_blocking=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            image_features = model.encode_image(images)
            text_features = model.encode_text(tokens)
            loss = multi_positive_clip_loss(
                torch, image_features, text_features, model.logit_scale.exp(), captions
            )
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)

        count = len(captions)
        loss_sum += float(loss) * count
        example_count += count
        image_batches.append(image_features.float().cpu())
        valid_captions.extend(captions)

    if not image_batches:
        raise RuntimeError("No valid images were available for evaluation")

    caption_to_group: dict[str, int] = {}
    unique_captions: list[str] = []
    group_ids: list[int] = []
    for caption in valid_captions:
        if caption not in caption_to_group:
            caption_to_group[caption] = len(unique_captions)
            unique_captions.append(caption)
        group_ids.append(caption_to_group[caption])

    image_features = torch.cat(image_batches)
    text_features = encode_texts(
        torch,
        model,
        tokenizer,
        unique_captions,
        device,
        text_batch_size,
        use_amp,
    )
    metrics: dict[str, float | int] = {
        "contrastive_loss": loss_sum / example_count,
        "evaluated_images": example_count,
        "unique_captions": len(unique_captions),
        "skipped_images": len(failures),
    }
    metrics.update(
        retrieval_metrics(
            torch,
            image_features,
            text_features,
            torch.tensor(group_ids),
            recall_k,
            device,
            similarity_batch_size,
        )
    )
    return metrics, failures


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    training = config.get("training", {})
    output_dir = resolve_path(config, training.get("output_dir", "artifacts/checkpoints"))
    checkpoint_path = (
        args.checkpoint.resolve() if args.checkpoint else output_dir / "best_model.pt"
    )
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint_path}")

    dataset_format = args.dataset_format or config.get("dataset", {}).get("format", "bdd100k")
    if args.splits is None:
        args.splits = ["test" if dataset_format == "nuscenes" else "val"]
    configured_root = args.dataset_root or resolve_path(config, config["paths"]["dataset_root"])
    dataset_root = (
        Path(configured_root).resolve()
        if dataset_format == "nuscenes"
        else find_dataset_root(configured_root)
    )
    records, parse_errors = load_records(dataset_root, args.splits, dataset_format)
    records = select_records(records, args.max_samples, seed)
    if not records:
        raise SystemExit(f"No annotated images found for splits={args.splits}")

    open_clip, torch, DataLoader, tqdm = import_dependencies()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False")
    use_amp = args.precision == "amp" and device == "cuda"

    state_dict, checkpoint_metadata = load_checkpoint(torch, checkpoint_path)
    model_name = str(checkpoint_metadata.get("model_name", config["model"]["name"]))
    configured_training_splits = checkpoint_metadata.get(
        "splits", training.get("splits", [])
    )
    training_split_overlap = sorted(set(args.splits) & set(configured_training_splits))
    # The checkpoint contains every parameter, so no pretrained download is needed.
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=None, device="cpu"
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise SystemExit(
            f"Checkpoint architecture does not match model.name={model_name!r}: {exc}"
        ) from exc
    del state_dict
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)

    print(f"checkpoint={checkpoint_path}")
    print(f"model={model_name}")
    print(f"dataset_root={dataset_root}")
    print(f"dataset_format={dataset_format}")
    print(f"splits={','.join(args.splits)}")
    print(f"requested_images={len(records):,}")
    print(f"metadata_parse_errors={len(parse_errors):,}")
    print(f"device={device} precision={'amp' if use_amp else 'fp32'}")
    if training_split_overlap:
        print(
            "warning=evaluation overlaps configured training splits: "
            + ",".join(training_split_overlap)
            + "; metrics measure fit, not held-out generalization"
        )
    started = time.perf_counter()
    metrics, image_errors = evaluate(
        torch,
        DataLoader,
        tqdm,
        model,
        preprocess,
        tokenizer,
        records,
        device,
        args.batch_size,
        args.text_batch_size,
        args.similarity_batch_size,
        args.num_workers,
        use_amp,
        args.recall_k,
    )
    elapsed = time.perf_counter() - started

    output_path = (
        args.output.resolve()
        if args.output
        else resolve_path(config, config.get("paths", {}).get("reports_dir", "reports"))
        / "model_eval.json"
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
        "model_name": model_name,
        "dataset_root": str(dataset_root),
        "splits": list(args.splits),
        "configured_training_splits": list(configured_training_splits),
        "training_split_overlap": training_split_overlap,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "metadata_parse_errors": parse_errors,
        "image_errors": image_errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"report={output_path}")


if __name__ == "__main__":
    main()
