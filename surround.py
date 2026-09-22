"""Train, evaluate, index and query the from-scratch SurroundSearch baseline."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.surround import CAMERAS, read_manifest
from src.data.surround_dataset import SurroundDataset, collate_surround
from src.models.surround import SurroundSearch, contrastive_loss, tokenize
from src.utils.config import load_config, resolve_path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def autocast(device, enabled):
    return torch.autocast("cuda", dtype=torch.float16) if enabled and device == "cuda" else nullcontext()


def make_loader(records, classes, config, args, training=False):
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(
        SurroundDataset(records, classes, config["model"], training),
        batch_size=args.batch_size or config["training"]["batch_size"], shuffle=training,
        num_workers=args.workers, pin_memory=args.device == "cuda",
        # Recreate training workers each epoch so augmentation RNG can be seeded
        # identically after resume; validation workers have no random transforms.
        persistent_workers=args.workers > 0 and not training,
        collate_fn=collate_surround, generator=generator,
    )


def unique_text_features(model, captions, device, amp):
    unique = list(dict.fromkeys(captions))
    lookup = {text: i for i, text in enumerate(unique)}
    encoded = []
    with torch.no_grad():
        for start in range(0, len(unique), 128):
            with autocast(device, amp):
                encoded.append(model.encode_text(tokenize(unique[start:start+128], model.max_text_bytes).to(device)).cpu())
    features = torch.cat(encoded)
    if not torch.isfinite(features).all():
        raise RuntimeError("Non-finite text embeddings")
    return features, torch.tensor([lookup[c] for c in captions]), unique


def retrieval_metrics(images, texts, groups):
    """Full-split retrieval, exact-caption positives; bound similarity RAM by blocks."""
    if len(images) == 0 or len(texts) == 0:
        raise ValueError("Empty retrieval gallery")
    output = {}
    for direction, queries, gallery in (("image_to_text", images, texts), ("text_to_image", texts, images)):
        hits = {k: 0 for k in (1, 5, 10)}
        for start in range(0, len(queries), 256):
            scores = queries[start:start+256].float() @ gallery.float().T
            top = scores.topk(min(10, len(gallery)), dim=1).indices
            if direction == "image_to_text":
                matched = top.eq(groups[start:start+len(scores), None])
            else:
                matched = groups[top].eq(torch.arange(start, start+len(scores))[:, None])
            for k in hits:
                hits[k] += matched[:, :k].any(dim=1).sum().item()
        output.update({f"{direction}_recall_at_{k}": count / len(queries) for k, count in hits.items()})
    output.update(images=len(images), unique_captions=len(texts))
    return output


@torch.no_grad()
def encode(model, loader, device, amp):
    model.eval()
    view_features, scene_features, view_captions, scene_captions = [], [], [], []
    view_mapping, scene_mapping = [], []
    for images, mask, _labels, captions, descriptions, indices in tqdm(loader, desc="Encoding surround samples"):
        with autocast(device, amp):
            result = model(images.to(device), mask.to(device))
        if not torch.isfinite(result["view"]).all() or not torch.isfinite(result["scene"]).all():
            raise RuntimeError("Non-finite image embeddings")
        view_features.append(result["view"][mask.to(device)].cpu())
        scene_features.append(result["scene"].cpu())
        view_captions.extend(c for c, valid in zip(captions, mask.flatten().tolist()) if valid)
        scene_captions.extend(descriptions)
        for index in indices:
            group = loader.dataset.records[index]
            scene_mapping.append(group)
            view_mapping.extend(group["views"][c] for c in CAMERAS if c in group["views"])
    return {"view": torch.cat(view_features), "scene": torch.cat(scene_features),
            "view_captions": view_captions, "scene_captions": scene_captions,
            "view_mapping": view_mapping, "scene_mapping": scene_mapping}


def evaluate(model, loader, device, amp):
    features = encode(model, loader, device, amp)
    metrics = {}
    for level in ("view", "scene"):
        texts, groups, _ = unique_text_features(model, features[f"{level}_captions"], device, amp)
        metrics[level] = retrieval_metrics(features[level], texts, groups)
    metrics["selection_score"] = sum(metrics[level][f"{direction}_recall_at_10"]
                                     for level in ("view", "scene")
                                     for direction in ("image_to_text", "text_to_image")) / 4
    metrics["relevance"] = "exact caption equality, weak-label proxy; not human semantic relevance"
    return metrics


def split_records(processed, split, limit=None, seed=42):
    names = ("train", "val", "test") if split == "all" else (split,)
    records = [record for name in names for record in read_manifest(processed / f"{name}.jsonl")]
    if limit is not None:
        if limit < 2:
            raise ValueError("--max-samples must be >= 2")
        records = random.Random(seed).sample(records, min(limit, len(records)))
    if not records:
        raise ValueError(f"No {split} records")
    return records


def assert_disjoint(partitions):
    seen_scenes, seen_samples, seen_images = set(), set(), set()
    for name, records in partitions.items():
        scenes = {r["scene_token"] for r in records}
        samples = [r["sample_token"] for r in records]
        images = [v["image_id"] for r in records for v in r["views"].values()]
        if len(samples) != len(set(samples)) or len(images) != len(set(images)):
            raise ValueError(f"Duplicate sample/image in {name}")
        if seen_scenes & scenes or seen_samples & set(samples) or seen_images & set(images):
            raise ValueError(f"Partition leakage involving {name}")
        seen_scenes.update(scenes)
        seen_samples.update(samples)
        seen_images.update(images)


def train(config, args, processed, classes, output):
    partitions = {split: split_records(processed, split) for split in ("train", "val", "test")}
    assert_disjoint(partitions)
    records = split_records(processed, "train", args.max_samples, config["seed"])
    validation = split_records(processed, "val", args.max_samples, config["seed"])
    if len(records) < 2:
        raise ValueError("Training needs at least two samples")
    manifest_hash = {s: file_hash(processed / f"{s}.jsonl") for s in ("train", "val", "test")}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not args.resume:
        raise ValueError(f"Run exists at {output}; use --resume or a new --output-dir")
    settings = config["training"]
    amp = settings["precision"] == "amp" and args.device == "cuda"
    model = SurroundSearch(config["model"], len(classes)).to(args.device)
    loader = make_loader(records, classes, config, args, True)
    val_loader = make_loader(validation, classes, config, args)
    epochs = args.epochs or settings["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    total_steps = len(loader) * epochs
    warmup = min(settings["warmup_epochs"] * len(loader), max(1, total_steps // 4))
    def rate(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1, (step-warmup) / max(1, total_steps-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    positives = np.zeros(len(classes))
    class_ids = {name: i for i, name in enumerate(classes)}
    total_views = sum(len(r["views"]) for r in records)
    for record in records:
        for view in record["views"].values():
            for name in view["objects"]:
                positives[class_ids[name]] += 1
    pos_weight = torch.tensor(np.clip((total_views - positives) / np.maximum(positives, 1), 1, 10),
                              device=args.device, dtype=torch.float32)
    start, history, stale, best, significant_best = 0, [], 0, -1.0, -1.0
    signature = {"model": config["model"], "training": settings, "classes": classes,
                 "manifest_hash": manifest_hash, "seed": config["seed"], "epochs": epochs,
                 "batch_size": loader.batch_size, "max_samples": args.max_samples}
    def payload(epoch):
        return {"architecture": "SurroundSearch-v1", "model_config": config["model"], "classes": classes,
                "model": model.state_dict(), "epoch": epoch, "signature": signature,
                "best_score": best, "significant_best": significant_best, "stale": stale, "history": history}
    if args.resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature:
            raise ValueError("Resume requires matching dataset, model and training settings")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, history = checkpoint["epoch"], checkpoint["history"]
        best, significant_best, stale = checkpoint["best_score"], checkpoint["significant_best"], checkpoint["stale"]
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    else:
        baseline = evaluate(model, val_loader, args.device, amp)
        best = significant_best = baseline["selection_score"]
        history.append({"epoch": 0, "validation": baseline})
        save_checkpoint(output / "best.pt", payload(0))
        print(f"untrained_validation_score={best:.6f}", flush=True)
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()),
                      "training_samples": len(records), "training_images": total_views,
                      "validation_samples": len(validation), "batch_samples": loader.batch_size,
                      "device": args.device, "output": str(output)}, indent=2), flush=True)
    for epoch in range(start, epochs):
        if stale >= settings["patience"]:
            break
        loader.generator.manual_seed(config["seed"] + epoch)
        model.train()
        sums, count = {}, 0
        for images, mask, labels, captions, descriptions, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, mask, labels = images.to(args.device), mask.to(args.device), labels.to(args.device)
            view_captions = [c for c, valid in zip(captions, mask.flatten().tolist()) if valid]
            # Deduplicate text encoding within the batch; preserve all positive pairs.
            all_captions = view_captions + descriptions
            unique = list(dict.fromkeys(all_captions))
            ids = {text: i for i, text in enumerate(unique)}
            optimizer.zero_grad(set_to_none=True)
            with autocast(args.device, amp):
                result = model(images, mask, settings["camera_dropout"])
                text = model.encode_text(tokenize(unique, model.max_text_bytes).to(args.device))
                text = text[[ids[c] for c in all_captions]]
                scale = model.logit_scale.exp().clamp(max=100)
                view_loss = contrastive_loss(result["view"][mask], text[:len(view_captions)], view_captions, scale)
                scene_loss = contrastive_loss(result["scene"], text[len(view_captions):], descriptions, scale)
                object_loss = F.binary_cross_entropy_with_logits(result["object_logits"][mask].float(), labels[mask], pos_weight=pos_weight)
                loss = (settings["view_loss_weight"] * view_loss + settings["scene_loss_weight"] * scene_loss
                        + settings["object_loss_weight"] * object_loss)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; refusing to save invalid weights")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))
            for key, value in (("loss", loss), ("view_loss", view_loss), ("scene_loss", scene_loss), ("object_loss", object_loss)):
                sums[key] = sums.get(key, 0) + float(value.detach()) * len(images)
            count += len(images)
        metrics = evaluate(model, val_loader, args.device, amp)
        score = metrics["selection_score"]
        improved = score > best
        best = max(best, score)
        if score > significant_best + settings["min_delta"]:
            significant_best, stale = score, 0
        else:
            stale += 1
        history.append({"epoch": epoch+1, **{k: v / count for k, v in sums.items()}, "validation": metrics})
        state = payload(epoch+1)
        if improved:
            save_checkpoint(output / "best.pt", state)
        state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else [])
        save_checkpoint(output / "last.pt", state)
        save_json(output / "history.json", history)
        print(f"epoch={epoch+1} train_loss={sums['loss']/count:.5f} val_score={score:.5f} best={best:.5f} patience={stale}/{settings['patience']}", flush=True)
    save_json(output / "summary.json", {"best_validation_score": best, "epochs_completed": len(history)-1,
                                       "training_images": total_views, "validation_samples": len(validation),
                                       "best_checkpoint": str(output / "best.pt"), "smoke_test": args.max_samples is not None})


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "SurroundSearch-v1":
        raise ValueError("Expected SurroundSearch checkpoint, not OpenCLIP weights")
    model = SurroundSearch(checkpoint["model_config"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "eval", "index", "query"))
    parser.add_argument("--config", default="configs/surround.yaml")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-samples", type=int, help="Smoke tests only; limits samples per split")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=("train", "val", "test", "all"))
    parser.add_argument("--query", default="")
    parser.add_argument("--level", choices=("view", "scene"), default="view")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["training"]
    if config["model"].get("initialization") != "scratch":
        parser.error("SurroundSearch-v1 currently supports initialization: scratch")
    if not 0 <= settings["camera_dropout"] <= 1 or settings["patience"] < 1 or settings["min_delta"] < 0:
        parser.error("Invalid camera dropout or early-stopping settings")
    if settings["learning_rate"] <= 0 or settings["weight_decay"] < 0:
        parser.error("Learning rate must be positive and weight decay nonnegative")
    args.workers = config["training"]["num_workers"] if args.workers is None else args.workers
    if args.workers < 0 or args.top_k < 1 or (args.batch_size is not None and args.batch_size < 2) or (args.epochs is not None and args.epochs < 1):
        parser.error("workers >= 0, top-k >= 1, batch-size >= 2 and epochs >= 1 required")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    processed = resolve_path(config, config["dataset"]["processed_dir"])
    output = args.output_dir.resolve() if args.output_dir else resolve_path(config, config["training"]["output_dir"])
    if args.command == "train":
        if args.max_samples and not args.output_dir:
            parser.error("Smoke training requires a separate --output-dir")
        stats = json.loads((processed / "stats.json").read_text(encoding="utf-8"))
        train(config, args, processed, stats["classes"], output)
        return
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else output / "best.pt"
    model, checkpoint = load_model(checkpoint_path, args.device)
    config["model"] = checkpoint["model_config"]
    amp = args.device == "cuda" and config["training"]["precision"] == "amp"
    if args.command == "query":
        if not args.query.strip():
            parser.error("--query is required")
        directory = output / f"index_{args.split or 'all'}"
        info = json.loads((directory / "info.json").read_text(encoding="utf-8"))
        if info["checkpoint_sha256"] != file_hash(checkpoint_path):
            raise ValueError("Index/checkpoint mismatch: rebuild the index")
        embeddings = np.load(directory / f"{args.level}.npy", mmap_mode="r")
        mapping = json.loads((directory / f"{args.level}.json").read_text(encoding="utf-8"))
        with torch.no_grad(), autocast(args.device, amp):
            text = model.encode_text(tokenize([args.query], model.max_text_bytes).to(args.device)).cpu().numpy()[0]
        scores = embeddings @ text
        matches = np.argsort(-scores)[:args.top_k]
        print(json.dumps([dict(mapping[i], score=float(scores[i])) for i in matches], ensure_ascii=False, indent=2))
        return
    split = args.split or ("test" if args.command == "eval" else "all")
    for name in (("train", "val", "test") if split == "all" else (split,)):
        if file_hash(processed / f"{name}.jsonl") != checkpoint["signature"]["manifest_hash"][name]:
            raise ValueError(f"The {name} manifest changed since training; use matching data/checkpoint")
    records = split_records(processed, split, args.max_samples, config["seed"])
    loader = make_loader(records, checkpoint["classes"], config, args)
    if args.command == "eval":
        result = evaluate(model, loader, args.device, amp)
        result.update(split=split, checkpoint=str(checkpoint_path), checkpoint_epoch=checkpoint["epoch"],
                      limited=args.max_samples is not None, manifest_hash=file_hash(processed / f"{split}.jsonl") if split != "all" else None)
        save_json(output / f"evaluation_{split}{'_limited' if args.max_samples else ''}.json", result)
        print(json.dumps(result, indent=2))
    else:
        features = encode(model, loader, args.device, amp)
        directory = output / f"index_{split}"
        directory.mkdir(parents=True, exist_ok=True)
        for level in ("view", "scene"):
            np.save(directory / f"{level}.npy", features[level].numpy().astype("float32"))
            save_json(directory / f"{level}.json", features[f"{level}_mapping"])
        save_json(directory / "info.json", {"checkpoint_sha256": file_hash(checkpoint_path), "split": split,
                                            "images": len(features["view"]), "samples": len(features["scene"]),
                                            "limited": args.max_samples is not None})
        print(f"Wrote normalized view and scene indexes: {directory}")


if __name__ == "__main__":
    main()
