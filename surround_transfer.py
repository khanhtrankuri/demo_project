"""Frozen OpenCLIP + residual multi-camera adapters; v1 splits and metrics unchanged."""
import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.surround import CAMERAS, read_manifest
from src.models.surround_transfer import SurroundTransfer, balanced_contrastive_loss
from src.utils.config import load_config, resolve_path
from surround import assert_disjoint, autocast, file_hash, retrieval_metrics, save_checkpoint, save_json


def backbone(config, device):
    import open_clip
    from huggingface_hub import hf_hub_download
    settings = config["model"]
    # Use the existing local base weights, never the previous fine-tuned checkpoint.
    weights = hf_hub_download(settings["pretrained_repo"], settings["pretrained_file"], local_files_only=True)
    model, _, preprocess = open_clip.create_model_and_transforms(settings["backbone"], pretrained=weights, device=device)
    model.eval().requires_grad_(False)
    return model, preprocess, open_clip.get_tokenizer(settings["backbone"]), file_hash(weights)


class FullFieldImages:
    def __init__(self, views, preprocess):
        self.views, self.preprocess = views, preprocess

    def __len__(self):
        return len(self.views)

    def __getitem__(self, index):
        with Image.open(self.views[index]["image_path"]) as source:
            image = source.convert("RGB")
            # Padding prevents CLIP's center crop from dropping lateral objects.
            side = max(image.size)
            image = ImageOps.pad(image, (side, side), color=(123, 117, 104))
            return self.preprocess(image)


def cache_path(output, split):
    return output / "features" / f"{split}.pt"


@torch.no_grad()
def cache(config, args, output, processed):
    model, preprocess, tokenizer, weights_hash = backbone(config, args.device)
    classes = json.loads((processed / "stats.json").read_text(encoding="utf-8"))["classes"]
    splits = [args.split] if args.split else ["train", "val"]
    for split in splits:
        path = cache_path(output, split)
        manifest_hash = file_hash(processed / f"{split}.jsonl")
        identity = {"manifest_hash": manifest_hash, "weights_hash": weights_hash,
                    "backbone": config["model"]["backbone"], "preprocess": "full_field_pad_v1"}
        if path.exists():
            previous = torch.load(path, map_location="cpu", weights_only=False)
            if previous["identity"] != identity:
                raise ValueError(f"Stale feature cache: {path}; use a new output directory")
            print(f"Reusing {path}", flush=True)
            continue
        records = read_manifest(processed / f"{split}.jsonl")
        views = [r["views"][c] for r in records for c in CAMERAS if c in r["views"]]
        loader = DataLoader(FullFieldImages(views, preprocess), batch_size=config["runtime"]["cache_batch_size"],
                            num_workers=args.workers, pin_memory=args.device == "cuda")
        features = []
        for images in tqdm(loader, desc=f"Frozen CLIP {split}"):
            with autocast(args.device, args.device == "cuda"):
                features.append(F.normalize(model.encode_image(images.to(args.device)).float(), dim=-1).cpu())
        flat_features = torch.cat(features)
        view_captions = [v["caption"] for v in views]
        scene_captions = [r["caption"] for r in records]
        unique = list(dict.fromkeys(view_captions + scene_captions))
        text_chunks = []
        for start in range(0, len(unique), 128):
            with autocast(args.device, args.device == "cuda"):
                text_chunks.append(F.normalize(model.encode_text(tokenizer(unique[start:start+128]).to(args.device)).float(), dim=-1).cpu())
        text_features = torch.cat(text_chunks)
        texts = {caption: text_features[i] for i, caption in enumerate(unique)}
        features = torch.zeros(len(records), 6, flat_features.shape[1])
        mask = torch.zeros(len(records), 6, dtype=torch.bool)
        labels = torch.zeros(len(records), 6, len(classes))
        view_text = torch.zeros_like(features)
        captions = []
        cursor = 0
        for i, record in enumerate(records):
            group_captions = []
            for j, camera in enumerate(CAMERAS):
                view = record["views"].get(camera)
                group_captions.append(view["caption"] if view else "")
                if view:
                    features[i, j] = flat_features[cursor]
                    view_text[i, j] = texts[view["caption"]]
                    mask[i, j] = True
                    for name in view["objects"]:
                        labels[i, j, classes.index(name)] = 1
                    cursor += 1
            captions.append(group_captions)
        tensors = dict(features=features, mask=mask, labels=labels, view_text=view_text,
                       scene_text=torch.stack([texts[c] for c in scene_captions]))
        if any(not torch.isfinite(t).all() for t in tensors.values()):
            raise ValueError("Non-finite cached features")
        path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(path, dict(tensors, identity=identity, classes=classes, records=records,
                                   captions=captions, scene_captions=scene_captions))
        print(f"Cached {len(views)} images -> {path}", flush=True)


def load_cache(output, processed, split):
    value = torch.load(cache_path(output, split), map_location="cpu", weights_only=False)
    if value["identity"]["manifest_hash"] != file_hash(processed / f"{split}.jsonl"):
        raise ValueError(f"Manifest/cache mismatch: {split}")
    return value


def validate_checkpoint_cache(checkpoint, data, split):
    expected = checkpoint["signature"].get(split)
    if expected is not None and expected != data["identity"]:
        raise ValueError("Checkpoint/cache mismatch")
    for key in ("weights_hash", "backbone", "preprocess"):
        if data["identity"][key] != checkpoint["signature"]["train"][key]:
            raise ValueError(f"Checkpoint/cache {key} mismatch")
    if data["classes"] != checkpoint["classes"]:
        raise ValueError("Checkpoint/cache class vocabulary mismatch")
    if split == "test" and data["identity"]["manifest_hash"] != checkpoint["signature"]["test_manifest"]:
        raise ValueError("Test set changed")


def select_texts(features, captions):
    indices, lookup, groups = [], {}, []
    for i, caption in enumerate(captions):
        if caption not in lookup:
            lookup[caption] = len(indices)
            indices.append(i)
        groups.append(lookup[caption])
    return features[indices], torch.tensor(groups)


@torch.no_grad()
def encode(model, data, device):
    model.eval()
    view_features, scene_features = [], []
    for start in range(0, len(data["features"]), 256):
        mask = data["mask"][start:start+256].to(device)
        output = model(data["features"][start:start+256].to(device), mask)
        if not torch.isfinite(output["view"][mask]).all() or not torch.isfinite(output["scene"]).all():
            raise ValueError("Non-finite retrieval embeddings")
        view_features.append(output["view"][mask].cpu())
        scene_features.append(output["scene"].cpu())
    return torch.cat(view_features), torch.cat(scene_features)


def evaluate(model, data, device):
    views, scenes = encode(model, data, device)
    captions = [c for group, mask in zip(data["captions"], data["mask"]) for c, present in zip(group, mask) if present]
    metrics = {}
    for level, images, text, values in (("view", views, data["view_text"][data["mask"]], captions),
                                       ("scene", scenes, data["scene_text"], data["scene_captions"])):
        unique, groups = select_texts(text, values)
        metrics[level] = retrieval_metrics(images, unique, groups)
    metrics["selection_score"] = sum(metrics[level][f"{direction}_recall_at_10"] for level in ("view", "scene")
                                     for direction in ("image_to_text", "text_to_image")) / 4
    metrics["relevance"] = "exact caption equality, weak-label proxy; not human semantic relevance"
    return metrics


def train(config, args, output, processed):
    train_data, val_data = [load_cache(output, processed, s) for s in ("train", "val")]
    for key in ("weights_hash", "backbone", "preprocess"):
        if train_data["identity"][key] != val_data["identity"][key]:
            raise ValueError("Train/validation encoders must match")
    if train_data["classes"] != val_data["classes"]:
        raise ValueError("Train/validation class vocabularies differ")
    assert_disjoint({s: read_manifest(processed / f"{s}.jsonl") for s in ("train", "val", "test")})
    if (output / "best.pt").exists() and not args.resume:
        raise ValueError("Training already exists; use --resume or a new output directory")
    model = SurroundTransfer(config["model"], len(train_data["classes"])).to(args.device)
    settings = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    signature = {"train": train_data["identity"], "val": val_data["identity"],
                 "test_manifest": file_hash(processed / "test.jsonl"),
                 "model_config": config["model"], "training_config": settings, "seed": config["seed"]}
    start = 0
    if args.resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Resume requires the same model, data and training settings")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start, history = saved["epoch"], saved["history"]
        best, significant, stale = saved["best_score"], saved["significant"], saved["stale"]
        baseline = history[0]["validation"]
    else:
        baseline = evaluate(model, val_data, args.device)
        best = significant = baseline["selection_score"]
        stale = 0
        history = [{"epoch": 0, "validation": baseline}]
    def state(epoch):
        return {"architecture": "SurroundTransfer-v2", "model": model.state_dict(), "model_config": config["model"],
                "classes": train_data["classes"], "signature": signature, "epoch": epoch, "best_score": best}
    if not args.resume:
        save_checkpoint(output / "best.pt", state(0))
        save_json(output / "baseline_validation.json", baseline)
    print(f"Frozen pretrained baseline validation={baseline['selection_score']:.6f}; best={best:.6f}", flush=True)
    tensors = {key: train_data[key].to(args.device) for key in ("features", "mask", "labels", "view_text", "scene_text")}
    positives = tensors["labels"][tensors["mask"]].sum(0)
    pos_weight = ((tensors["mask"].sum()-positives)/positives.clamp_min(1)).clamp(1, 10)
    for epoch in range(start, settings["epochs"]):
        if stale >= settings["patience"]:
            break
        model.train()
        generator = torch.Generator().manual_seed(config["seed"]+epoch)
        order = torch.randperm(len(tensors["features"]), generator=generator)
        losses = []
        for start in range(0, len(order), settings["batch_size"]):
            indices = order[start:start+settings["batch_size"]].tolist()
            x, mask = tensors["features"][indices], tensors["mask"][indices]
            view_captions = [c for i in indices for c, valid in zip(train_data["captions"][i], train_data["mask"][i]) if valid]
            descriptions = [train_data["scene_captions"][i] for i in indices]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, mask, settings["camera_dropout"])
            scale = model.logit_scale.exp().clamp(max=100)
            view_loss = balanced_contrastive_loss(prediction["view"][mask], tensors["view_text"][indices][mask], view_captions, scale)
            scene_loss = balanced_contrastive_loss(prediction["scene"], tensors["scene_text"][indices], descriptions, scale)
            object_loss = F.binary_cross_entropy_with_logits(prediction["object_logits"][mask], tensors["labels"][indices][mask], pos_weight=pos_weight)
            anchor = (1 - (prediction["view"][mask]*x[mask]).sum(-1)).mean()
            anchor += (1 - (prediction["scene"]*prediction["base_scene"]).sum(-1)).mean()
            loss = view_loss + settings["scene_weight"]*scene_loss + settings["object_weight"]*object_loss + settings["anchor_weight"]*anchor
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))
            losses.append(float(loss.detach()))
        metrics = evaluate(model, val_data, args.device)
        score = metrics["selection_score"]
        if score > best:
            best = score
            save_checkpoint(output / "best.pt", state(epoch+1))
        if score > significant + settings["min_delta"]:
            significant, stale = score, 0
        else:
            stale += 1
        history.append({"epoch": epoch+1, "train_loss": float(np.mean(losses)), "validation": metrics})
        save_json(output / "history.json", history)
        last = state(epoch+1)
        last.update(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
                    significant=significant, stale=stale, history=history)
        save_checkpoint(output / "last.pt", last)
        print(f"epoch={epoch+1} loss={np.mean(losses):.4f} validation={score:.6f} best={best:.6f} patience={stale}", flush=True)
        if stale >= settings["patience"]:
            break
    save_json(output / "summary.json", {"best_validation_score": best, "baseline_validation_score": baseline["selection_score"],
                                        "epochs_completed": len(history)-1, "target": .8, "target_reached_on_validation": best >= .8})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cache", "train", "eval", "index", "query"))
    parser.add_argument("--config", default="configs/surround_transfer.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=("cuda", "cpu"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--level", choices=("view", "scene"), default="view")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    config = load_config(args.config)
    args.workers = config["runtime"]["workers"] if args.workers is None else args.workers
    if args.workers < 0 or args.top_k < 1:
        parser.error("workers must be nonnegative; top-k must be positive")
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    torch.set_num_threads(4)
    output = resolve_path(config, config["training"]["output_dir"])
    processed = resolve_path(config, config["dataset"]["processed_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "cache":
        cache(config, args, output, processed)
        return
    if args.command == "train":
        train(config, args, output, processed)
        return
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model = SurroundTransfer(checkpoint["model_config"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device).eval()
    if args.command == "query":
        if not args.query:
            parser.error("--query is required")
        index_dir = output / f"index_{args.split or 'test'}"
        info = json.loads((index_dir / "info.json").read_text())
        if info["checkpoint_hash"] != file_hash(output / "best.pt"):
            raise ValueError("Rebuild index for the current checkpoint")
        if info["manifest_hash"] != file_hash(processed / f"{args.split or 'test'}.jsonl"):
            raise ValueError("Rebuild index for the current manifest")
        base, _, tokenizer, weights_hash = backbone(config, args.device)
        if weights_hash != info["weights_hash"]:
            raise ValueError("Backbone/index mismatch")
        with torch.no_grad(), autocast(args.device, args.device == "cuda"):
            text = F.normalize(base.encode_text(tokenizer([args.query]).to(args.device)).float(), dim=-1).cpu().numpy()[0]
        scores = np.load(index_dir / f"{args.level}.npy", mmap_mode="r") @ text
        mapping = json.loads((index_dir / f"{args.level}.json").read_text(encoding="utf-8"))
        print(json.dumps([dict(mapping[i], score=float(scores[i])) for i in np.argsort(-scores)[:args.top_k]], indent=2))
        return
    split = args.split or "test"
    data = load_cache(output, processed, split)
    validate_checkpoint_cache(checkpoint, data, split)
    if args.command == "eval":
        metrics = evaluate(model, data, args.device)
        metrics.update(split=split, checkpoint=str(output / "best.pt"), architecture=checkpoint["architecture"],
                       checkpoint_epoch=checkpoint["epoch"], limited=False,
                       manifest_hash=data["identity"]["manifest_hash"], target=.8,
                       target_reached=metrics["selection_score"] >= .8)
        save_json(output / f"evaluation_{split}.json", metrics)
        print(json.dumps(metrics, indent=2))
    else:
        views, scenes = encode(model, data, args.device)
        index_dir = output / f"index_{split}"
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "view.npy", views.numpy())
        np.save(index_dir / "scene.npy", scenes.numpy())
        save_json(index_dir / "view.json", [r["views"][c] for r in data["records"] for c in CAMERAS if c in r["views"]])
        save_json(index_dir / "scene.json", data["records"])
        save_json(index_dir / "info.json", {"checkpoint_hash": file_hash(output / "best.pt"),
                                          "weights_hash": data["identity"]["weights_hash"],
                                          "manifest_hash": data["identity"]["manifest_hash"]})
        print(index_dir)


if __name__ == "__main__":
    main()
