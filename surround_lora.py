"""CLIP ViT-L/14 vision LoRA for local six-camera training on an 8 GB GPU."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.surround import CAMERAS, read_manifest
from src.data.surround_dataset import collate_surround
from src.data.surround_lora import LoRASurroundDataset
from src.models.gradient_cache import cached_backward, precision_context
from src.models.surround_lora import SurroundLoRA
from src.models.surround_transfer import balanced_contrastive_loss
from src.utils.config import load_config, resolve_path
from surround import assert_disjoint, file_hash, retrieval_metrics, save_checkpoint, save_json
from surround_transfer import select_texts


BASE_FILES = ("model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json", "merges.txt", "vocab.json")


def download(config):
    from huggingface_hub import snapshot_download
    settings = config["model"]
    path = resolve_path(config, settings["local_dir"])
    snapshot_download(settings["repo"], revision=settings["revision"], local_dir=str(path),
                      allow_patterns=list(BASE_FILES), max_workers=2)
    print(path)


def load_model(config, classes, device):
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast
    settings = config["model"]
    path = resolve_path(config, settings["local_dir"])
    if not all((path / name).exists() for name in BASE_FILES):
        raise FileNotFoundError("Missing CLIP-L files. Run: python surround_lora.py download")
    precision = settings["precision"] if device == "cuda" else "fp32"
    if precision not in ("bf16", "fp32"):
        raise ValueError("Supported precision: bf16 or fp32 (no silent FP16 fallback)")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support BF16; explicitly choose fp32 and recheck memory")
    clip = CLIPModel.from_pretrained(path, local_files_only=True, use_safetensors=True,
                                    torch_dtype=torch.bfloat16 if precision == "bf16" else torch.float32,
                                    attn_implementation="sdpa")
    processor = CLIPImageProcessor.from_pretrained(path, local_files_only=True)
    if clip.config.vision_config.image_size != settings["image_size"] or processor.crop_size["height"] != settings["image_size"]:
        raise ValueError("Image size must match the selected pretrained CLIP")
    model = SurroundLoRA(clip, settings, len(classes)).to(device)
    tokenizer = CLIPTokenizerFast.from_pretrained(path, local_files_only=True)
    identity = {"repo": settings["repo"], "revision": settings["revision"], "precision": precision,
                "files": {name: file_hash(path / name) for name in BASE_FILES}, "preprocess": "clip_pad_full_fov_v1"}
    print(json.dumps(dict(model.parameter_counts(), precision=precision, device=device)), flush=True)
    return model, processor, tokenizer, identity


def loader(records, classes, processor, config, device, training=False, epoch=0):
    batch_size = config["training"]["effective_batch_size"] if training else config["runtime"]["eval_batch_size"]
    return DataLoader(LoRASurroundDataset(records, classes, processor, training), batch_size=batch_size,
                      shuffle=training, num_workers=config["runtime"]["workers"], pin_memory=device == "cuda",
                      collate_fn=collate_surround, generator=torch.Generator().manual_seed(config["seed"] + epoch))


@torch.no_grad()
def text_bank(model, tokenizer, records, config, device):
    # Only text is cacheable across updates: the image encoder's LoRA is trainable.
    captions = list(dict.fromkeys([r["views"][c]["caption"] for r in records for c in CAMERAS if c in r["views"]]
                                 + [r["caption"] for r in records]))
    features = []
    for start in range(0, len(captions), config["runtime"]["text_batch_size"]):
        tokens = tokenizer(captions[start:start+config["runtime"]["text_batch_size"]], padding=True,
                           truncation=True, max_length=model.clip.config.text_config.max_position_embeddings,
                           return_tensors="pt").to(device)
        with precision_context(device, config["model"]["precision"]):
            features.append(model.encode_text(tokens).cpu())
    features = torch.cat(features)
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite text features")
    return {caption: features[i] for i, caption in enumerate(captions)}


def text_rows(bank, captions, device="cpu"):
    return torch.stack([bank[c] for c in captions]).to(device)


@torch.no_grad()
def encode(model, batches, config, device):
    model.eval()
    views, scenes, view_captions, scene_captions, mapping, groups = [], [], [], [], [], []
    for images, mask, _, captions, descriptions, indices in tqdm(batches, desc="Encoding LoRA surround"):
        with precision_context(device, config["model"]["precision"]):
            raw = model.encode_views(images, mask)
        prediction = model.heads(raw, mask.to(device))
        views.append(prediction["view"][mask.to(device)].cpu())
        scenes.append(prediction["scene"].cpu())
        view_captions.extend(c for c, present in zip(captions, mask.flatten().tolist()) if present)
        scene_captions.extend(descriptions)
        for i in indices:
            row = batches.dataset.records[i]
            groups.append(row)
            mapping.extend(row["views"][c] for c in CAMERAS if c in row["views"])
    result = dict(view=torch.cat(views), scene=torch.cat(scenes), view_captions=view_captions,
                  scene_captions=scene_captions, view_mapping=mapping, scene_mapping=groups)
    if not torch.isfinite(result["view"]).all() or not torch.isfinite(result["scene"]).all():
        raise ValueError("Non-finite retrieval embeddings")
    return result


def metrics_for(encoded, bank):
    metrics = {}
    for level in ("view", "scene"):
        captions = encoded[f"{level}_captions"]
        unique, groups = select_texts(text_rows(bank, captions), captions)
        metrics[level] = retrieval_metrics(encoded[level], unique, groups)
    metrics["selection_score"] = sum(metrics[level][f"{direction}_recall_at_10"]
                                     for level in ("view", "scene") for direction in ("image_to_text", "text_to_image")) / 4
    metrics["relevance"] = "exact caption equality, weak-label proxy; not human semantic relevance"
    return metrics


def positive_weights(records, classes, device):
    counts = torch.zeros(len(classes), device=device)
    total = 0
    lookup = {c: i for i, c in enumerate(classes)}
    for record in records:
        for view in record["views"].values():
            total += 1
            for name in view["objects"]:
                counts[lookup[name]] += 1
    return ((total - counts) / counts.clamp_min(1)).clamp(1, 10)


def make_loss(model, batch, bank, settings, pos_weight, device):
    _, mask, labels, captions, descriptions, _ = batch
    captions = [c for c, present in zip(captions, mask.flatten().tolist()) if present]
    mask, labels = mask.to(device), labels.to(device)
    view_text, scene_text = text_rows(bank, captions, device), text_rows(bank, descriptions, device)
    def loss_fn(features):
        prediction = model.heads(features, mask, settings["camera_dropout"])
        scale = model.heads.logit_scale.exp().clamp(max=100)
        loss = balanced_contrastive_loss(prediction["view"][mask], view_text, captions, scale)
        loss += settings["scene_weight"] * balanced_contrastive_loss(prediction["scene"], scene_text, descriptions, scale)
        loss += settings["object_weight"] * F.binary_cross_entropy_with_logits(
            prediction["object_logits"][mask], labels[mask], pos_weight=pos_weight)
        anchor = (1 - (prediction["view"][mask] * features[mask].detach()).sum(-1)).mean()
        anchor += (1 - (prediction["scene"] * prediction["base_scene"].detach()).sum(-1)).mean()
        return loss + settings["anchor_weight"] * anchor
    return loss_fn


def optimizer_for(model, settings):
    return torch.optim.AdamW([
        dict(params=[p for p in model.clip.parameters() if p.requires_grad], lr=settings["learning_rate"]),
        dict(params=list(model.heads.parameters()), lr=settings["head_learning_rate"]),
    ], weight_decay=settings["weight_decay"])


def lr_factor(step, total, warmup_fraction):
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    return .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def optimize_batch(model, batch, bank, settings, pos_weight, optimizer, device, precision):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = cached_backward(model.encode_views, batch[0], batch[1], make_loss(model, batch, bank, settings, pos_weight, device),
                           settings["micro_batch_size"], device, precision)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        model.heads.logit_scale.clamp_(0, math.log(100))
    return loss


def memory_report(device):
    return {} if device != "cuda" else dict(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                                           peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)


def run_training(config, args, output, records, classes, model, processor, tokenizer, signature):
    settings = config["training"]
    smoke = args.command == "smoke"
    train_records, val_records = records["train"], records["val"]
    if smoke:
        order = torch.randperm(len(train_records), generator=torch.Generator().manual_seed(config["seed"])).tolist()
        train_records = [train_records[i] for i in order[:settings["effective_batch_size"] * args.smoke_steps]]
        val_records = val_records[:8]
    train_bank = text_bank(model, tokenizer, train_records, config, args.device)
    val_bank = text_bank(model, tokenizer, val_records, config, args.device)
    pos_weight = positive_weights(records["train"], classes, args.device)
    optimizer = optimizer_for(model, settings)
    started = time.perf_counter()
    if smoke:
        losses = []
        for batch in loader(train_records, classes, processor, config, args.device, training=True):
            losses.append(optimize_batch(model, batch, train_bank, settings, pos_weight, optimizer, args.device, config["model"]["precision"]))
            print(f"smoke_step={len(losses)} loss={losses[-1]:.5f}", flush=True)
        encoded = encode(model, loader(val_records, classes, processor, config, args.device), config, args.device)
        report = dict(model.parameter_counts(), **memory_report(args.device), losses=losses,
                      elapsed_seconds=time.perf_counter()-started, limited=True, quality_evaluation=False,
                      optimizer_steps=len(losses), training_samples=len(train_records),
                      effective_batch_size=settings["effective_batch_size"], micro_batch_size=settings["micro_batch_size"],
                      precision=signature["base"]["precision"],
                      gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                      lora_updated=any(p.detach().abs().max().item() > 0 for n, p in model.named_parameters() if "lora_B" in n),
                      frozen_gradients_absent=all(p.grad is None for p in model.parameters() if not p.requires_grad),
                      validation_encoded_views=len(encoded["view"]), validation_encoded_samples=len(encoded["scene"]),
                      validation_text_captions=len(val_bank))
        state = model.adapter_state()
        save_checkpoint(output / "smoke_adapter.pt", dict(adapter=state, signature=signature, smoke=True))
        model.load_adapter(torch.load(output / "smoke_adapter.pt", map_location="cpu", weights_only=False)["adapter"])
        save_json(output / "smoke_report.json", report)
        print(json.dumps(report, indent=2))
        return
    validation = lambda: metrics_for(encode(model, loader(val_records, classes, processor, config, args.device), config, args.device), val_bank)
    start_epoch, global_step, stale = 0, 0, 0
    if args.resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Resume requires identical base weights, manifests, model and training settings")
        model.load_adapter(saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, global_step, stale = saved["epoch"], saved["global_step"], saved["stale"]
        best, significant, history = saved["best_score"], saved["significant"], saved["history"]
        torch.set_rng_state(saved["torch_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
    else:
        baseline = validation()
        best = significant = baseline["selection_score"]
        history = [dict(epoch=0, validation=baseline)]
        save_json(output / "baseline_validation.json", baseline)
    def state(epoch):
        return dict(architecture="SurroundLoRA-v3", adapter=model.adapter_state(), signature=signature,
                    epoch=epoch, global_step=global_step, best_score=best, classes=classes)
    if not args.resume:
        save_checkpoint(output / "best.pt", state(0))
    steps_per_epoch = math.ceil(len(train_records) / settings["effective_batch_size"])
    total_steps = settings["epochs"] * steps_per_epoch
    print(f"train_samples={len(train_records)} effective_batch={settings['effective_batch_size']} "
          f"micro_batch={settings['micro_batch_size']} steps_per_epoch={steps_per_epoch} best_val={best:.6f}", flush=True)
    for epoch in range(start_epoch, settings["epochs"]):
        if stale >= settings["patience"]:
            break
        losses = []
        batches = tqdm(loader(train_records, classes, processor, config, args.device, training=True, epoch=epoch),
                       desc=f"LoRA epoch {epoch+1}/{settings['epochs']}")
        for batch in batches:
            factor = lr_factor(global_step, total_steps, settings["warmup_fraction"])
            for group, base_lr in zip(optimizer.param_groups, (settings["learning_rate"], settings["head_learning_rate"])):
                group["lr"] = base_lr * factor
            loss = optimize_batch(model, batch, train_bank, settings, pos_weight, optimizer, args.device, config["model"]["precision"])
            global_step += 1
            losses.append(loss)
            batches.set_postfix(loss=f"{np.mean(losses):.4f}")
        metrics = validation()
        score = metrics["selection_score"]
        if score > best:
            best = score
            save_checkpoint(output / "best.pt", state(epoch+1))
        if score > significant + settings["min_delta"]:
            significant, stale = score, 0
        else:
            stale += 1
        history.append(dict(epoch=epoch+1, train_loss=float(np.mean(losses)), validation=metrics))
        last = state(epoch+1)
        last.update(optimizer=optimizer.state_dict(), history=history, stale=stale, significant=significant,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
                    python_rng=random.getstate())
        save_checkpoint(output / "last.pt", last)
        save_json(output / "history.json", history)
        print(f"epoch={epoch+1} validation={score:.6f} best={best:.6f} patience={stale}", flush=True)
    save_json(output / "summary.json", dict(best_validation_score=best, epochs_completed=len(history)-1,
                                            global_steps=global_step, **memory_report(args.device),
                                            parameter_counts=model.parameter_counts(), target=.8,
                                            target_reached_on_validation=best >= .8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "smoke", "train", "eval", "index", "query"))
    parser.add_argument("--config", default="configs/surround_lora.yaml")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--query")
    parser.add_argument("--level", choices=("view", "scene"), default="view")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.workers is not None:
        config["runtime"]["workers"] = args.workers
    if config["runtime"]["workers"] < 0 or args.smoke_steps < 1 or args.top_k < 1:
        parser.error("workers >= 0; smoke-steps/top-k > 0 required")
    for key in ("micro_batch_size", "effective_batch_size", "epochs"):
        if config["training"][key] < 1:
            parser.error(f"{key} must be positive")
    if config["training"]["micro_batch_size"] > config["training"]["effective_batch_size"]:
        parser.error("micro_batch_size must not exceed effective_batch_size")
    if args.resume and args.command != "train":
        parser.error("--resume applies only to train")
    if args.command == "download":
        download(config)
        return
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    output = resolve_path(config, config["training"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "train" and (output / "best.pt").exists() and not args.resume:
        parser.error("Training already exists; use --resume or a new output directory")
    if args.resume and not (output / "last.pt").exists():
        parser.error("No epoch-boundary last.pt available to resume; use a new output directory")
    processed = resolve_path(config, config["dataset"]["processed_dir"])
    records = {split: read_manifest(processed / f"{split}.jsonl") for split in ("train", "val", "test")}
    assert_disjoint(records)
    classes = json.loads((processed / "stats.json").read_text(encoding="utf-8"))["classes"]
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, processor, tokenizer, identity = load_model(config, classes, args.device)
    signature = dict(base=identity, manifests={s: file_hash(processed / f"{s}.jsonl") for s in records}, classes=classes,
                     model=config["model"], training=config["training"], runtime=config["runtime"], seed=config["seed"])
    if args.command in ("train", "smoke"):
        run_training(config, args, output, records, classes, model, processor, tokenizer, signature)
        return
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    for key in ("base", "manifests", "classes", "model"):
        if checkpoint["signature"][key] != signature[key]:
            raise ValueError(f"Checkpoint {key} mismatch")
    model.load_adapter(checkpoint["adapter"])
    model.eval()
    index_dir = output / f"index_{args.split}"
    index_identity = dict(checkpoint_hash=file_hash(output / "best.pt"), base=identity,
                          manifest_hash=signature["manifests"][args.split])
    if args.command == "query":
        if not args.query:
            parser.error("--query is required")
        if json.loads((index_dir / "info.json").read_text()) != index_identity:
            raise ValueError("Index is stale; rebuild it with the selected checkpoint")
        tokens = tokenizer([args.query], return_tensors="pt", truncation=True,
                           max_length=model.clip.config.text_config.max_position_embeddings).to(args.device)
        with precision_context(args.device, config["model"]["precision"]):
            query = model.encode_text(tokens).cpu().numpy()[0]
        scores = np.load(index_dir / f"{args.level}.npy", mmap_mode="r") @ query
        mapping = json.loads((index_dir / f"{args.level}.json").read_text(encoding="utf-8"))
        print(json.dumps([dict(mapping[i], score=float(scores[i])) for i in np.argsort(-scores)[:args.top_k]], indent=2))
        return
    selected = records[args.split]
    encoded = encode(model, loader(selected, classes, processor, config, args.device), config, args.device)
    if args.command == "eval":
        bank = text_bank(model, tokenizer, selected, config, args.device)
        metrics = metrics_for(encoded, bank)
        metrics.update(split=args.split, checkpoint=str(output / "best.pt"), checkpoint_epoch=checkpoint["epoch"],
                       architecture=checkpoint["architecture"], limited=False, manifest_hash=index_identity["manifest_hash"],
                       target=.8, target_reached=metrics["selection_score"] >= .8)
        save_json(output / f"evaluation_{args.split}.json", metrics)
        print(json.dumps(metrics, indent=2))
    else:
        index_dir.mkdir(parents=True, exist_ok=True)
        for level in ("view", "scene"):
            np.save(index_dir / f"{level}.npy", encoded[level].numpy())
            save_json(index_dir / f"{level}.json", encoded[f"{level}_mapping"])
        save_json(index_dir / "info.json", index_identity)
        print(index_dir)


if __name__ == "__main__":
    main()
