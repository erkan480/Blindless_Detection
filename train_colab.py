"""Portable APTOS transfer learning; validation-only selection, explicit final test."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score
from src.ai.retina import CLASSES, PREPROCESS, ROOT, build_model, transform
from src.ai.preprocessing import crop_retina

SIZES = {"efficientnet_b0": 224, "efficientnet_b3": 300, "efficientnet_v2_s": 384}


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def manifests(directory):
    """Ignore local paths; retain audited IDs, labels, original pixel hashes/order."""
    result, ids, hashes = {}, set(), set()
    for split in ("train", "val", "test"):
        with (directory / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Empty split: {split}")
        for row in rows:
            ident, digest, label = row["id_code"], row["sha256"], int(row["diagnosis"])
            if not ident or any(c not in "0123456789abcdef" for c in ident.lower()):
                raise ValueError("Unexpected APTOS image ID")
            if label not in range(5) or len(digest) != 64:
                raise ValueError("Invalid label/hash")
            if ident in ids or digest in hashes:
                raise ValueError("Duplicate image ID/hash within or across audited splits")
            ids.add(ident)
            hashes.add(digest)
        result[split] = rows
    serial = {s: [{k: r[k] for k in ("id_code", "diagnosis", "sha256")} for r in rows]
              for s, rows in result.items()}
    digest = hashlib.sha256(json.dumps(serial, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result, digest


def prepare(args):
    rows, digest = manifests(args.manifests)
    if args.source is None:
        raise ValueError("--source must contain official train.csv and train_images/")
    with (args.source / "train.csv").open(encoding="utf-8-sig", newline="") as f:
        labels = {r["id_code"]: int(r["diagnosis"]) for r in csv.DictReader(f)}
    args.cache.mkdir(parents=True, exist_ok=True)
    atomic_json({"status": "building", "split_digest": digest, "cache_size": args.cache_size}, args.cache / "cache.json")
    count = 0
    for records in rows.values():
        for row in records:
            ident = row["id_code"]
            if labels.get(ident) != int(row["diagnosis"]):
                raise ValueError(f"Original label mismatch: {ident}")
            with Image.open(args.source / "train_images" / f"{ident}.png") as image:
                image.load()
                rgb = image.convert("RGB")
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                if pixel_hash != row["sha256"]:
                    raise ValueError(f"Original image mismatch: {ident}")
                target, temporary = args.cache / f"{ident}.png", args.cache / f"{ident}.tmp.png"
                crop_retina(rgb).resize((args.cache_size, args.cache_size), Image.Resampling.LANCZOS).save(temporary)
                os.replace(temporary, target)
            count += 1
            if count % 200 == 0:
                print(f"Verified/cache rebuilt: {count}", flush=True)
    atomic_json({"status": "ready", "split_digest": digest, "cache_size": args.cache_size,
                 "preprocess": PREPROCESS, "images": count}, args.cache / "cache.json")
    print(f"READY: {count} original images verified; original splits preserved", flush=True)


def training_transform(size):
    # Mild scale/rotation changes keep the retina; no augmentation on validation/test.
    return transforms.Compose([
        transforms.Resize((size, size)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=20, scale=(.95, 1.05), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=.12, contrast=.12, saturation=.05), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


class Retinas(Dataset):
    def __init__(self, rows, cache, size, training=False):
        self.rows, self.cache = rows, cache
        self.transform = training_transform(size) if training else transform(size, False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.cache / f'{row["id_code"]}.png') as image:
            x = self.transform(image.convert("RGB"))
        return x, int(row["diagnosis"])


def metrics(labels, logits):
    predictions = logits.argmax(1)
    qwk = cohen_kappa_score(labels, predictions, weights="quadratic", labels=list(range(5)))
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=list(range(5)), average="macro", zero_division=0)),
            "qwk": float(qwk) if np.isfinite(qwk) else None,
            "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(5))).tolist(),
            "classes": classification_report(labels, predictions, labels=list(range(5)), target_names=CLASSES,
                                             output_dict=True, zero_division=0)}


def selective_metrics(labels, logits, temperature, threshold):
    scores, predicted = (logits / temperature).softmax(1).max(1)
    accepted = scores >= threshold
    return {"coverage": float(accepted.float().mean()), "accepted_count": int(accepted.sum()),
            "accepted_accuracy": float((predicted[accepted] == labels[accepted]).float().mean()) if accepted.any() else None,
            "temperature": temperature, "threshold": threshold,
            "note": "Softmax/coverage are research diagnostics, not medical reliability guarantees."}


def loader(rows, args, training=False, smoke=False):
    dataset = Retinas(rows, args.cache, args.image_size, training)
    if smoke:
        dataset = Subset(dataset, list(range(min(len(dataset), args.batch_size * 2))))
    # Fresh workers each epoch derive their seeds from the saved parent Torch RNG.
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=training, num_workers=args.workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=False)


def evaluate(model, data_loader, device):
    model.eval()
    labels, logits = [], []
    with torch.inference_mode():
        for x, y in data_loader:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(x.to(device))
            logits.append(output.float().cpu())
            labels.append(y)
    return torch.cat(labels), torch.cat(logits)


def checked_data(args):
    rows, digest = manifests(args.manifests)
    meta = json.loads((args.cache / "cache.json").read_text(encoding="utf-8"))
    if (meta.get("status") != "ready" or meta.get("split_digest") != digest or
            meta.get("preprocess") != PREPROCESS or meta.get("cache_size") != args.cache_size):
        raise ValueError("Cache incomplete or split/preprocessing mismatch. Run prepare.")
    return rows, digest


def device_for(args):
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        return torch.device("cuda")
    if args.action != "smoke":
        raise RuntimeError("Select T4 GPU in Colab. This notebook requires CUDA; TPU/XLA is not supported.")
    return torch.device("cpu")


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_config(args, digest):
    names = ("architecture", "image_size", "cache_size", "batch_size", "workers", "seed", "epochs", "warmup_epochs",
             "patience", "mixup", "lr", "head_lr", "weight_decay", "threshold")
    return {**{name: getattr(args, name) for name in names}, "split_digest": digest,
            "preprocess": PREPROCESS, "recipe": "colab-v1"}


def checkpoint_metadata(args, digest):
    return {"classes": CLASSES, "preprocess": PREPROCESS, "architecture": args.architecture,
            "image_size": args.image_size, "cache_size": args.cache_size,
            "split_digest": digest, "temperature": 1., "threshold": args.threshold}


def train(args):
    rows, digest = checked_data(args)
    smoke = args.action == "smoke"
    device = device_for(args)
    args.output.mkdir(parents=True, exist_ok=True)
    last, best_path = args.output / "last.pt", args.output / "best.pt"
    if (last.exists() or best_path.exists()) and not args.resume:
        raise FileExistsError("Run exists. Choose another output directory or use --resume.")
    if args.resume and not (last.exists() or best_path.exists()):
        raise FileNotFoundError("No last.pt or best.pt to resume")
    if args.resume and smoke:
        raise ValueError("Smoke checks do not resume")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    train_loader, val_loader = loader(rows["train"], args, True, smoke), loader(rows["val"], args, False, smoke)
    config = run_config(args, digest)
    model = build_model(pretrained=not (args.resume or smoke), architecture=args.architecture).to(device)
    counts = np.bincount([int(r["diagnosis"]) for r in rows["train"]], minlength=5)
    if np.any(counts == 0):
        raise ValueError("Every training class must be represented")
    weights = torch.tensor(1 / np.sqrt(counts), dtype=torch.float32, device=device)
    weights /= weights.mean()
    # One balancing method: class-weighted loss, no oversampling.
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW([{"params": model.features.parameters(), "lr": args.lr},
                                   {"params": model.classifier.parameters(), "lr": args.head_lr}],
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start, best, stale = 0, -float("inf"), 0
    if args.resume:
        checkpoints = [torch.load(path, map_location="cpu", weights_only=True)
                       for path in (last, best_path) if path.exists()]
        if any(checkpoint["config"] != config for checkpoint in checkpoints):
            raise ValueError("Resume config differs. Preserve model, split, cache and training parameters.")
        # best is saved first. A crash before last is replaced can leave best one
        # epoch ahead (or the only file): resume its complete optimizer/RNG state.
        ck = max(checkpoints, key=lambda checkpoint: checkpoint["epoch"])
        if not best_path.exists():
            raise FileNotFoundError("Best checkpoint missing; restore it before resuming this run")
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        scheduler.load_state_dict(ck["scheduler"])
        start, best, stale = ck["epoch"] + 1, ck["best_qwk"], ck["stale"]
        restore_rng(ck["rng"])
    atomic_json({**config, "torch": str(torch.__version__), "device": str(device),
                 "initialization": "random (smoke only)" if smoke else "ImageNet1K",
                 "splits": {s: len(r) for s, r in rows.items()}}, args.output / "config.json")
    for epoch in range(start, 1 if smoke else args.epochs):
        if not smoke and stale >= args.patience and epoch >= args.warmup_epochs + 3:
            break
        started, warmup = time.time(), epoch < args.warmup_epochs and not smoke
        for parameter in model.features.parameters():
            parameter.requires_grad_(not warmup)
        model.train()
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        total, n = 0., 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            use_mixup = args.mixup > 0 and not warmup and len(y) > 1
            lam = float(np.random.beta(args.mixup, args.mixup)) if use_mixup else 1.
            order = torch.randperm(len(y), device=device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(lam * x + (1 - lam) * x[order] if use_mixup else x)
                loss = lam * criterion(output, y) + (1 - lam) * criterion(output, y[order])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(y)
            n += len(y)
            if step % 40 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs}: batch {step + 1}/{len(train_loader)}, loss={float(loss.detach()):.4f}", flush=True)
        labels, logits = evaluate(model, val_loader, device)
        if logits.shape != (len(labels), 5) or not torch.isfinite(logits).all():
            raise RuntimeError("Invalid validation output")
        m = metrics(labels.numpy(), logits.numpy())
        if smoke:
            atomic_save({**checkpoint_metadata(args, digest), "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                         "display_name": "SMOKE ONLY - untrained", "smoke_only": True}, args.output / "smoke_only.pt")
            atomic_json({"status": "passed", "batches": len(train_loader), "validation_examples": len(labels),
                         "architecture": args.architecture, "device": str(device)}, args.output / "smoke.json")
            print("SMOKE PASSED: real image forward/backward + validation + portable checkpoint. Not a trained model.", flush=True)
            return
        if m["qwk"] is None:
            raise RuntimeError("Validation QWK is undefined")
        improved = m["qwk"] > best
        best, stale = (m["qwk"], 0) if improved else (best, stale + 1)
        scheduler.step(m["qwk"])
        state = {**checkpoint_metadata(args, digest), "state_dict": model.state_dict(), "epoch": epoch,
                 "best_qwk": best, "stale": stale, "config": config, "rng": rng_state(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()}
        if improved:
            atomic_save(state, best_path)
        atomic_save(state, last)
        event = {"epoch": epoch + 1, "loss": total / n, "seconds": time.time() - started, **m}
        with (args.output / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        atomic_json({"status": "training", "last_epoch": epoch + 1, "best_val_qwk": best}, args.output / "status.json")
        print(f"Validation QWK={m['qwk']:.4f}, macro F1={m['macro_f1']:.4f}, severe recall={m['classes']['Severe']['recall']:.4f}; best={best:.4f}", flush=True)
    atomic_json({"status": "trained", "next": "export; test only after final model selection", "best_val_qwk": best}, args.output / "status.json")
    print("TRAINING COMPLETE. No test images evaluated. Next: export.", flush=True)


def validate_checkpoint(ck, args, digest):
    for key, value in checkpoint_metadata(args, digest).items():
        if key not in ("temperature", "threshold") and ck.get(key) != value:
            raise ValueError(f"Checkpoint contract mismatch: {key}")


def export(args):
    rows, digest = checked_data(args)
    device = device_for(args)
    ck = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
    validate_checkpoint(ck, args, digest)
    if args.threshold != ck["threshold"]:
        raise ValueError("Export threshold must match this run's preregistered threshold")
    if (args.output / "test_metrics.json").exists():
        raise RuntimeError("Final test recorded; do not recalibrate this run after seeing test results.")
    model = build_model(architecture=args.architecture).to(device)
    model.load_state_dict(ck["state_dict"])
    labels, logits = evaluate(model, loader(rows["val"], args), device)
    candidates = torch.logspace(math.log10(.25), math.log10(8.), 161)
    temperature = float(min(candidates, key=lambda t: float(nn.functional.cross_entropy(logits / t, labels))))
    report = {**metrics(labels.numpy(), logits.numpy()), **selective_metrics(labels, logits, temperature, args.threshold),
              "nll_before": float(nn.functional.cross_entropy(logits, labels)),
              "nll_after": float(nn.functional.cross_entropy(logits / temperature, labels)),
              "selected_epoch": ck["epoch"] + 1, "split_digest": digest}
    destination = args.output / f"aptos_{args.architecture}.pt"
    artifact = {**checkpoint_metadata(args, digest), "state_dict": ck["state_dict"], "temperature": temperature,
                "selected_epoch": ck["epoch"] + 1, "initialization": "ImageNet1K",
                "display_name": args.architecture.replace("_", " ") + " / APTOS Colab",
                "validation_metrics": report, "training_recipe": "colab-v1"}
    atomic_save(artifact, destination)
    atomic_json(report, args.output / "validation_metrics.json")
    atomic_json({"status": "exported", "model": destination.name, "test_evaluated": False}, args.output / "status.json")
    print(f"EXPORTED: {destination}. Validation calibration only; no test evaluation.", flush=True)


def test(args):
    if not args.confirm_final_model:
        raise RuntimeError("Select final model on validation; then pass --confirm-final-model explicitly.")
    if (args.output / "test_metrics.json").exists():
        raise FileExistsError("Final test recorded. Read test_metrics.json; do not retune using this set.")
    rows, digest = checked_data(args)
    device = device_for(args)
    checkpoint = args.output / f"aptos_{args.architecture}.pt"
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint(ck, args, digest)
    model = build_model(architecture=args.architecture).to(device)
    model.load_state_dict(ck["state_dict"])
    labels, logits = evaluate(model, loader(rows["test"], args), device)
    report = {**metrics(labels.numpy(), logits.numpy()),
              **selective_metrics(labels, logits, ck["temperature"], ck["threshold"]),
              "split_digest": digest, "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "note": "Held-out labeled APTOS train subset; competition test.csv has no public labels. This set was inspected for the B0 baseline, so it is a development benchmark, not fresh external validation."}
    atomic_json(report, args.output / "test_metrics.json")
    scores, predicted = (logits / ck["temperature"]).softmax(1).max(1)
    with (args.output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id_code", "true", "predicted", "score", "accepted"])
        for row, y, p, score in zip(rows["test"], labels, predicted, scores):
            writer.writerow([row["id_code"], int(y), int(p), float(score), float(score) >= ck["threshold"]])
    atomic_json({"status": "final_test_complete", "model": checkpoint.name}, args.output / "status.json")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "smoke", "export", "test"])
    parser.add_argument("--architecture", choices=list(SIZES), default="efficientnet_b3")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "cache512")
    parser.add_argument("--cache-size", type=int, default=512)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "colab_b3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixup", type=float, default=.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=.8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-final-model", action="store_true")
    args = parser.parse_args()
    args.image_size = args.image_size or SIZES[args.architecture]
    if not (128 <= args.image_size <= 512 and args.image_size <= args.cache_size <= 1024):
        parser.error("Require 128 <= image_size <= 512 and image_size <= cache_size <= 1024")
    if not 0 < args.threshold < 1 or args.mixup < 0 or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid threshold/mixup/batch/workers")
    if args.epochs < 1 or args.patience < 1 or args.warmup_epochs < 0 or args.lr <= 0 or args.head_lr <= 0:
        parser.error("Invalid training settings")
    try:
        {"prepare": prepare, "train": train, "smoke": train, "export": export, "test": test}[args.action](args)
    except Exception as error:
        if args.action in ("train", "smoke") and not isinstance(error, (FileExistsError, FileNotFoundError, ValueError)):
            atomic_json({"status": "interrupted_or_failed", "error": str(error),
                         "resume": "Identical configuration + --resume; resumes last completed epoch."}, args.output / "status.json")
        raise


if __name__ == "__main__":
    main()
