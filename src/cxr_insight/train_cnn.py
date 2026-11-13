"""Train the multi-label CNN with per-epoch checkpoints and automatic resume.

The checkpoint with the best *validation* macro AUROC is kept; per-class
decision thresholds are then fitted on validation data. The test split is not
read here — evaluate it once, afterwards, with ``cxr-evaluate-cnn``.

If ``<out>/last.pt`` exists the run resumes from it (model, optimiser, scheduler,
epoch, history and RNG state), so an interrupted Colab session can continue.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .cnn import CXRDataset, build_model, read_split, save_classifier
from .metrics import choose_thresholds, classification_report
from .nih import NIH_LABELS
from .runinfo import run_metadata


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    scores, targets = [], []
    for images, labels in loader:
        scores.append(torch.sigmoid(model(images.to(device))).cpu())
        targets.append(labels)
    return torch.cat(targets).numpy().astype(int), torch.cat(scores).numpy()


def train_epoch(model, loader, optimiser, scheduler, loss_fn, device, log_every=50):
    model.train()
    total, seen, start = 0.0, 0, time.time()
    for step, (images, labels) in enumerate(loader, 1):
        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(model(images.to(device)), labels.to(device))
        loss.backward()
        optimiser.step()
        scheduler.step()
        total += loss.item() * len(images)
        seen += len(images)
        if step % log_every == 0:
            print(f"  step {step}/{len(loader)} loss {total / seen:.4f} "
                  f"({time.time() - start:.0f}s)", flush=True)
    return total / max(seen, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, type=Path, help="Directory from cxr-nih-prepare")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--session-epochs", type=int, default=None,
                        help="Stop after this many epochs in this session (resume later)")
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    train_rows = read_split(args.splits / "train.jsonl")
    val_rows = read_split(args.splits / "validation.jsonl")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        CXRDataset(train_rows, args.image_dir, "train"), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, generator=generator)
    val_loader = torch.utils.data.DataLoader(
        CXRDataset(val_rows, args.image_dir, "validation"), batch_size=args.batch_size,
        num_workers=args.workers)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    start_epoch, best_auroc, history = 0, -1.0, []

    last = args.out / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["args"]["epochs"] != args.epochs or state["args"]["batch_size"] != args.batch_size:
            raise SystemExit("last.pt was created with different epochs/batch size; use a new --out")
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_auroc, history = state["epoch"], state["best_auroc"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["loader_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        print(f"Resumed from epoch {start_epoch} (best val macro AUROC {best_auroc:.4f})")

    summary_path = args.splits / "summary.json"
    data_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    stop = args.epochs if args.session_epochs is None else min(args.epochs, start_epoch + args.session_epochs)
    for epoch in range(start_epoch, stop):
        print(f"Epoch {epoch + 1}/{args.epochs} on {device}", flush=True)
        loss = train_epoch(model, train_loader, optimiser, scheduler, loss_fn, device)
        y_val, s_val = collect(model, val_loader, device)
        report = classification_report(y_val, s_val, NIH_LABELS, [0.5] * len(NIH_LABELS))
        val_auroc = report["macro_auroc"] or 0.0
        history.append({"epoch": epoch + 1, "train_loss": loss, "val_macro_auroc": val_auroc})
        print(f"  train loss {loss:.4f} | val macro AUROC {val_auroc:.4f} "
              f"({report['auroc_defined_for']} labels defined)", flush=True)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), args.out / "best_weights.pt")
        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1,
                    "best_auroc": best_auroc, "history": history, "args": vars(args) | {
                        "splits": str(args.splits), "image_dir": str(args.image_dir),
                        "out": str(args.out)},
                    "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state()},
                   last.with_suffix(".tmp"))
        last.with_suffix(".tmp").replace(last)  # atomic: never leave a half-written checkpoint

    if stop < args.epochs:
        print(f"Paused after epoch {stop}/{args.epochs}. Re-run the same command to resume.")
        return
    model.load_state_dict(torch.load(args.out / "best_weights.pt", map_location=device))
    y_val, s_val = collect(model, val_loader, device)
    thresholds = choose_thresholds(y_val, s_val)
    val_report = classification_report(y_val, s_val, NIH_LABELS, thresholds)
    metadata = run_metadata(args.seed, {
        "task": "multi-label (14 NIH findings; 'No Finding' = all zeros)",
        "architecture": "torchvision efficientnet_b0, ImageNet-1k weights" if not args.no_pretrained
        else "efficientnet_b0, random init",
        "device": device, "args": {k: str(v) for k, v in vars(args).items() if k != "session_epochs"},
        "data": {k: data_summary.get(k) for k in ("dataset", "source", "subset", "images_used",
                                                  "seed", "data_entry_sha256", "test_list_sha256")}
        | {"split_sha256": {k: v.get("sha256") for k, v in data_summary.get("splits", {}).items()}},
        "history": history, "best_val_macro_auroc": best_auroc,
        "thresholds_fitted_on": "validation",
    })
    save_classifier(args.out / "classifier.pt", model.cpu(), thresholds, metadata)
    (args.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.out / "validation_metrics.json").write_text(json.dumps(val_report, indent=2), encoding="utf-8")
    print(f"Saved {args.out / 'classifier.pt'} (best val macro AUROC {best_auroc:.4f})")


if __name__ == "__main__":
    main()
