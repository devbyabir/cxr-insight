"""Evaluate a trained classifier once on the frozen test split.

Writes raw per-image probabilities, per-class metrics with confusion counts,
patient-level bootstrap CI for macro AUROC, and ROC / confusion plots.
Thresholds come from the checkpoint (fitted on validation), never from test.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .cnn import CXRDataset, load_classifier, read_split
from .metrics import bootstrap_macro_auroc, classification_report, roc_curve
from .nih import NIH_LABELS
from .train_cnn import collect


def plot(report, y, s, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for k, row in enumerate(report["per_class"]):
        if row["auroc"] is None:
            continue
        fpr, tpr = roc_curve(y[:, k], s[:, k])
        # Legend shows the exact computed AUROC — no adjustment of any kind.
        ax.plot(fpr, tpr, lw=1.2, label=f"{row['label']} ({row['auroc']:.3f}, n+={row['support']})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    macro = report["macro_auroc"]
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Test ROC (macro AUROC " + (f"{macro:.3f})" if macro is not None else "undefined)"))
    if macro is not None:
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 7, figsize=(16, 5))
    for ax, row in zip(axes.flat, report["per_class"]):
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, value, ha="center", va="center", fontsize=8)
        ax.set_title(row["label"], fontsize=8)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=6)
        ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=6)
    fig.suptitle("Per-class confusion matrices (test, validation-fitted thresholds)")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=False)
    model, checkpoint = load_classifier(args.checkpoint)
    rows = read_split(args.splits / "test.jsonl")
    loader = torch.utils.data.DataLoader(CXRDataset(rows, args.image_dir, "test"),
                                         batch_size=32, num_workers=args.workers)
    y, s = collect(model, loader, "cpu")
    report = classification_report(y, s, NIH_LABELS, checkpoint["thresholds"])
    report["macro_auroc_ci95"] = bootstrap_macro_auroc(
        y, s, [r["patient_id"] for r in rows], args.bootstrap)
    summary = json.loads((args.splits / "summary.json").read_text(encoding="utf-8"))
    report["data"] = {"dataset": summary["dataset"], "subset": summary["subset"],
                      "images_used_total": summary["images_used"],
                      "test_images": len(rows),
                      "test_patients": len({r["patient_id"] for r in rows})}
    report["checkpoint_git"] = checkpoint["metadata"].get("git")
    with (args.out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", "patient_id", "labels", *NIH_LABELS])
        for row, probs in zip(rows, s):
            writer.writerow([row["image_path"], row["patient_id"], row["labels"],
                             *(f"{p:.6f}" for p in probs)])
    (args.out / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot(report, y, s, args.out)
    print(json.dumps({k: report[k] for k in ("n_images", "macro_auroc", "macro_auroc_ci95",
                                             "micro_f1", "data")}, indent=2))


if __name__ == "__main__":
    main()
