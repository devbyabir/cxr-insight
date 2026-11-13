"""Build patient-grouped, multi-label splits for NIH ChestX-ray14.

Data source: NIH Clinical Center, https://nihcc.app.box.com/v/ChestXray-NIHCC
(cite Wang et al., CVPR 2017; acknowledge the NIH Clinical Center as provider).

Each image may carry several findings ("Effusion|Mass"); "No Finding" means all
14 labels are 0 and is not a class. The official ``test_list.txt`` is NIH's
patient-wise test partition and is kept as the frozen test set; the remaining
images are split into train/validation by patient. When only part of the image
archive is present, results describe that subset, not the full dataset.
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from .data import digest, validate_splits

NIH_LABELS = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema",
    "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
)
NO_FINDING = "No Finding"


def parse_labels(text):
    """'Effusion|Mass' -> ('Effusion', 'Mass'); 'No Finding' -> ()."""
    parts = [p.strip() for p in text.split("|") if p.strip()]
    if parts == [NO_FINDING]:
        return ()
    unknown = set(parts) - set(NIH_LABELS)
    if not parts or unknown or NO_FINDING in parts:
        raise ValueError(f"Invalid finding labels: {text!r}")
    return tuple(sorted(set(parts), key=NIH_LABELS.index))


def to_multi_hot(labels):
    present = set(labels)
    return [int(label in present) for label in NIH_LABELS]


def read_list(path):
    return {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}


def build_manifest(data_entry, image_dir, test_list, hash_images=True):
    """Rows for every metadata entry whose image file is present."""
    image_dir = Path(image_dir)
    # Skip macOS archive metadata (__MACOSX/, "._name.png"): same names, not images.
    available = {p.name: p for p in image_dir.rglob("*.png")
                 if not p.name.startswith("._") and "__MACOSX" not in p.parts}
    test_names = read_list(test_list)
    rows, missing = [], 0
    with Path(data_entry).open(encoding="utf-8-sig", newline="") as stream:
        for entry in csv.DictReader(stream):
            name = entry["Image Index"]
            path = available.get(name)
            if path is None:
                missing += 1
                continue
            rows.append({
                "image_path": path.relative_to(image_dir).as_posix(),
                "patient_id": str(int(entry["Patient ID"])),
                "labels": "|".join(parse_labels(entry["Finding Labels"])),
                "view": entry["View Position"],
                "official_partition": "test" if name in test_names else "train_val",
                "sha256": digest(path) if hash_images else name,
            })
    if not rows:
        raise ValueError("No metadata rows matched images on disk")
    return rows, missing


def split(rows, seed=42, validation=0.15):
    """Frozen official test set; train/validation split by patient from the rest."""
    if not 0 < validation < 1:
        raise ValueError("validation must be in (0, 1)")
    rows = sorted(rows, key=lambda r: (int(r["patient_id"]), r["image_path"]))
    test = [r for r in rows if r["official_partition"] == "test"]
    rest = [r for r in rows if r["official_partition"] != "test"]
    patients = sorted({r["patient_id"] for r in rest})
    if len(patients) < 2:
        raise ValueError("Need at least two train_val patients")
    random.Random(seed).shuffle(patients)
    n_val = max(1, round(len(patients) * validation))
    val_patients = set(patients[:n_val])
    result = {
        "train": [r for r in rest if r["patient_id"] not in val_patients],
        "validation": [r for r in rest if r["patient_id"] in val_patients],
        "test": test,
    }
    validate_splits(result)
    return result


def label_counts(rows):
    counts = Counter(label for r in rows for label in r["labels"].split("|") if label)
    return {label: counts.get(label, 0) for label in NIH_LABELS} | {
        NO_FINDING: sum(1 for r in rows if not r["labels"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-entry", required=True, type=Path)
    parser.add_argument("--test-list", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation", type=float, default=0.15)
    args = parser.parse_args(argv)
    rows, missing = build_manifest(args.data_entry, args.image_dir, args.test_list)
    splits = split(rows, args.seed, args.validation)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "dataset": "NIH ChestX-ray14",
        "source": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "subset": missing > 0,
        "images_used": len(rows), "metadata_rows_without_image": missing,
        "seed": args.seed, "validation_fraction": args.validation,
        "data_entry_sha256": digest(args.data_entry), "test_list_sha256": digest(args.test_list),
        "splits": {},
    }
    for name, part in splits.items():
        path = args.output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in part), encoding="utf-8")
        summary["splits"][name] = {
            "images": len(part), "patients": len({r["patient_id"] for r in part}),
            "label_counts": label_counts(part), "sha256": digest(path)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
