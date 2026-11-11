"""Validate, deduplicate, and split local image/report manifests by patient."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

REQUIRED = {"image_path", "patient_id", "report", "report_source"}
SOURCES = {"synthetic", "clinician", "unknown"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_manifest(path, image_root):
    root = Path(image_root).resolve()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(f"Required columns: {sorted(REQUIRED)}")
        rows = []
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"Malformed CSV at row {line}")
            row = {k: (v or "").strip() for k, v in row.items()}
            if any(not row[k] for k in REQUIRED):
                raise ValueError(f"Missing required value at row {line}")
            if row["report_source"] not in SOURCES:
                raise ValueError(f"Invalid report_source at row {line}")
            image = (root / row["image_path"]).resolve()
            if not image.is_relative_to(root):
                raise ValueError(f"Image escapes image root at row {line}")
            if not image.is_file():
                raise ValueError(f"Image missing at row {line}")
            row["image_path"] = image.relative_to(root).as_posix()
            row["sha256"] = digest(image)
            rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def deduplicate(rows):
    unique = {}
    for row in rows:
        key = row["sha256"]
        if key in unique:
            previous = unique[key]
            # Identity conflicts cannot be silently resolved: patient-disjoint
            # splits are meaningless if one image maps to multiple patients.
            for field in ("patient_id", "report", "report_source", "labels"):
                if previous.get(field, "") != row.get(field, ""):
                    raise ValueError(f"Duplicate image has conflicting {field}")
        else:
            unique[key] = row
    return sorted(unique.values(), key=lambda row: (row["patient_id"], row["sha256"]))


def split_patients(rows, seed=42, validation=0.15, test=0.15):
    if not (0 < validation < 1 and 0 < test < 1 and validation + test < 1):
        raise ValueError("Validation/test fractions must be positive and sum to < 1")
    patients = sorted({row["patient_id"] for row in rows})
    if len(patients) < 3:
        raise ValueError("At least three distinct patient groups are required")
    random.Random(seed).shuffle(patients)
    n_val = max(1, int(len(patients) * validation))
    n_test = max(1, int(len(patients) * test))
    if n_val + n_test >= len(patients):
        raise ValueError("Split leaves no training patients")
    mapping = {p: "test" for p in patients[:n_test]}
    mapping.update({p: "validation" for p in patients[n_test:n_test + n_val]})
    result = {name: [] for name in ("train", "validation", "test")}
    for row in rows:
        result[mapping.get(row["patient_id"], "train")].append(row)
    validate_splits(result)
    return result


def validate_splits(splits):
    for field in ("patient_id", "sha256"):
        seen = set()
        for name, rows in splits.items():
            values = {r[field] for r in rows}
            if values & seen:
                raise ValueError(f"Cross-split {field} overlap in {name}")
            seen.update(values)
    if any(not rows for rows in splits.values()):
        raise ValueError("All partitions must contain examples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    original = read_manifest(args.manifest, args.image_root)
    unique = deduplicate(original)
    splits = split_patients(unique, args.seed)
    # Never overwrite a frozen split by accident.
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"seed": args.seed, "input_sha256": digest(args.manifest),
               "input_rows": len(original), "unique_images": len(unique),
               "removed_duplicates": len(original) - len(unique), "splits": {}}
    for name, rows in splits.items():
        path = args.output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        summary["splits"][name] = {"rows": len(rows),
            "patients": len({r["patient_id"] for r in rows}), "sha256": digest(path)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
