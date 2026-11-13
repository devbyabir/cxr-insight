import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_insight.nih import NIH_LABELS, build_manifest, main, parse_labels, split, to_multi_hot


class LabelTests(unittest.TestCase):
    def test_fourteen_labels_without_no_finding(self):
        self.assertEqual(len(NIH_LABELS), 14)
        self.assertNotIn("No Finding", NIH_LABELS)

    def test_multi_label_parsing(self):
        self.assertEqual(parse_labels("Mass|Effusion"), ("Effusion", "Mass"))
        self.assertEqual(parse_labels("No Finding"), ())
        self.assertEqual(parse_labels("Mass|Mass"), ("Mass",))

    def test_invalid_labels(self):
        for bad in ("", "Tumour", "No Finding|Mass"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_labels(bad)

    def test_multi_hot(self):
        vec = to_multi_hot(("Effusion", "Mass"))
        self.assertEqual(sum(vec), 2)
        self.assertEqual(vec[NIH_LABELS.index("Mass")], 1)
        self.assertEqual(to_multi_hot(()), [0] * 14)


class ManifestSplitTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        images = self.root / "images"
        images.mkdir()
        entries, test = [], []
        for patient in range(1, 21):
            for visit in range(3):
                name = f"{patient:08d}_{visit:03d}.png"
                if not (patient == 20 and visit == 2):  # one metadata row has no image
                    (images / name).write_bytes(name.encode())
                labels = "No Finding" if visit == 0 else "Effusion|Mass" if visit == 1 else "Hernia"
                entries.append({"Image Index": name, "Finding Labels": labels,
                                "Patient ID": str(patient), "View Position": "PA"})
                if patient > 16:
                    test.append(name)
        self.data_entry = self.root / "Data_Entry.csv"
        with self.data_entry.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
        self.test_list = self.root / "test_list.txt"
        self.test_list.write_text("\n".join(test), encoding="utf-8")

    def test_manifest_skips_missing_images(self):
        rows, missing = build_manifest(self.data_entry, self.root / "images", self.test_list)
        self.assertEqual((len(rows), missing), (59, 1))
        self.assertEqual(rows[1]["labels"], "Effusion|Mass")
        self.assertEqual(rows[0]["labels"], "")
        self.assertEqual(rows[0]["patient_id"], "1")

    def test_official_test_frozen_and_patients_disjoint(self):
        rows, _ = build_manifest(self.data_entry, self.root / "images", self.test_list)
        parts = split(rows, seed=1)
        self.assertEqual({r["patient_id"] for r in parts["test"]}, {"17", "18", "19", "20"})
        ids = [{r["patient_id"] for r in parts[n]} for n in ("train", "validation", "test")]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(sum(map(len, parts.values())), 59)
        # Deterministic regardless of input row order.
        self.assertEqual(split(rows, seed=1), split(list(reversed(rows)), seed=1))

    def test_cli_summary_marks_subset(self):
        out = self.root / "splits"
        with redirect_stdout(io.StringIO()):
            main(["--data-entry", str(self.data_entry), "--test-list", str(self.test_list),
                  "--image-dir", str(self.root / "images"), "--output", str(out)])
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["subset"])
        self.assertEqual(summary["splits"]["test"]["label_counts"]["Mass"], 4)
        self.assertEqual(summary["splits"]["test"]["label_counts"]["No Finding"], 4)


if __name__ == "__main__":
    unittest.main()
