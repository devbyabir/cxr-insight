import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_insight.data import deduplicate, read_manifest, split_patients, validate_splits


class DataTests(unittest.TestCase):
    def rows(self):
        return [dict(image_path=f"{i}.png", patient_id=f"p{i // 2}",
                     sha256=str(i), report="FINDINGS: Example", report_source="synthetic")
                for i in range(40)]

    def test_patient_and_image_disjoint(self):
        result = split_patients(self.rows())
        validate_splits(result)
        self.assertEqual(sum(map(len, result.values())), 40)
        self.assertEqual({r["patient_id"] for r in result["train"]} &
                         {r["patient_id"] for r in result["test"]}, set())

    def test_reproducible_independent_of_input_order(self):
        first = split_patients(deduplicate(self.rows()))
        second = split_patients(deduplicate(list(reversed(self.rows()))))
        self.assertEqual(first, second)

    def test_duplicates_removed_and_conflicts_blocked(self):
        rows = self.rows()
        self.assertEqual(len(deduplicate(rows + [dict(rows[0])])), 40)
        for field in ("patient_id", "report", "report_source", "labels"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                deduplicate(rows + [dict(rows[0], **{field: "conflict"})])

    def test_insufficient_patients_rejected(self):
        with self.assertRaises(ValueError):
            split_patients(self.rows()[:4])

    def test_bad_fractions_rejected(self):
        with self.assertRaises(ValueError):
            split_patients(self.rows(), validation=.8, test=.4)

    def test_overlap_rejected(self):
        result = split_patients(self.rows())
        result["test"].append(result["train"][0])
        with self.assertRaises(ValueError):
            validate_splits(result)

    def test_manifest_hashes_bytes_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "image.png").write_bytes(b"test bytes")
            path = root / "manifest.csv"
            def write(image):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["image_path", "patient_id", "report", "report_source"])
                    writer.writeheader()
                    writer.writerow(dict(image_path=image, patient_id="p1", report="test", report_source="synthetic"))
            write("image.png")
            self.assertEqual(len(read_manifest(path, root)[0]["sha256"]), 64)
            write("../outside.png")
            with self.assertRaises(ValueError):
                read_manifest(path, root)


if __name__ == "__main__":
    unittest.main()
