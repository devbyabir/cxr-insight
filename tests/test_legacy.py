import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_insight.inference import build_instruction
from cxr_insight.legacy import clean_legacy, main, read_legacy

LEAKY = ("user\n\nassistant\n\nTECHNIQUE: PA\n\nFINDINGS: Multiple clips over the left breast."
         "\n\nIMPRESSION: Mass.")
CLEAN = "TECHNIQUE: PA\nFINDINGS: Clear.\nIMPRESSION: Normal."


def row(path, finding, report):
    return {"img_path": path, "instructions": build_instruction(finding), "reports": report}


class LegacyTests(unittest.TestCase):
    def rows(self):
        a = row("/content/Dataset/Mass/1_2.png", "Mass", LEAKY)
        b = row("/content/Dataset/No Finding/3_4.png", "No Finding", CLEAN)
        return [a, b, dict(a), dict(a), dict(b)]

    def test_exact_duplicates_removed_and_counted(self):
        cleaned, audit = clean_legacy(self.rows())
        self.assertEqual((audit["input_rows"], audit["unique_rows"], audit["exact_duplicate_rows"]),
                         (5, 2, 3))
        self.assertEqual(len(cleaned), 2)

    def test_near_duplicates_kept(self):
        rows = self.rows()[:2] + [row("/content/Dataset/Mass/1_2.png", "Mass", CLEAN)]
        _, audit = clean_legacy(rows)
        self.assertEqual(audit["unique_rows"], 3)
        self.assertEqual(audit["unique_image_paths"], 2)

    def test_rows_labelled_and_artifacts_flagged(self):
        cleaned, audit = clean_legacy(self.rows())
        first = cleaned[0]
        self.assertEqual(first["image_path"], "Mass/1_2.png")
        self.assertEqual((first["folder"], first["prompt_label"]), ("Mass", "Mass"))
        self.assertTrue(first["report"].startswith("TECHNIQUE"))
        self.assertEqual(first["report_source"], "synthetic")
        self.assertEqual(first["example_leak"], "multiple clips;left breast")
        self.assertEqual(cleaned[1]["prompt_label"], "No Finding")
        self.assertEqual((audit["rows_with_role_prefix"], audit["rows_with_example_leak"],
                          audit["rows_with_all_sections"]), (1, 1, 2))
        self.assertFalse(audit["patient_ids_available"])
        self.assertEqual(audit["rows_where_prompt_label_differs_from_folder"], 0)

    def test_label_mismatch_counted(self):
        _, audit = clean_legacy([row("/content/Dataset/Mass/9_9.png", "Nodule", CLEAN)])
        self.assertEqual(audit["rows_where_prompt_label_differs_from_folder"], 1)

    def test_missing_finding_rejected(self):
        with self.assertRaises(ValueError):
            clean_legacy([{"img_path": "a/b.png", "instructions": "describe", "reports": CLEAN}])

    def test_cli_writes_outputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "legacy.csv"
            with src.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["img_path", "instructions", "reports"])
                writer.writeheader()
                writer.writerows(self.rows())
            self.assertEqual(len(read_legacy(src)), 5)
            out = Path(tmp) / "out"
            with redirect_stdout(io.StringIO()):
                main([str(src), "--output", str(out)])
            audit = json.loads((out / "legacy_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["exact_duplicate_rows"], 3)
            self.assertEqual(len(audit["input_sha256"]), 64)
            with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
                main([str(src), "--output", str(out)])

    def test_wrong_columns_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "bad.csv"
            src.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_legacy(src)


if __name__ == "__main__":
    unittest.main()
