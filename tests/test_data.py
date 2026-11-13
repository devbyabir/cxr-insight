import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_insight.data import deduplicate, main, read_manifest, split_patients, validate_splits


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


FIELDS = ["image_path", "patient_id", "report", "report_source"]


class ManifestFixture(unittest.TestCase):
    """Temp image root + manifest writer; holds no tests itself."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.manifest = self.root / "manifest.csv"

    def image(self, name, content=None):
        (self.root / name).write_bytes(content if content is not None else name.encode())
        return name

    def write(self, rows, fields=FIELDS, encoding="utf-8"):
        with self.manifest.open("w", newline="", encoding=encoding) as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)

    def row(self, i, patient=None, source="synthetic"):
        return [self.image(f"{i}.png"), patient or f"p{i}", "FINDINGS: x", source]

    def assertRejected(self, message):
        with self.assertRaisesRegex(ValueError, message):
            read_manifest(self.manifest, self.root)


class ManifestTests(ManifestFixture):
    def test_valid_rows_normalised(self):
        self.write([[f" {self.image('a.png')} ", " p1 ", " text ", "clinician"]])
        row = read_manifest(self.manifest, self.root)[0]
        self.assertEqual((row["image_path"], row["patient_id"], row["report"]), ("a.png", "p1", "text"))

    def test_utf8_bom_header_accepted(self):
        self.write([self.row(1)], encoding="utf-8-sig")
        self.assertEqual(len(read_manifest(self.manifest, self.root)), 1)

    def test_missing_column(self):
        self.write([["a.png", "p1", "text"]], fields=FIELDS[:3])
        self.assertRejected("Required columns")

    def test_empty_manifest(self):
        self.write([])
        self.assertRejected("empty")

    def test_blank_required_value(self):
        self.write([[self.image("a.png"), "  ", "text", "synthetic"]])
        self.assertRejected("Missing required value at row 2")

    def test_invalid_report_source(self):
        self.write([self.row(1), self.row(2, source="radiologist")])
        self.assertRejected("Invalid report_source at row 3")

    def test_missing_image(self):
        self.write([["absent.png", "p1", "text", "synthetic"]])
        self.assertRejected("Image missing")

    def test_extra_field_is_malformed(self):
        self.write([self.row(1) + ["extra"]])
        self.assertRejected("Malformed CSV")

    def test_absolute_path_outside_root_rejected(self):
        with tempfile.TemporaryDirectory() as other:
            outside = Path(other) / "x.png"
            outside.write_bytes(b"x")
            self.write([[str(outside), "p1", "text", "synthetic"]])
            self.assertRejected("escapes")

    def test_identical_bytes_share_hash(self):
        self.image("a.png", b"same"); self.image("b.png", b"same")
        self.write([["a.png", "p1", "t", "synthetic"], ["b.png", "p1", "t", "synthetic"]])
        rows = read_manifest(self.manifest, self.root)
        self.assertEqual(rows[0]["sha256"], rows[1]["sha256"])
        self.assertEqual(len(deduplicate(rows)), 1)


class CliTests(ManifestFixture):
    def run_main(self, output):
        argv = ["cxr-data", str(self.manifest), "--image-root", str(self.root),
                "--output", str(output), "--seed", "7"]
        with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            main()

    def test_writes_splits_and_summary(self):
        rows = [self.row(i, patient=f"p{i // 2}") for i in range(20)]
        rows.append(list(rows[0]))  # exact duplicate row
        self.write(rows)
        out = self.root / "splits"
        self.run_main(out)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["seed"], summary["input_rows"], summary["unique_images"],
                          summary["removed_duplicates"]), (7, 21, 20, 1))
        loaded = {name: [json.loads(line) for line in
                         (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
                  for name in ("train", "validation", "test")}
        validate_splits(loaded)
        self.assertEqual(sum(map(len, loaded.values())), 20)
        for name, rows_ in loaded.items():
            self.assertEqual(summary["splits"][name]["rows"], len(rows_))

    def test_refuses_to_overwrite_existing_output(self):
        self.write([self.row(i) for i in range(10)])
        out = self.root / "splits"
        out.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_main(out)
        self.assertEqual(list(out.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
