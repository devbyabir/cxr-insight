"""Torch-dependent tests; skipped when torch/torchvision are not installed."""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAS_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "torchvision", "numpy", "PIL"))


def write_split_dir(root, n_patients=6):
    from PIL import Image
    images = root / "images"
    images.mkdir()
    splits = root / "splits"
    splits.mkdir()
    labels = ["", "Effusion|Mass", "Hernia", "Mass"]
    rows = {"train": [], "validation": [], "test": []}
    for p in range(n_patients):
        part = "train" if p < n_patients - 2 else "validation" if p == n_patients - 2 else "test"
        for v in range(4):
            name = f"{p:08d}_{v:03d}.png"
            Image.new("L", (64, 64), 40 * v + p).save(images / name)
            rows[part].append({"image_path": name, "patient_id": str(p), "labels": labels[v]})
    for name, part in rows.items():
        (splits / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
    (splits / "summary.json").write_text(json.dumps(
        {"dataset": "synthetic test", "subset": True, "images_used": 4 * n_patients,
         "source": "test", "splits": {}}))
    return images, splits


@unittest.skipUnless(HAS_TORCH, "torch/torchvision required")
class CnnTests(unittest.TestCase):
    def setUp(self):
        from cxr_insight import cnn
        self.cnn = cnn

    def test_augmentation_only_for_train_and_never_flips(self):
        from torchvision import transforms
        train = [type(t) for t in self.cnn.transform_for("train").transforms]
        evaluation = [type(t) for t in self.cnn.transform_for("test").transforms]
        self.assertIn(transforms.RandomResizedCrop, train)
        self.assertNotIn(transforms.RandomHorizontalFlip, train)
        self.assertNotIn(transforms.RandomVerticalFlip, train)
        self.assertEqual(evaluation, [transforms.Resize, transforms.ToTensor, transforms.Normalize])
        self.assertEqual([type(t) for t in self.cnn.transform_for("validation").transforms], evaluation)
        with self.assertRaises(ValueError):
            self.cnn.transform_for("training")

    def test_eval_transform_is_deterministic_and_handles_grayscale(self):
        import torch
        from PIL import Image
        image = Image.new("L", (300, 200), 100)
        a = self.cnn.eval_transform()(self.cnn.load_rgb(self._save(image)))
        b = self.cnn.eval_transform()(image.convert("RGB"))
        self.assertEqual(tuple(a.shape), (3, 224, 224))
        self.assertTrue(torch.equal(a, b))

    def _save(self, image):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        image.save(tmp.name)
        self.addCleanup(Path(tmp.name).unlink)
        return tmp.name

    def test_model_outputs_fourteen_logits(self):
        import torch
        model = self.cnn.build_model(pretrained=False).eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(2, 3, 224, 224)).shape), (2, 14))

    def test_dataset_targets_are_multi_hot(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, splits = write_split_dir(Path(tmp))
            rows = self.cnn.read_split(splits / "train.jsonl")
            dataset = self.cnn.CXRDataset(rows, images, "validation")
            self.assertEqual(dataset[0][1].sum().item(), 0)  # No Finding
            self.assertEqual(dataset[1][1].sum().item(), 2)  # Effusion|Mass

    def test_checkpoint_round_trip_and_grad_cam(self):
        from PIL import Image
        model = self.cnn.build_model(pretrained=False).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.pt"
            self.cnn.save_classifier(path, model, [0.3] * 14, {"note": "test"})
            loaded, checkpoint = self.cnn.load_classifier(path)
            image = Image.new("L", (128, 128), 90)
            self.assertEqual(self.cnn.predict_probabilities(model, image),
                             self.cnn.predict_probabilities(loaded, image))
            self.assertEqual(checkpoint["thresholds"], [0.3] * 14)
            cam = self.cnn.grad_cam(loaded, image, 3)
            self.assertEqual(cam.shape, (7, 7))
            self.assertTrue(0 <= cam.min() and cam.max() <= 1)

    def test_roc_plot_handles_undefined_macro_auroc(self):
        import numpy as np
        from cxr_insight.evaluate_cnn import plot
        from cxr_insight.metrics import classification_report
        from cxr_insight.nih import NIH_LABELS
        y = np.zeros((4, 14), dtype=int)  # no positives: every AUROC undefined
        s = np.full((4, 14), 0.2)
        report = classification_report(y, s, NIH_LABELS, [0.5] * 14)
        self.assertIsNone(report["macro_auroc"])
        with tempfile.TemporaryDirectory() as tmp:
            plot(report, y, s, Path(tmp))
            self.assertTrue((Path(tmp) / "roc.png").exists())

    def test_rejects_foreign_checkpoint(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pt"
            torch.save({"architecture": "resnet50", "labels": []}, path)
            with self.assertRaises(ValueError):
                self.cnn.load_classifier(path)

    def test_training_resumes_and_evaluation_runs(self):
        from cxr_insight import evaluate_cnn, train_cnn
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, splits = write_split_dir(root)
            out = root / "run"
            base = ["--splits", str(splits), "--image-dir", str(images), "--out", str(out),
                    "--batch-size", "4", "--workers", "0", "--no-pretrained"]
            with redirect_stdout(io.StringIO()):
                train_cnn.main(base + ["--epochs", "1"])
            self.assertTrue((out / "classifier.pt").exists())
            # Simulate an interrupted 2-epoch run: a fresh dir, one epoch, then resume.
            out2 = root / "run2"
            base2 = [a if a != str(out) else str(out2) for a in base]
            log = io.StringIO()
            with redirect_stdout(log):
                train_cnn.main(base2 + ["--epochs", "2", "--session-epochs", "1"])
            self.assertIn("Paused after epoch 1/2", log.getvalue())
            self.assertFalse((out2 / "classifier.pt").exists())
            log = io.StringIO()
            with redirect_stdout(log):
                train_cnn.main(base2 + ["--epochs", "2"])
            self.assertIn("Resumed from epoch 1", log.getvalue())
            meta = json.loads((out2 / "run_metadata.json").read_text())
            self.assertEqual([h["epoch"] for h in meta["history"]], [1, 2])
            self.assertEqual(meta["thresholds_fitted_on"], "validation")
            with redirect_stdout(io.StringIO()):
                evaluate_cnn.main(["--checkpoint", str(out / "classifier.pt"), "--splits", str(splits),
                                   "--image-dir", str(images), "--out", str(root / "eval"),
                                   "--bootstrap", "20", "--workers", "0"])
            metrics = json.loads((root / "eval" / "test_metrics.json").read_text())
            self.assertEqual(metrics["n_images"], 4)
            self.assertTrue(metrics["data"]["subset"])
            self.assertTrue((root / "eval" / "roc.png").exists())


if __name__ == "__main__":
    unittest.main()
