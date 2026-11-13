import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_insight.models import (LABELS, FindingClassifier, Prediction, load_report_model,
                                rank_predictions, select_findings)

HAS_IMAGING = all(importlib.util.find_spec(m) for m in ("numpy", "PIL"))


def scores(**named):
    values = [0.01] * len(LABELS)
    for label, score in named.items():
        values[LABELS.index(label)] = score
    return values


class FakeKeras:
    def __init__(self, output):
        self.output, self.batches = output, []

    def predict(self, batch, verbose=0):
        self.batches.append(batch)
        return [self.output]


class RankingTests(unittest.TestCase):
    def test_labels_match_legacy_checkpoint_order(self):
        self.assertEqual(len(LABELS), 15)
        self.assertEqual(LABELS.index("No Finding"), 10)
        self.assertEqual(list(LABELS), sorted(LABELS))

    def test_rank_sorted_descending(self):
        ranked = rank_predictions(scores(Mass=0.9, Edema=0.5))
        self.assertEqual([p.label for p in ranked[:2]], ["Mass", "Edema"])
        self.assertIsInstance(ranked[0].score, float)

    def test_rank_length_mismatch(self):
        with self.assertRaises(ValueError):
            rank_predictions([0.5] * 14)

    def test_argmax_by_default(self):
        ranked = rank_predictions(scores(Mass=0.6, Edema=0.55))
        self.assertEqual(select_findings(ranked), ["Mass"])

    def test_threshold_multilabel(self):
        ranked = rank_predictions(scores(Mass=0.6, Edema=0.55, Hernia=0.2))
        self.assertEqual(select_findings(ranked, threshold=0.5), ["Mass", "Edema"])

    def test_no_finding_not_mixed_with_positives(self):
        ranked = rank_predictions(scores(**{"No Finding": 0.9, "Effusion": 0.6}))
        self.assertEqual(select_findings(ranked, threshold=0.5), ["Effusion"])
        ranked = rank_predictions(scores(**{"No Finding": 0.9}))
        self.assertEqual(select_findings(ranked, threshold=0.5), ["No Finding"])

    def test_threshold_fallback_to_top(self):
        ranked = rank_predictions(scores(Mass=0.3))
        self.assertEqual(select_findings(ranked, threshold=0.5), ["Mass"])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            select_findings([])
        with self.assertRaises(ValueError):
            select_findings([Prediction("Mass", 0.5)], threshold=1.5)


@unittest.skipUnless(HAS_IMAGING, "numpy and Pillow required")
class ClassifierTests(unittest.TestCase):
    def test_preprocesses_like_legacy_notebook(self):
        from PIL import Image
        fake = FakeKeras(scores(Pneumonia=0.8))
        result = FindingClassifier(fake).predict(Image.new("L", (512, 400), 128))
        batch = fake.batches[0]
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(str(batch.dtype), "uint8")
        self.assertEqual(int(batch.max()), 128)  # no rescaling applied
        self.assertEqual(result[0].label, "Pneumonia")

    def test_missing_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            FindingClassifier.from_keras("does-not-exist.keras")


class ReportModelLoaderTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("unsloth"), "only meaningful without unsloth")
    def test_requires_unsloth(self):
        with self.assertRaises(ImportError):
            load_report_model("org/model")

    def test_adapter_dir_validated(self):
        fake = type(sys)("unsloth")
        fake.FastVisionModel = object
        with mock.patch.dict(sys.modules, {"unsloth": fake}),                 tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            load_report_model("org/model", adapter_path=tmp)

    def test_loads_adapter_without_attaching_fresh_lora(self):
        calls = []

        class FastVisionModel:
            @staticmethod
            def from_pretrained(source, **kw):
                calls.append(("from_pretrained", source, kw))
                return "model", "processor"

            @staticmethod
            def for_inference(model):
                calls.append(("for_inference", model))

            @staticmethod
            def get_peft_model(*a, **k):
                raise AssertionError("must not attach an untrained adapter")

        fake = type(sys)("unsloth")
        fake.FastVisionModel = FastVisionModel
        with mock.patch.dict(sys.modules, {"unsloth": fake}):
            loaded = load_report_model("org/model", revision="abc")
            self.assertEqual(calls[0], ("from_pretrained", "org/model",
                                        {"load_in_4bit": True, "revision": "abc"}))
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "adapter_config.json").write_text("{}")
                adapted = load_report_model("org/model", adapter_path=tmp, load_in_4bit=False)
                self.assertEqual(calls[2], ("from_pretrained", tmp, {"load_in_4bit": False}))
            self.assertEqual((loaded.model, loaded.processor, loaded.adapter_path),
                             ("model", "processor", None))
            self.assertEqual(adapted.adapter_path, tmp)


class FakeParam:
    def __init__(self, total):
        self.total = total

    def detach(self):
        return self

    def abs(self):
        return self

    def sum(self):
        return self.total


class FakeModel:
    def __init__(self, params):
        self.params = params

    def named_parameters(self):
        return iter(self.params)


class AdapterVerificationTests(unittest.TestCase):
    def test_trained_adapter_passes(self):
        from cxr_insight.models import verify_trained_adapter
        model = FakeModel([("q.lora_A.default.weight", FakeParam(3.0)),
                           ("q.lora_B.default.weight", FakeParam(0.2)),
                           ("v.lora_B.default.weight", FakeParam(0.0))])
        self.assertEqual(verify_trained_adapter(model),
                         {"lora_A_tensors": 1, "lora_B_tensors": 2, "nonzero_lora_B_tensors": 1})

    def test_fresh_or_missing_adapter_rejected(self):
        from cxr_insight.models import verify_trained_adapter
        fresh = FakeModel([("q.lora_A.default.weight", FakeParam(3.0)),
                           ("q.lora_B.default.weight", FakeParam(0.0))])
        with self.assertRaisesRegex(RuntimeError, "untrained"):
            verify_trained_adapter(fresh)
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            verify_trained_adapter(FakeModel([("dense.weight", FakeParam(1.0))]))


if __name__ == "__main__":
    unittest.main()
