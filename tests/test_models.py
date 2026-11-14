import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_insight.models import load_report_model


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
