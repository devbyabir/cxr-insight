import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_insight.inference import (DISCLAIMER, LEGACY_INSTRUCTION_TEMPLATE, GenerationSettings,
                                   build_instruction, build_messages, generate_report,
                                   generate_text, parse_report)
from cxr_insight.models import LABELS, Prediction, ReportModel

HAS_PIL = importlib.util.find_spec("PIL") is not None

LEGACY_TARGET = ("user\n\nassistant\n\nTECHNIQUE: Chest PA and lateral\n\n"
                 "FINDINGS: The cardiac contours are normal. Multiple clips are again seen "
                 "projecting over the left breast.\n\nIMPRESSION: Nodule detected in the lungs.")


class Tensor(list):
    @property
    def shape(self):
        return (1, len(self[0]))


class Encoded(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt):
        self.calls.append(("template", messages, add_generation_prompt))
        return "<prompt>"

    def __call__(self, image, text, add_special_tokens, return_tensors):
        self.calls.append(("encode", image, text, add_special_tokens, return_tensors))
        return Encoded(input_ids=Tensor([[1, 2, 3]]))

    def decode(self, tokens, skip_special_tokens):
        self.calls.append(("decode", list(tokens), skip_special_tokens))
        return " REPORT "


class FakeVLM:
    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[1, 2, 3, 40, 41]]  # prompt tokens + 2 new tokens


def fake_report_model():
    return ReportModel(FakeVLM(), FakeProcessor(), "org/model", "rev1", None, device="cpu")


class PromptTests(unittest.TestCase):
    def test_legacy_template_shape(self):
        text = build_instruction("Nodule")
        self.assertTrue(text.startswith("\n" + " " * 18 + "You are an expert radiographer."))
        self.assertIn("the detected disease is Nodule.", text)
        self.assertTrue(text.endswith("IMPRESSION.\n" + " " * 14))
        self.assertIn("\n\n" + " " * 18 + "example\n\n", text)  # blank lines carry no indent
        self.assertEqual(text.count("\n"), 20)
        self.assertNotIn("{finding}", text)

    def test_every_label_renders(self):
        for label in LABELS:
            self.assertIn(f"disease is {label}.", build_instruction(label))

    def test_multiple_findings_joined(self):
        self.assertIn("disease is Mass, Edema.", build_instruction(["Mass", "Edema"]))

    def test_braces_in_template_are_safe(self):
        self.assertEqual(build_instruction("X", template="{a} {finding}"), "{a} X")

    def test_invalid_findings(self):
        for bad in ([], [""], ["  "], ["a\nb"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                build_instruction(bad)

    def test_messages_put_image_before_text(self):
        content = build_messages("hi")[0]["content"]
        self.assertEqual([c["type"] for c in content], ["image", "text"])
        self.assertEqual(content[1]["text"], "hi")

    def test_template_is_constant(self):
        self.assertEqual(LEGACY_INSTRUCTION_TEMPLATE.count("{finding}"), 1)


class ParseTests(unittest.TestCase):
    def test_strips_role_prefix_and_flags_example_leak(self):
        parsed = parse_report(LEGACY_TARGET)
        self.assertTrue(parsed.complete)
        self.assertEqual(parsed.sections["TECHNIQUE"], "Chest PA and lateral")
        self.assertEqual(parsed.sections["IMPRESSION"], "Nodule detected in the lungs.")
        self.assertEqual(parsed.leaked_example_phrases, ["multiple clips", "left breast"])

    def test_wrapped_lines_joined_case_insensitive(self):
        parsed = parse_report("technique: PA\nFindings: lungs\n  are clear.\nIMPRESSION:  Normal.")
        self.assertEqual(parsed.sections["FINDINGS"], "lungs are clear.")
        self.assertEqual(parsed.leaked_example_phrases, [])

    def test_missing_and_empty_sections(self):
        parsed = parse_report("FINDINGS: Clear.\nIMPRESSION:")
        self.assertFalse(parsed.complete)
        self.assertEqual(parsed.missing, ["TECHNIQUE", "IMPRESSION"])

    def test_header_must_start_line(self):
        parsed = parse_report("TECHNIQUE: PA. See FINDINGS: below")
        self.assertEqual(list(parsed.sections), ["TECHNIQUE"])

    def test_role_prefix_only_stripped_at_start(self):
        parsed = parse_report("user\nassistant\nTECHNIQUE: assistant-guided PA")
        self.assertEqual(parsed.sections["TECHNIQUE"], "assistant-guided PA")
        # A word that merely starts with "user" is not a role marker.
        self.assertEqual(parse_report("Usermode\nTECHNIQUE: PA").sections["TECHNIQUE"], "PA")

    def test_duplicate_header_keeps_first(self):
        parsed = parse_report("IMPRESSION: first\nIMPRESSION: second")
        self.assertEqual(parsed.sections["IMPRESSION"], "first")


class GenerationTests(unittest.TestCase):
    def test_settings_defaults_match_legacy(self):
        kwargs = GenerationSettings().generate_kwargs()
        self.assertEqual(kwargs, {"max_new_tokens": 512, "use_cache": True, "do_sample": True,
                                  "temperature": 0.5, "min_p": 0.1})

    def test_greedy_omits_sampling_params(self):
        kwargs = GenerationSettings(do_sample=False, temperature=0).generate_kwargs()
        self.assertNotIn("temperature", kwargs)

    def test_invalid_settings(self):
        for kw in ({"max_new_tokens": 0}, {"max_new_tokens": True}, {"temperature": 0},
                   {"min_p": 1}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                GenerationSettings(**kw)

    def test_generate_text_decodes_only_new_tokens(self):
        rm = fake_report_model()
        text = generate_text(rm, "IMG", "instr")
        self.assertEqual(text, "REPORT")
        calls = rm.processor.calls
        self.assertEqual(calls[0][2], True)  # add_generation_prompt
        self.assertEqual(calls[1], ("encode", "IMG", "<prompt>", False, "pt"))
        self.assertEqual(calls[2], ("decode", [40, 41], True))
        self.assertEqual(rm.model.kwargs["device"], "cpu")
        self.assertEqual(rm.model.kwargs["max_new_tokens"], 512)


class FakeClassifier:
    def predict(self, image):
        self.image = image
        return [Prediction("Effusion", 0.7), Prediction("Mass", 0.6), Prediction("No Finding", 0.1)]


@unittest.skipUnless(HAS_PIL, "Pillow required")
class PipelineTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.image_path = Path(tmp.name) / "cxr.png"
        Image.new("L", (64, 64), 200).save(self.image_path)

    def test_classifier_drives_prompt_and_result_serialises(self):
        clf = FakeClassifier()
        result = generate_report(self.image_path, fake_report_model(), classifier=clf, threshold=0.5)
        self.assertEqual(clf.image.mode, "RGB")
        self.assertEqual(result.findings, ["Effusion", "Mass"])
        self.assertIn("disease is Effusion, Mass.", result.instruction)
        self.assertEqual(result.raw_text, "REPORT")
        self.assertFalse(result.parsed.complete)
        data = result.to_dict()
        self.assertEqual((data["model_id"], data["revision"], data["disclaimer"]),
                         ("org/model", "rev1", DISCLAIMER))
        self.assertEqual(data["settings"]["temperature"], 0.5)
        json.dumps(data)

    def test_explicit_findings_without_classifier(self):
        result = generate_report(self.image_path, fake_report_model(), findings="Hernia")
        self.assertEqual((result.findings, result.predictions), (["Hernia"], []))

    def test_exactly_one_source_of_findings(self):
        with self.assertRaises(ValueError):
            generate_report(self.image_path, fake_report_model())
        with self.assertRaises(ValueError):
            generate_report(self.image_path, fake_report_model(), classifier=FakeClassifier(),
                            findings="Mass")

    def test_missing_image(self):
        with self.assertRaises(FileNotFoundError):
            generate_report(self.image_path.with_name("nope.png"), fake_report_model(),
                            findings="Mass")


if __name__ == "__main__":
    unittest.main()
