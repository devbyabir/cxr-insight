import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_insight.report_eval import bleu4, compare, main, mentioned_findings, score_case

REF = "TECHNIQUE: PA.\nFINDINGS: Small left pleural effusion. No pneumothorax.\nIMPRESSION: Effusion."


class LabellerTests(unittest.TestCase):
    def test_negation_respected(self):
        self.assertEqual(mentioned_findings(REF), {"Effusion"})
        self.assertEqual(mentioned_findings("There is no evidence of pneumothorax or effusion."), set())
        self.assertEqual(mentioned_findings("Mild cardiomegaly; lungs without consolidation"),
                         {"Cardiomegaly"})

    def test_word_boundaries(self):
        self.assertEqual(mentioned_findings("Massive improvement"), {"Mass"})  # documented crudeness
        self.assertEqual(mentioned_findings("The masses are stable"), {"Mass"})
        self.assertEqual(mentioned_findings("Amass"), set())


class ScoringTests(unittest.TestCase):
    def test_bleu_bounds(self):
        self.assertAlmostEqual(bleu4("a b c d e", "a b c d e"), 1.0)
        self.assertLess(bleu4("a b c d e", "x y z"), 0.3)
        self.assertEqual(bleu4("a b", ""), 0.0)

    def test_hallucination_and_omission(self):
        case = score_case(REF, "TECHNIQUE: PA\nFINDINGS: Pneumothorax.\nIMPRESSION: Pneumothorax.")
        self.assertEqual((case["hallucinated"], case["omitted"]), (["Pneumothorax"], ["Effusion"]))
        self.assertTrue(case["complete_sections"])
        self.assertFalse(case["finding_set_match"])

    def test_compare_requires_same_cases(self):
        a = {"1": {"id": "1", "reference": REF, "generated": REF}}
        with self.assertRaises(ValueError):
            compare({"baseline": a, "tuned": {}})
        with self.assertRaises(ValueError):
            compare({"baseline": a, "tuned": {"1": {"id": "1", "reference": "x", "generated": REF}}})

    def test_cli_compares_systems(self):
        leak = "FINDINGS: Multiple clips over the left breast. Pneumothorax."
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name, gen in (("baseline", leak), ("tuned", REF)):
                paths[name] = Path(tmp) / f"{name}.jsonl"
                paths[name].write_text(json.dumps({"id": "c1", "reference": REF, "generated": gen}) + "\n")
            out = Path(tmp) / "cmp.json"
            with redirect_stdout(io.StringIO()):
                main([f"baseline={paths['baseline']}", f"tuned={paths['tuned']}", "--out", str(out)])
            result = json.loads(out.read_text())
            self.assertEqual(result["systems"]["baseline"]["example_leak_rate"], 1.0)
            self.assertEqual(result["systems"]["tuned"]["finding_set_exact_match_rate"], 1.0)
            self.assertEqual(result["failures"]["baseline"][0]["hallucinated"], ["Pneumothorax"])
            self.assertEqual(result["failures"]["tuned"], [])
            self.assertIn("not a validated clinical labeller", result["note"])


if __name__ == "__main__":
    unittest.main()
