import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class MetricsTests(unittest.TestCase):
    def setUp(self):
        from cxr_insight import metrics
        self.m = metrics

    def test_auroc_known_values(self):
        self.assertAlmostEqual(self.m.auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertEqual(self.m.auroc([0, 1], [0.2, 0.9]), 1.0)
        self.assertEqual(self.m.auroc([0, 1], [0.9, 0.2]), 0.0)
        self.assertEqual(self.m.auroc([0, 1], [0.5, 0.5]), 0.5)  # ties count half

    def test_auroc_undefined_without_both_classes(self):
        self.assertIsNone(self.m.auroc([1, 1], [0.2, 0.3]))
        self.assertIsNone(self.m.auroc([0, 0], [0.2, 0.3]))

    def test_auroc_matches_pairwise_definition(self):
        import numpy as np
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        s = np.round(rng.random(200), 2)  # rounding creates ties
        pos, neg = s[y == 1], s[y == 0]
        expected = ((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg))
        self.assertAlmostEqual(self.m.auroc(y, s), expected)

    def test_roc_curve_endpoints(self):
        fpr, tpr = self.m.roc_curve([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.3])
        self.assertEqual((fpr[0], tpr[0], fpr[-1], tpr[-1]), (0.0, 0.0, 1.0, 1.0))

    def test_report_counts_and_aggregates(self):
        y = [[1, 0], [1, 1], [0, 0], [0, 1]]
        s = [[0.9, 0.2], [0.4, 0.8], [0.6, 0.1], [0.1, 0.3]]
        report = self.m.classification_report(y, s, ["A", "B"], [0.5, 0.5])
        a, b = report["per_class"]
        self.assertEqual((a["tp"], a["fp"], a["fn"], a["tn"]), (1, 1, 1, 1))
        self.assertEqual((b["tp"], b["fp"], b["fn"], b["tn"]), (1, 0, 1, 2))
        self.assertAlmostEqual(a["f1"], 0.5)
        self.assertAlmostEqual(b["precision"], 1.0)
        self.assertAlmostEqual(report["micro_precision"], 2 / 3)
        self.assertAlmostEqual(report["micro_recall"], 0.5)
        self.assertEqual(report["auroc_defined_for"], 2)

    def test_undefined_class_excluded_from_macro(self):
        report = self.m.classification_report([[1, 0], [0, 0]], [[0.9, 0.1], [0.1, 0.2]],
                                              ["A", "B"], [0.5, 0.5])
        self.assertIsNone(report["per_class"][1]["auroc"])
        self.assertEqual(report["macro_auroc"], 1.0)
        self.assertEqual(report["macro_f1_supported"], 1.0)

    def test_thresholds_fit_f1(self):
        t = self.m.choose_thresholds([[1], [1], [0], [0]], [[0.8], [0.6], [0.55], [0.1]])
        self.assertEqual(t, [0.6])
        self.assertEqual(self.m.choose_thresholds([[0], [0]], [[0.3], [0.4]]), [0.5])

    def test_shape_and_label_validation(self):
        with self.assertRaises(ValueError):
            self.m.auroc([0, 1, 1], [0.1, 0.2])
        with self.assertRaises(ValueError):
            self.m.auroc([0, 2], [0.1, 0.2])
        with self.assertRaises(ValueError):
            self.m.classification_report([[1, 0]], [[0.5, 0.5]], ["A"], [0.5])

    def test_bootstrap_resamples_patients(self):
        y = [[0], [1], [0], [1], [0], [1]]
        s = [[0.1], [0.9], [0.2], [0.8], [0.3], [0.7]]
        ci = self.m.bootstrap_macro_auroc(y, s, ["p1", "p1", "p2", "p2", "p3", "p3"], 200, seed=1)
        self.assertEqual((ci["low"], ci["high"], ci["unit"]), (1.0, 1.0, "patient"))


if __name__ == "__main__":
    unittest.main()
