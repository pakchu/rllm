import unittest

from training.audit_label_priors import _summarize_scores, parse_labels


class TestAuditLabelPriors(unittest.TestCase):
    def test_parse_labels_requires_unique_multiple_labels(self):
        self.assertEqual(parse_labels("A,B,C"), ("A", "B", "C"))
        with self.assertRaises(ValueError):
            parse_labels("A")
        with self.assertRaises(ValueError):
            parse_labels("A,B,A")

    def test_summarize_scores_reports_dominant_label_and_spread(self):
        rows = [
            {"target": "X", "prediction": "X", "score": {"X": {"mean": -1.0}, "Y": {"mean": -2.0}}},
            {"target": "Y", "prediction": "X", "score": {"X": {"mean": -1.5}, "Y": {"mean": -2.5}}},
        ]
        out = _summarize_scores(rows, ("X", "Y"), "mean")
        self.assertEqual(out["dominant_label"], "X")
        self.assertAlmostEqual(out["mean_score_spread"], 1.0)
        self.assertEqual(out["prediction_counts"], {"X": 2})
        self.assertTrue(out["target_metrics"]["has_targets"])
        self.assertAlmostEqual(out["target_metrics"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
