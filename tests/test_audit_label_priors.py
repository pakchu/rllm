import json
import tempfile
import unittest
from pathlib import Path

from training.audit_label_priors import _resume_rows, _summarize_scores, parse_labels


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
        rows[0]["action"] = {"side": "LONG"}
        out = _summarize_scores(rows, ("X", "Y"), "mean")
        self.assertEqual(out["dominant_label"], "X")
        self.assertAlmostEqual(out["mean_score_spread"], 1.0)
        self.assertEqual(out["prediction_counts"], {"X": 2})
        self.assertTrue(out["target_metrics"]["has_targets"])
        self.assertAlmostEqual(out["target_metrics"]["accuracy"], 0.5)

    def test_resume_rows_loads_existing_score_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            path.write_text(json.dumps({"score_rows": [{"prediction": "X"}]}))
            self.assertEqual(_resume_rows(path), [{"prediction": "X"}])

    def test_resume_rows_rejects_missing_score_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            path.write_text(json.dumps({"score_rows": "omitted"}))
            with self.assertRaises(ValueError):
                _resume_rows(path)


if __name__ == "__main__":
    unittest.main()
