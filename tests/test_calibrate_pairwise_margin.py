import json
import tempfile
import unittest
from pathlib import Path

from training.calibrate_pairwise_margin import PairwiseMarginCalibrationConfig, choose_threshold, run


class TestCalibratePairwiseMargin(unittest.TestCase):
    def test_choose_threshold_can_remove_a_prior(self):
        rows = [
            {"target": "A", "margin_a_minus_b": 0.8},
            {"target": "A", "margin_a_minus_b": 0.7},
            {"target": "B", "margin_a_minus_b": 0.6},
            {"target": "B", "margin_a_minus_b": 0.5},
        ]
        t = choose_threshold(rows)
        self.assertGreater(t, 0.5)
        self.assertLess(t, 0.7)

    def test_run_writes_eval_predictions_without_using_eval_for_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal = Path(tmp) / "cal.jsonl"
            ev = Path(tmp) / "eval.jsonl"
            out = Path(tmp) / "report.json"
            pred = Path(tmp) / "pred.jsonl"
            rows = [
                {"target": "A", "margin_a_minus_b": 0.8},
                {"target": "B", "margin_a_minus_b": 0.4},
            ]
            cal.write_text("".join(json.dumps(r) + "\n" for r in rows))
            ev.write_text("".join(json.dumps(r) + "\n" for r in rows))
            report = run(PairwiseMarginCalibrationConfig(str(cal), str(ev), str(out), str(pred)))
            self.assertTrue(report["leakage_guard"]["threshold_uses_eval_scores"] is False)
            self.assertEqual(report["eval_metrics"]["accuracy"], 1.0)
            self.assertTrue(out.exists())
            self.assertTrue(pred.exists())


if __name__ == "__main__":
    unittest.main()
