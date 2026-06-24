import json
import tempfile
import unittest
from pathlib import Path

from training.fast_eval_ordinal_utility import FastEvalOrdinalUtilityConfig, _metrics, _prediction_rows, run


class TestFastEvalOrdinalUtility(unittest.TestCase):
    def test_metrics_include_rank_error(self):
        rows = [{"target": "AVOID"}, {"target": "LOW"}, {"target": "HIGH"}]
        m = _metrics(rows, ["LOW", "LOW", "MID"])
        self.assertAlmostEqual(m["accuracy"], 1 / 3)
        self.assertAlmostEqual(m["mean_abs_rank_error"], 2 / 3)
        self.assertEqual(m["prediction_counts"], {"LOW": 2, "MID": 1})

    def test_prediction_rows_keep_action_and_margin(self):
        rows = [{"date": "d", "signal_pos": 1, "target": "HIGH", "action": {"side": "LONG"}}]
        out = _prediction_rows(rows, ["MID"], [0.5])
        self.assertEqual(out[0]["high_minus_low_margin"], 0.5)
        self.assertEqual(out[0]["action"]["side"], "LONG")

    def test_target_echo_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "ord.jsonl"
            out = Path(tmp) / "report.json"
            pred = Path(tmp) / "pred.jsonl"
            inp.write_text(json.dumps({"date": "d", "signal_pos": 1, "prompt": "p", "target": "MID"}) + "\n")
            report = run(FastEvalOrdinalUtilityConfig(input_jsonl=str(inp), output=str(out), predictions_output=str(pred)))
            self.assertEqual(report["metrics"]["accuracy"], 1.0)
            self.assertTrue(out.exists())
            self.assertTrue(pred.exists())


if __name__ == "__main__":
    unittest.main()
