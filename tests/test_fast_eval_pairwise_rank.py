import json
import tempfile
import unittest
from pathlib import Path

from training.fast_eval_pairwise_rank import FastEvalPairwiseRankConfig, _metrics, _prediction_rows, run


class TestFastEvalPairwiseRank(unittest.TestCase):
    def test_metrics_reports_ab_accuracy_and_confusion(self):
        rows = [{"target": "A"}, {"target": "B"}, {"target": "B"}]
        m = _metrics(rows, ["A", "A", "B"])
        self.assertAlmostEqual(m["accuracy"], 2 / 3)
        self.assertEqual(m["target_counts"], {"A": 1, "B": 2})
        self.assertEqual(m["prediction_counts"], {"A": 2, "B": 1})
        self.assertEqual(m["confusion"]["target=B|pred=A"], 1)

    def test_prediction_rows_keep_actions_for_tournament_debugging(self):
        rows = [{"date": "d", "signal_pos": 1, "target": "B", "chosen_action": {"side": "LONG"}, "rejected_action": {"side": "SHORT"}}]
        out = _prediction_rows(rows, ["A"], [0.25])
        self.assertEqual(out[0]["prediction"], "A")
        self.assertEqual(out[0]["target"], "B")
        self.assertEqual(out[0]["margin_a_minus_b"], 0.25)
        self.assertEqual(out[0]["chosen_action"]["side"], "LONG")

    def test_target_echo_cli_writes_report_and_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "pairs.jsonl"
            out = Path(tmp) / "report.json"
            pred = Path(tmp) / "pred.jsonl"
            inp.write_text(json.dumps({"date": "d", "signal_pos": 1, "prompt": "p", "target": "B"}) + "\n")
            report = run(FastEvalPairwiseRankConfig(input_jsonl=str(inp), output=str(out), predictions_output=str(pred)))
            self.assertEqual(report["metrics"]["accuracy"], 1.0)
            self.assertTrue(out.exists())
            self.assertTrue(pred.exists())
            self.assertTrue(report["leakage_guard"]["target_echo_is_oracle_only"])


if __name__ == "__main__":
    unittest.main()
