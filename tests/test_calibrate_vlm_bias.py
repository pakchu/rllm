import json
import tempfile
import unittest
from pathlib import Path

from training.calibrate_vlm_bias import (
    _frange_inclusive,
    _bias_grid_from_legacy_args,
    _candidate_constraint_report,
    _parse_label_value_specs,
    calibrate_action_biases,
    load_action_scores,
    score_metrics,
)
from training.eval_vlm_policy import summarize_action_metrics, select_action_from_scores


class TestCalibrateVlmBias(unittest.TestCase):
    def test_frange_inclusive(self):
        vals = _frange_inclusive(-0.2, 0.2, 0.1)
        self.assertEqual(vals, [-0.2, -0.1, 0.0, 0.1, 0.2])

    def test_load_action_scores(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rep.json"
            p.write_text(
                json.dumps(
                    {
                        "action_scores": [
                            {"target": "BUY", "scores": {"BUY": 0.2, "HOLD": 0.1, "SELL": 0.0}}
                        ]
                    }
                )
            )
            rows = load_action_scores([str(p)])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["target"], "BUY")

    def test_calibrate_action_biases_improves_objective(self):
        rows = [
            {"target": "BUY", "scores": {"BUY": 0.0, "HOLD": 0.1, "SELL": 0.2}},
            {"target": "BUY", "scores": {"BUY": 0.0, "HOLD": 0.2, "SELL": 0.3}},
            {"target": "SELL", "scores": {"BUY": 0.2, "HOLD": 0.1, "SELL": 0.0}},
            {"target": "SELL", "scores": {"BUY": 0.3, "HOLD": 0.2, "SELL": 0.0}},
            {"target": "HOLD", "scores": {"BUY": 0.0, "HOLD": 0.2, "SELL": 0.1}},
            {"target": "HOLD", "scores": {"BUY": 0.1, "HOLD": 0.3, "SELL": 0.2}},
        ]
        report = calibrate_action_biases(
            rows=rows,
            buy_min=-0.4,
            buy_max=0.4,
            buy_step=0.2,
            hold_min=-0.4,
            hold_max=0.4,
            hold_step=0.2,
            sell_min=-0.4,
            sell_max=0.4,
            sell_step=0.2,
            top_k=3,
        )
        best = report["best"]
        preds0 = []
        targets = []
        for r in rows:
            p0, _ = select_action_from_scores(r["scores"], action_biases={"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0})
            preds0.append(p0)
            targets.append(r["target"])
        m0 = summarize_action_metrics(targets=targets, predictions=preds0)
        baseline_obj = score_metrics(m0)
        self.assertGreaterEqual(best["objective"], baseline_obj)
        self.assertEqual(len(report["top_candidates"]), 3)

    def test_calibrate_action_biases_trade_gate(self):
        rows = [
            {"target": "TRADE", "scores": {"TRADE": 0.0, "NO_TRADE": 0.2}},
            {"target": "TRADE", "scores": {"TRADE": 0.1, "NO_TRADE": 0.2}},
            {"target": "NO_TRADE", "scores": {"TRADE": 0.2, "NO_TRADE": 0.1}},
            {"target": "NO_TRADE", "scores": {"TRADE": 0.3, "NO_TRADE": 0.2}},
        ]
        report = calibrate_action_biases(
            rows=rows,
            labels=("TRADE", "NO_TRADE"),
            bias_grid={
                "TRADE": (-0.4, 0.4, 0.2),
                "NO_TRADE": (-0.4, 0.4, 0.2),
            },
            top_k=2,
        )
        self.assertEqual(report["grid"]["labels"], ["TRADE", "NO_TRADE"])
        self.assertEqual(len(report["top_candidates"]), 2)
        self.assertIn("TRADE", report["best"]["biases"])
        self.assertIn("NO_TRADE", report["best"]["biases"])

    def test_bias_grid_from_legacy_args_trade_side(self):
        grid = _bias_grid_from_legacy_args(
            ("LONG", "SHORT"),
            buy_min=-0.8,
            buy_max=0.8,
            buy_step=0.1,
            hold_min=-0.6,
            hold_max=0.6,
            hold_step=0.1,
            sell_min=-0.8,
            sell_max=0.8,
            sell_step=0.1,
            trade_min=-1.5,
            trade_max=1.5,
            trade_step=0.1,
            no_trade_min=-1.5,
            no_trade_max=1.5,
            no_trade_step=0.1,
            long_min=-1.2,
            long_max=1.2,
            long_step=0.2,
            short_min=-0.9,
            short_max=0.9,
            short_step=0.3,
        )
        self.assertEqual(grid["LONG"], (-1.2, 1.2, 0.2))
        self.assertEqual(grid["SHORT"], (-0.9, 0.9, 0.3))

    def test_parse_label_value_specs(self):
        parsed = _parse_label_value_specs(["TRADE=0.25", "NO_TRADE=0.4"], name="x")
        self.assertEqual(parsed, {"TRADE": 0.25, "NO_TRADE": 0.4})

    def test_candidate_constraint_report(self):
        metrics = {
            "num_samples": 100,
            "pred_counts": {"TRADE": 70, "NO_TRADE": 30},
            "per_class": {
                "TRADE": {"recall": 0.6},
                "NO_TRADE": {"recall": 0.4},
            },
        }
        rep = _candidate_constraint_report(
            metrics,
            min_recall_by_label={"TRADE": 0.5, "NO_TRADE": 0.3},
            min_pred_frac_by_label={"TRADE": 0.2},
            max_pred_frac_by_label={"TRADE": 0.8},
        )
        self.assertTrue(rep["feasible"])
        rep_fail = _candidate_constraint_report(
            metrics,
            min_recall_by_label={"NO_TRADE": 0.5},
        )
        self.assertFalse(rep_fail["feasible"])

    def test_calibrate_action_biases_with_constraints(self):
        rows = [
            {"target": "TRADE", "scores": {"TRADE": 0.0, "NO_TRADE": 0.2}},
            {"target": "TRADE", "scores": {"TRADE": 0.1, "NO_TRADE": 0.2}},
            {"target": "NO_TRADE", "scores": {"TRADE": 0.2, "NO_TRADE": 0.1}},
            {"target": "NO_TRADE", "scores": {"TRADE": 0.3, "NO_TRADE": 0.2}},
        ]
        report = calibrate_action_biases(
            rows=rows,
            labels=("TRADE", "NO_TRADE"),
            bias_grid={
                "TRADE": (-0.4, 0.4, 0.2),
                "NO_TRADE": (-0.4, 0.4, 0.2),
            },
            min_pred_frac_by_label={"TRADE": 0.25},
            max_pred_frac_by_label={"TRADE": 0.75},
            top_k=2,
        )
        self.assertIn("constraints", report)
        self.assertIn("feasible_count", report["constraints"])


if __name__ == "__main__":
    unittest.main()
