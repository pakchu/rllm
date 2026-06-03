import json
import tempfile
import unittest
from pathlib import Path

from training.eval_compact_path_shape_analyzer import evaluate_compact_path_shape_analyzer, parse_compact_path_shape_json


def _row(action="TREND", horizon=36, risk="SMALL"):
    target = {
        "trend_side": "LONG",
        "action_path": action,
        "horizon_bars": horizon,
        "horizon_policy": "SHORT_STEP" if horizon in {36, 72} else "SKIP_STEP" if horizon == 0 else "LONG_STEP",
        "edge_quality": "STRONG" if action != "NONE" else "NO_EDGE",
        "risk_budget": risk,
        "score_bucket": "HIGH" if action != "NONE" else "NEGATIVE_OR_TOO_WEAK",
        "direction_stability": "TREND_STABLE",
        "reversal_pressure": "LOW",
    }
    return {"task": "compact_path_shape_analyzer", "prompt": "Past-only analyzer summary: {}", "target": json.dumps(target)}


class TestEvalCompactPathShapeAnalyzer(unittest.TestCase):
    def test_parser_normalizes_bad_values_to_safe_defaults(self):
        parsed = parse_compact_path_shape_json('{"action_path":"bad","horizon_bars":999,"risk_budget":"bad"}')
        self.assertEqual(parsed["action_path"], "NONE")
        self.assertEqual(parsed["horizon_bars"], "0")
        self.assertEqual(parsed["horizon_policy"], "SKIP_STEP")
        self.assertEqual(parsed["risk_budget"], "AVOID_OR_TINY")

    def test_target_echo_eval_reports_exact_metrics_and_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "eval.jsonl"
            out = root / "report.json"
            preds = root / "preds.jsonl"
            rows = [_row("TREND", 36, "SMALL"), _row("NONE", 0, "AVOID_OR_TINY")]
            data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            report = evaluate_compact_path_shape_analyzer(
                eval_jsonl=str(data),
                output=str(out),
                prediction_output=str(preds),
                prediction_mode="target_echo",
            )
            self.assertEqual(report["metrics"]["num_samples"], 2)
            self.assertEqual(report["metrics"]["exact_all_keys_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["exact_primary_keys_accuracy"], 1.0)
            self.assertTrue(report["leakage_guard"]["target_echo_is_oracle_only"])
            self.assertEqual(len(preds.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
