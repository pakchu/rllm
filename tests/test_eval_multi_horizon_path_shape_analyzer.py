import json
import tempfile
import unittest
from pathlib import Path

from training.eval_multi_horizon_path_shape_analyzer import evaluate_path_shape_analyzer, parse_path_shape_json


def _target(stability: str = "TREND_STABLE") -> dict:
    return {
        "trend_side": "LONG",
        "direction_stability": stability,
        "reversal_pressure": "LOW",
        "risk_profile": "LOW_PATH_RISK",
        "horizons": {
            "36": {
                "trend_return_bucket": "POSITIVE",
                "fade_return_bucket": "NEGATIVE",
                "trend_mae_bucket": "LOW",
                "fade_mae_bucket": "MEDIUM",
                "relative_edge": "TREND_SLIGHTLY_STRONGER",
                "best_path": "TREND",
                "tradable_path_count": 1,
            },
            "72": {
                "trend_return_bucket": "STRONG_POSITIVE",
                "fade_return_bucket": "STRONG_NEGATIVE",
                "trend_mae_bucket": "LOW",
                "fade_mae_bucket": "HIGH",
                "relative_edge": "TREND_STRONGER",
                "best_path": "TREND",
                "tradable_path_count": 1,
            },
        },
        "summary_counts": {"trend_wins": 2, "fade_wins": 0, "mixed": 0, "none": 0},
    }


def _row(stability: str = "TREND_STABLE") -> dict:
    return {
        "task": "multi_horizon_path_shape_analyzer",
        "date": "2025-01-01",
        "prompt": "past only",
        "target": json.dumps(_target(stability), separators=(",", ":")),
    }


class TestEvalMultiHorizonPathShapeAnalyzer(unittest.TestCase):
    def test_parse_path_shape_json_defaults_invalid_nested_values(self):
        parsed = parse_path_shape_json('{"trend_side":"bad","horizons":{"36":{"best_path":"trend","tradable_path_count":9}}}', horizons=(36, 72))
        self.assertEqual(parsed["trend_side"], "NONE")
        self.assertEqual(parsed["direction_stability"], "NO_STABLE_EDGE")
        self.assertEqual(parsed["horizons"]["36"]["best_path"], "TREND")
        self.assertEqual(parsed["horizons"]["36"]["tradable_path_count"], 2)
        self.assertEqual(parsed["horizons"]["72"]["best_path"], "NONE")

    def test_target_echo_writes_predictions_and_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "shape.jsonl"
            out = root / "eval.json"
            pred = root / "pred.jsonl"
            rows = [_row("TREND_STABLE"), _row("FADE_STABLE")]
            src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            report = evaluate_path_shape_analyzer(
                eval_jsonl=str(src),
                output=str(out),
                prediction_output=str(pred),
                prediction_mode="target_echo",
                hold_bars_list="36,72",
            )
            self.assertEqual(report["metrics"]["exact_all_keys_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["exact_top_level_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["horizon_key_micro_accuracy"]["best_path"], 1.0)
            lines = [json.loads(x) for x in pred.read_text().splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertIn("prediction", lines[0])
            self.assertTrue(report["leakage_guard"]["target_echo_is_oracle_only"])


if __name__ == "__main__":
    unittest.main()
