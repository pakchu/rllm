import json
import tempfile
import unittest
from pathlib import Path

from training.compact_path_shape_analyzer_data import (
    CompactPathShapeConfig,
    build_compact_record,
    derive_compact_path_shape_target,
    main,
    summarize_compact_records,
)


def _source_target(trend_bucket="STRONG_POSITIVE", fade_bucket="NEGATIVE", trend_mae="LOW", fade_mae="HIGH"):
    return {
        "trend_side": "LONG",
        "direction_stability": "TREND_STABLE",
        "reversal_pressure": "LOW",
        "risk_profile": "LOW_PATH_RISK",
        "summary_counts": {"trend_wins": 2, "fade_wins": 0, "mixed": 0, "none": 0},
        "horizons": {
            "36": {
                "trend_return_bucket": trend_bucket,
                "fade_return_bucket": fade_bucket,
                "trend_mae_bucket": trend_mae,
                "fade_mae_bucket": fade_mae,
                "relative_edge": "TREND_STRONGER",
                "best_path": "TREND",
                "tradable_path_count": 1,
            },
            "72": {
                "trend_return_bucket": "POSITIVE",
                "fade_return_bucket": "NEGATIVE",
                "trend_mae_bucket": "MEDIUM",
                "fade_mae_bucket": "HIGH",
                "relative_edge": "TREND_SLIGHTLY_STRONGER",
                "best_path": "TREND",
                "tradable_path_count": 1,
            },
        },
    }


def _source_record():
    return {
        "task": "multi_horizon_path_shape_analyzer",
        "date": "2025-01-01 00:00:00",
        "signal_pos": 10,
        "prompt": "header\nPast-only analyzer summary: {\"regime\":\"UP\",\"momentum\":1}",
        "target": json.dumps(_source_target()),
    }


class TestCompactPathShapeAnalyzerData(unittest.TestCase):
    def test_derives_compact_trade_horizon_state(self):
        cfg = CompactPathShapeConfig(hold_bars_list=(36, 72), min_trade_score=1.25)
        target = derive_compact_path_shape_target(json.dumps(_source_target()), cfg)
        self.assertEqual(target["trend_side"], "LONG")
        self.assertEqual(target["action_path"], "TREND")
        self.assertEqual(target["horizon_bars"], 36)
        self.assertEqual(target["horizon_policy"], "SHORT_STEP")
        self.assertEqual(target["edge_quality"], "STRONG")
        self.assertEqual(target["risk_budget"], "AGGRESSIVE_OK")

    def test_weak_paths_become_skip_state(self):
        cfg = CompactPathShapeConfig(hold_bars_list=(36, 72), min_trade_score=1.25)
        weak = _source_target(trend_bucket="WEAK_POSITIVE", fade_bucket="FLAT_NEGATIVE", trend_mae="HIGH", fade_mae="HIGH")
        weak["horizons"]["72"]["trend_return_bucket"] = "WEAK_POSITIVE"
        weak["horizons"]["72"]["trend_mae_bucket"] = "HIGH"
        target = derive_compact_path_shape_target(json.dumps(weak), cfg)
        self.assertEqual(target["action_path"], "NONE")
        self.assertEqual(target["horizon_bars"], 0)
        self.assertEqual(target["horizon_policy"], "SKIP_STEP")
        self.assertEqual(target["risk_budget"], "AVOID_OR_TINY")

    def test_build_record_is_past_only_and_shorter_than_source_target(self):
        cfg = CompactPathShapeConfig(hold_bars_list=(36, 72))
        rec = build_compact_record(_source_record(), cfg)
        self.assertEqual(rec["task"], "compact_path_shape_analyzer")
        self.assertIn("Past-only analyzer summary", rec["prompt"])
        self.assertFalse(rec["leakage_guard"]["prompt_uses_future_path"])
        self.assertTrue(rec["leakage_guard"]["target_is_compressed_from_future_path_shape_label"])
        self.assertLess(len(rec["target"]), len(_source_record()["target"]))
        summary = summarize_compact_records([rec])
        self.assertEqual(summary["num_records"], 1)
        self.assertEqual(summary["action_path"]["TREND"], 1)

    def test_cli_writes_compact_records_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "source.jsonl"
            out = root / "compact.jsonl"
            summary = root / "summary.json"
            src.write_text(json.dumps(_source_record()) + "\n")
            import sys

            old = sys.argv
            try:
                sys.argv = [
                    "prog",
                    "--records",
                    str(src),
                    "--output",
                    str(out),
                    "--summary-output",
                    str(summary),
                    "--hold-bars-list",
                    "36,72",
                ]
                main()
            finally:
                sys.argv = old
            self.assertEqual(len(out.read_text().splitlines()), 1)
            payload = json.loads(summary.read_text())
            self.assertEqual(payload["records"]["num_records"], 1)
            self.assertTrue(payload["records"]["leakage_guard"]["targets_are_router_states_not_final_orders"])


if __name__ == "__main__":
    unittest.main()
