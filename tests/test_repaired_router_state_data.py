import json
import tempfile
import unittest
from pathlib import Path

from training.repaired_router_state_data import (
    RepairedRouterStateConfig,
    build_repaired_record,
    derive_repaired_router_state_target,
    main,
    summarize_repaired_records,
)


def _target(trend_ret="STRONG_POSITIVE", fade_ret="NEGATIVE", trend_mae="LOW", fade_mae="HIGH", stability="TREND_STABLE"):
    return {
        "trend_side": "LONG",
        "direction_stability": stability,
        "reversal_pressure": "LOW",
        "risk_profile": "LOW_PATH_RISK",
        "summary_counts": {},
        "horizons": {
            "36": {"trend_return_bucket": trend_ret, "fade_return_bucket": fade_ret, "trend_mae_bucket": trend_mae, "fade_mae_bucket": fade_mae, "relative_edge": "TREND_STRONGER", "best_path": "TREND", "tradable_path_count": 1},
            "72": {"trend_return_bucket": "POSITIVE", "fade_return_bucket": "NEGATIVE", "trend_mae_bucket": "MEDIUM", "fade_mae_bucket": "HIGH", "relative_edge": "TREND_SLIGHTLY_STRONGER", "best_path": "TREND", "tradable_path_count": 1},
        },
    }


def _record(target=None):
    return {"task": "multi_horizon_path_shape_analyzer", "date": "2025-01-01", "signal_pos": 1, "prompt": 'x\nPast-only analyzer summary: {"regime":"UP"}', "target": json.dumps(target or _target())}


class TestRepairedRouterStateData(unittest.TestCase):
    def test_derives_tradeable_trend_target(self):
        cfg = RepairedRouterStateConfig(hold_bars_list=(36, 72))
        target = derive_repaired_router_state_target(json.dumps(_target()), cfg)
        self.assertEqual(target["trend_continuation_quality"], "CONTINUE_STRONG")
        self.assertEqual(target["fade_warning"], "NO_FADE_WARNING")
        self.assertEqual(target["skip_reason"], "TRADEABLE_TREND")
        self.assertEqual(target["primary_route"], "TREND")
        self.assertEqual(target["horizon_policy"], "SHORT_STEP")

    def test_derives_fade_warning_without_forcing_risk_budget(self):
        cfg = RepairedRouterStateConfig(hold_bars_list=(36, 72))
        raw = _target(trend_ret="NEGATIVE", fade_ret="STRONG_POSITIVE", trend_mae="HIGH", fade_mae="LOW", stability="HORIZON_CONFLICT")
        target = derive_repaired_router_state_target(json.dumps(raw), cfg)
        self.assertEqual(target["fade_warning"], "FADE_STRONG")
        self.assertEqual(target["primary_route"], "FADE")
        self.assertNotIn("risk_budget", target)

    def test_build_record_and_summary_keep_leakage_flags(self):
        cfg = RepairedRouterStateConfig(hold_bars_list=(36, 72))
        rec = build_repaired_record(_record(), cfg)
        self.assertEqual(rec["task"], "repaired_router_state_analyzer")
        self.assertFalse(rec["leakage_guard"]["prompt_uses_future_path"])
        self.assertTrue(rec["leakage_guard"]["risk_budget_removed_due_to_class_collapse"])
        summary = summarize_repaired_records([rec])
        self.assertEqual(summary["primary_route"]["TREND"], 1)

    def test_cli_writes_records(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src.jsonl"
            out = root / "out.jsonl"
            summary = root / "summary.json"
            src.write_text(json.dumps(_record()) + "\n")
            import sys
            old = sys.argv
            try:
                sys.argv = ["prog", "--records", str(src), "--output", str(out), "--summary-output", str(summary), "--hold-bars-list", "36,72"]
                main()
            finally:
                sys.argv = old
            self.assertEqual(len(out.read_text().splitlines()), 1)
            self.assertEqual(json.loads(summary.read_text())["records"]["num_records"], 1)


if __name__ == "__main__":
    unittest.main()
