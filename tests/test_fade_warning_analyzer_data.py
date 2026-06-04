import json
import tempfile
import unittest
from pathlib import Path

from training.fade_warning_analyzer_data import (
    FadeWarningConfig,
    build_fade_warning_record,
    derive_fade_warning_target,
    main,
    summarize_fade_warning_records,
)


def _target(fade_ret="STRONG_POSITIVE", trend_ret="NEGATIVE"):
    return {
        "trend_side": "LONG",
        "direction_stability": "HORIZON_CONFLICT",
        "reversal_pressure": "HIGH",
        "risk_profile": "LOW_PATH_RISK",
        "summary_counts": {},
        "horizons": {
            "36": {"trend_return_bucket": trend_ret, "fade_return_bucket": fade_ret, "trend_mae_bucket": "HIGH", "fade_mae_bucket": "LOW", "relative_edge": "FADE_STRONGER", "best_path": "FADE", "tradable_path_count": 1},
            "72": {"trend_return_bucket": "NEGATIVE", "fade_return_bucket": "POSITIVE", "trend_mae_bucket": "HIGH", "fade_mae_bucket": "MEDIUM", "relative_edge": "FADE_STRONGER", "best_path": "FADE", "tradable_path_count": 1},
        },
    }


def _record():
    return {"task": "multi_horizon_path_shape_analyzer", "date": "2025-01-01", "signal_pos": 1, "prompt": 'x\nPast-only analyzer summary: {"regime":"REVERSAL"}', "target": json.dumps(_target())}


class TestFadeWarningAnalyzerData(unittest.TestCase):
    def test_derives_narrow_fade_warning_target(self):
        cfg = FadeWarningConfig(hold_bars_list=(36, 72))
        target = derive_fade_warning_target(json.dumps(_target()), cfg)
        self.assertEqual(target["fade_warning"], "FADE_STRONG")
        self.assertEqual(target["skip_reason"], "TRADEABLE_FADE")
        self.assertIn("trend_continuation_quality", target)
        self.assertNotIn("primary_route", target)

    def test_build_record_is_past_only(self):
        cfg = FadeWarningConfig(hold_bars_list=(36, 72))
        rec = build_fade_warning_record(_record(), cfg)
        self.assertEqual(rec["task"], "fade_warning_analyzer")
        self.assertIn("Do not output an order", rec["prompt"])
        self.assertFalse(rec["leakage_guard"]["prompt_uses_future_path"])
        summary = summarize_fade_warning_records([rec])
        self.assertEqual(summary["fade_warning"]["FADE_STRONG"], 1)

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
