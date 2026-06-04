import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.fade_warning_filter_backtest import (
    FadeWarningFilterConfig,
    route_fade_warning_record,
    run_sweep,
    run_backtest,
    simulate_fade_warning_filter_records,
)
from training.strict_bar_backtest import load_market_bars


def _market(path: Path, n: int = 240) -> None:
    prices = []
    p = 100.0
    for _ in range(n):
        p *= 1.001
        prices.append(p)
    pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n, freq="5min"), "open": prices, "high": [x*1.001 for x in prices], "low": [x*0.999 for x in prices], "close": prices}).to_csv(path, index=False)


def _payload(fade="NO_FADE_WARNING", side="LONG"):
    return json.dumps({"trend_side": side, "fade_warning": fade, "skip_reason": "NO_EDGE", "trend_continuation_quality": "CONTINUE_STRONG"})


def _record(ts: str, fade="NO_FADE_WARNING", side="LONG"):
    return {"date": ts, "prediction": _payload(fade, side), "target": _payload("FADE_STRONG", side)}


class TestFadeWarningFilterBacktest(unittest.TestCase):
    def test_route_filters_fade_warning_without_opening_trade(self):
        cfg = FadeWarningFilterConfig(skip_fade_warnings=("FADE_STRONG",))
        self.assertEqual(route_fade_warning_record(_record("2025-01-01", "NO_FADE_WARNING", "LONG"), cfg), ("LONG", "trend_allowed"))
        self.assertEqual(route_fade_warning_record(_record("2025-01-01", "FADE_STRONG", "LONG"), cfg), ("NONE", "skip_FADE_STRONG"))
        flip = FadeWarningFilterConfig(skip_fade_warnings=(), flip_fade_strong=True)
        self.assertEqual(route_fade_warning_record(_record("2025-01-01", "FADE_STRONG", "LONG"), flip), ("SHORT", "flip_fade_strong_diagnostic"))

    def test_simulation_uses_model_prediction_and_strict_path(self):
        with tempfile.TemporaryDirectory() as td:
            market_path = Path(td) / "m.csv"
            _market(market_path)
            market = load_market_bars(str(market_path))
            records = [_record(str(ts), "NO_FADE_WARNING", "LONG") for ts in pd.date_range("2025-01-01 01:00", periods=4, freq="60min")]
            out = simulate_fade_warning_filter_records(records, market, FadeWarningFilterConfig(hold_bars=6, cooldown_bars=0))
            self.assertGreater(out["sim"]["trade_entries"], 0)
            self.assertEqual(out["sim"]["return_application"], "actual_ohlc_bar_by_bar_strict_mdd_fade_warning_filter")
            self.assertIn("ci95_mean_trade_ret_pct", out["trade_stats"])

    def test_backtest_cli_runner_writes_leakage_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            market = root / "m.csv"
            recs = root / "r.jsonl"
            out = root / "out.json"
            _market(market)
            rows = [_record(str(ts), "NO_FADE_WARNING", "LONG") for ts in pd.date_range("2025-01-01 01:00", periods=3, freq="60min")]
            recs.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            args = argparse.Namespace(records=str(recs), market_csv=str(market), output=str(out), start_date="", end_date="", train_start="", train_end="", val_start="", val_end="", oos_start="", oos_end="", hold_bars=6, cooldown_bars=0, entry_delay_bars=1, leverage=0.5, fee_rate=0.0004, slippage_rate=0.0001, skip_fade_warnings="FADE_STRONG", use_target=False, flip_fade_strong=False)
            report = run_backtest(args)
            self.assertTrue(report["leakage_guard"]["uses_model_predictions_when_use_target_false"])
            self.assertTrue(report["leakage_guard"]["fade_warning_is_filter_not_entry_policy"])
            self.assertTrue(out.exists())

    def test_sweep_selects_on_val_and_reports_oos(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            market = root / "m.csv"
            val = root / "val.jsonl"
            oos = root / "oos.jsonl"
            out = root / "sweep.json"
            _market(market)
            rows = [_record(str(ts), "NO_FADE_WARNING", "LONG") for ts in pd.date_range("2025-01-01 01:00", periods=4, freq="60min")]
            val.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            oos.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            args = argparse.Namespace(val_records=str(val), oos_records=str(oos), market_csv=str(market), output=str(out), hold_bars_list="6,12", cooldown_bars_list="0", skip_fade_warning_sets="FADE_STRONG;", include_flip_diagnostic=False, entry_delay_bars=1, leverage=0.5, fee_rate=0.0004, slippage_rate=0.0001, min_trades=1, max_mdd=25.0, top_k=2)
            report = run_sweep(args)
            self.assertEqual(report["num_candidates"], 4)
            self.assertTrue(report["leakage_guard"]["selected_on_val_only"])
            self.assertTrue(report["leakage_guard"]["oos_not_used_for_selection"])
            self.assertIn("selected_oos", report)


if __name__ == "__main__":
    unittest.main()
