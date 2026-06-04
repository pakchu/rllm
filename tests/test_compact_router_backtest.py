import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.compact_router_backtest import (
    CompactRouterExecutionConfig,
    hold_for_policy,
    route_compact_record,
    run_backtest,
    simulate_compact_router_records,
)
from training.strict_bar_backtest import load_market_bars


def _market(path: Path, n: int = 220) -> None:
    prices = []
    p = 100.0
    for _ in range(n):
        p *= 1.001
        prices.append(p)
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="5min"),
            "open": prices,
            "high": [x * 1.001 for x in prices],
            "low": [x * 0.999 for x in prices],
            "close": prices,
        }
    ).to_csv(path, index=False)


def _payload(**overrides):
    obj = {
        "trend_side": "LONG",
        "action_path": "TREND",
        "horizon_bars": 72,
        "horizon_policy": "SHORT_STEP",
        "edge_quality": "STRONG",
        "risk_budget": "SMALL",
        "score_bucket": "HIGH",
        "direction_stability": "TREND_STABLE",
        "reversal_pressure": "LOW",
    }
    obj.update(overrides)
    return json.dumps(obj, separators=(",", ":"))


def _record(ts: str, **overrides):
    return {"date": ts, "prediction": _payload(**overrides), "target": _payload(action_path="FADE")}


class TestCompactRouterBacktest(unittest.TestCase):
    def test_route_uses_learned_fields_without_action_path(self):
        cfg = CompactRouterExecutionConfig(routing_mode="learned_fields")
        rec = _record("2025-01-01", action_path="FADE", trend_side="LONG")
        self.assertEqual(route_compact_record(rec, cfg), ("LONG", cfg.short_hold_bars, "trend_side_horizon_edge"))
        weak = _record("2025-01-01", edge_quality="WEAK")
        self.assertEqual(route_compact_record(weak, cfg)[0], "NONE")

    def test_route_action_path_mode_can_fade_trend(self):
        cfg = CompactRouterExecutionConfig(routing_mode="action_path")
        rec = _record("2025-01-01", action_path="FADE", trend_side="LONG")
        self.assertEqual(route_compact_record(rec, cfg), ("SHORT", cfg.short_hold_bars, "action_path_fade"))

    def test_hold_for_policy_maps_step_buckets(self):
        cfg = CompactRouterExecutionConfig(short_hold_bars=36, mid_hold_bars=144, long_hold_bars=432)
        self.assertEqual(hold_for_policy("SHORT_STEP", cfg), 36)
        self.assertEqual(hold_for_policy("MID_STEP", cfg), 144)
        self.assertEqual(hold_for_policy("LONG_STEP", cfg), 432)
        self.assertEqual(hold_for_policy("SKIP_STEP", cfg), 0)

    def test_simulation_uses_strict_ohlc_and_model_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            market_path = Path(td) / "m.csv"
            _market(market_path)
            market = load_market_bars(str(market_path))
            records = [_record(str(ts), trend_side="LONG") for ts in pd.date_range("2025-01-01 01:00", periods=4, freq="60min")]
            out = simulate_compact_router_records(records, market, CompactRouterExecutionConfig(short_hold_bars=6, cooldown_bars=0))
            self.assertGreater(out["sim"]["trade_entries"], 0)
            self.assertEqual(out["sim"]["return_application"], "actual_ohlc_bar_by_bar_strict_mdd_compact_router")
            self.assertIn("ci95_mean_trade_ret_pct", out["trade_stats"])

    def test_cli_runner_writes_model_prediction_leakage_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            market_path = root / "m.csv"
            records_path = root / "r.jsonl"
            out_path = root / "out.json"
            _market(market_path)
            rows = [_record(str(ts), trend_side="LONG") for ts in pd.date_range("2025-01-01 01:00", periods=3, freq="60min")]
            records_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            args = argparse.Namespace(
                records=str(records_path),
                market_csv=str(market_path),
                output=str(out_path),
                start_date="",
                end_date="",
                train_start="",
                train_end="",
                val_start="",
                val_end="",
                oos_start="",
                oos_end="",
                cooldown_bars=0,
                entry_delay_bars=1,
                leverage=0.5,
                fee_rate=0.0004,
                slippage_rate=0.0001,
                short_hold_bars=6,
                mid_hold_bars=12,
                long_hold_bars=24,
                min_edge_quality="STRONG",
                routing_mode="learned_fields",
                use_target=False,
            )
            out = run_backtest(args)
            self.assertTrue(out["leakage_guard"]["uses_model_predictions_when_use_target_false"])
            self.assertFalse(out["leakage_guard"]["target_mode_is_oracle_only"])
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
