import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.edge_decay_router_backtest import (
    EdgeRouterExecutionConfig,
    route_record,
    run_backtest,
    simulate_router_records,
)
from training.strict_bar_backtest import load_market_bars


def _market(path: Path, n: int = 140) -> None:
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


def _record(ts: str, hint: str, side: str = "LONG") -> dict:
    return {
        "date": ts,
        "target": json.dumps(
            {
                "recommended_router_hint": hint,
                "trend_side": side,
                "edge_decay_label": "EDGE_PERSIST",
                "transition_label": "TREND_CONTINUATION",
                "risk_label": "LOW_ADVERSE_EXCURSION",
            },
            separators=(",", ":"),
        ),
    }


class TestEdgeDecayRouterBacktest(unittest.TestCase):
    def test_route_record_maps_hints_without_gate_labels(self):
        cfg = EdgeRouterExecutionConfig()
        self.assertEqual(route_record(_record("2025-01-01", "ALLOW_TREND_SPECIALIST", "LONG"), cfg), "LONG")
        self.assertEqual(route_record(_record("2025-01-01", "CONSIDER_REVERSAL_SPECIALIST", "LONG"), cfg), "SHORT")
        self.assertEqual(route_record(_record("2025-01-01", "REDUCE_OR_SKIP_TREND_SPECIALIST", "LONG"), cfg), "NONE")

    def test_simulate_router_records_uses_strict_ohlc_path(self):
        with tempfile.TemporaryDirectory() as td:
            market_path = Path(td) / "m.csv"
            _market(market_path)
            market = load_market_bars(str(market_path))
            records = [_record(str(ts), "ALLOW_TREND_SPECIALIST", "LONG") for ts in pd.date_range("2025-01-01 01:00", periods=4, freq="60min")]
            out = simulate_router_records(records, market, EdgeRouterExecutionConfig(hold_bars=6, cooldown_bars=0, leverage=0.5))
            self.assertGreater(out["sim"]["trade_entries"], 0)
            self.assertEqual(out["sim"]["return_application"], "actual_ohlc_bar_by_bar_strict_mdd_edge_router")
            self.assertIn("ci95_mean_trade_ret_pct", out["trade_stats"])

    def test_cli_runner_writes_oracle_leakage_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            market_path = root / "m.csv"
            records_path = root / "r.jsonl"
            out_path = root / "out.json"
            _market(market_path)
            rows = [_record(str(ts), "ALLOW_TREND_SPECIALIST", "LONG") for ts in pd.date_range("2025-01-01 01:00", periods=3, freq="60min")]
            records_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            import argparse

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
                hold_bars=6,
                cooldown_bars=0,
                entry_delay_bars=1,
                leverage=0.5,
                fee_rate=0.0004,
                slippage_rate=0.0001,
                allow_hints="ALLOW_TREND_SPECIALIST",
                reversal_hints="CONSIDER_REVERSAL_SPECIALIST",
                skip_hints="REDUCE_OR_SKIP_TREND_SPECIALIST,RANGE_ROUTER_ONLY,LOW_CONFIDENCE_ROUTER",
            )
            out = run_backtest(args)
            self.assertTrue(out["leakage_guard"]["oracle_targets_may_use_future_path"])
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
