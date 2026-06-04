import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.sweep_compact_router_backtest import run_sweep


def _market(path: Path, n: int = 260) -> None:
    prices = []
    p = 100.0
    for _ in range(n):
        p *= 1.001
        prices.append(p)
    pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n, freq="5min"), "open": prices, "high": [x*1.001 for x in prices], "low": [x*0.999 for x in prices], "close": prices}).to_csv(path, index=False)


def _record(ts: str, side="LONG"):
    payload = {"trend_side": side, "action_path": "TREND", "horizon_bars": 36, "horizon_policy": "SHORT_STEP", "edge_quality": "STRONG", "risk_budget": "SMALL", "score_bucket": "HIGH", "direction_stability": "TREND_STABLE", "reversal_pressure": "LOW"}
    return {"date": ts, "prediction": json.dumps(payload), "target": json.dumps(payload)}


class TestSweepCompactRouterBacktest(unittest.TestCase):
    def test_sweep_selects_on_val_and_reports_oos(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            market = root / "m.csv"
            val = root / "val.jsonl"
            oos = root / "oos.jsonl"
            out = root / "out.json"
            _market(market)
            rows = [_record(str(ts)) for ts in pd.date_range("2025-01-01 01:00", periods=4, freq="60min")]
            val.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            oos.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            args = argparse.Namespace(
                val_records=str(val), oos_records=str(oos), market_csv=str(market), output=str(out),
                routing_modes="learned_fields", min_edge_qualities="STRONG", cooldown_bars_list="0,12",
                short_hold_bars_list="6", mid_hold_bars_list="12", long_hold_bars_list="24",
                entry_delay_bars=1, leverage=0.5, fee_rate=0.0004, slippage_rate=0.0001,
                min_trades=1, max_mdd=25.0, top_k=2,
            )
            report = run_sweep(args)
            self.assertEqual(report["num_candidates"], 2)
            self.assertTrue(report["leakage_guard"]["selected_on_val_only"])
            self.assertTrue(report["leakage_guard"]["oos_not_used_for_selection"])
            self.assertIn("selected_oos", report)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
