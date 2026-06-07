import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.economic_preference_data import EconomicPreferenceConfig, build_economic_preference_jsonl, build_economic_preference_pairs
from training.strict_bar_backtest import load_market_bars


def _market(path: Path, n: int = 160) -> None:
    prices = []
    p = 100.0
    for _ in range(n):
        p *= 1.001
        prices.append(p)
    pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n, freq="5min"), "open": prices, "high": [x*1.001 for x in prices], "low": [x*0.999 for x in prices], "close": prices}).to_csv(path, index=False)


def _record(pos=10, side="LONG"):
    target = {"trend_side": side, "direction_stability": "TREND_STABLE", "reversal_pressure": "LOW", "risk_profile": "LOW_PATH_RISK", "horizons": {}, "summary_counts": {}}
    return {"task": "multi_horizon_path_shape_analyzer", "date": str(pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=5*pos)), "signal_pos": pos, "prompt": 'Header\nPast-only analyzer summary: {"regime":"UP"}', "target": json.dumps(target)}


class TestEconomicPreferenceData(unittest.TestCase):
    def test_build_pairs_prefers_profitable_trend_over_skip(self):
        with tempfile.TemporaryDirectory() as td:
            market_path = Path(td) / "m.csv"
            _market(market_path)
            market = load_market_bars(str(market_path))
            pairs = build_economic_preference_pairs([_record(10, "LONG")], market, EconomicPreferenceConfig(hold_bars_list=(6,), fee_rate=0, slippage_rate=0, leverage=1, mae_penalty=0, min_utility_gap=0.0001, max_pairs_per_row=2))
            self.assertGreaterEqual(len(pairs), 1)
            chosen = json.loads(pairs[0]["chosen"])
            self.assertEqual(chosen["gate"], "TRADE")
            self.assertEqual(chosen["side"], "LONG")
            self.assertFalse(pairs[0]["leakage_guard"]["prompt_uses_future_path"])
            self.assertTrue(pairs[0]["leakage_guard"]["preference_is_counterfactual_economic_not_path_classification"])

    def test_no_trade_utility_bias_prefers_skip_over_weak_trade(self):
        with tempfile.TemporaryDirectory() as td:
            market_path = Path(td) / "m.csv"
            _market(market_path)
            market = load_market_bars(str(market_path))
            pairs = build_economic_preference_pairs(
                [_record(10, "LONG")],
                market,
                EconomicPreferenceConfig(
                    hold_bars_list=(6,),
                    fee_rate=0,
                    slippage_rate=0,
                    leverage=1,
                    mae_penalty=0,
                    min_utility_gap=0.0001,
                    max_pairs_per_row=2,
                    no_trade_utility=0.02,
                ),
            )
            self.assertGreaterEqual(len(pairs), 1)
            chosen = json.loads(pairs[0]["chosen"])
            rejected = json.loads(pairs[0]["rejected"])
            self.assertEqual(chosen, {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0})
            self.assertEqual(rejected["gate"], "TRADE")
            self.assertEqual(pairs[0]["chosen_action"]["rank_utility"], 0.02)

    def test_cli_writes_preferences_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            market_path = root / "m.csv"
            records = root / "records.jsonl"
            out = root / "pref.jsonl"
            summary = root / "summary.json"
            _market(market_path)
            records.write_text(json.dumps(_record(10, "LONG")) + "\n")
            report = build_economic_preference_jsonl(records=str(records), market_csv=str(market_path), output=str(out), summary_output=str(summary), hold_bars_list="6", fee_rate=0, slippage_rate=0, leverage=1, mae_penalty=0, min_utility_gap=0.0001)
            self.assertTrue(out.exists())
            self.assertTrue(summary.exists())
            self.assertEqual(report["preferences"]["pairs"], len(out.read_text().splitlines()))
            self.assertTrue(report["preferences"]["leakage_guard"]["preferences_use_future_ohlc_paths"])


if __name__ == "__main__":
    unittest.main()
