import unittest

import pandas as pd

from training.path_outcome_dataset import (
    PathOutcomeConfig,
    build_path_outcome_records,
    compute_trade_path_outcome,
    make_path_outcome_record,
    summarize_records,
)


def _market(opens, highs=None, lows=None):
    highs = highs or opens
    lows = lows or opens
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(opens), freq="5min"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": opens,
        }
    )


class TestPathOutcomeDataset(unittest.TestCase):
    def test_long_and_short_outcomes_use_delayed_entry_path(self):
        market = _market([100, 100, 103, 104], highs=[100, 101, 105, 104], lows=[100, 99, 102, 104])
        cfg = PathOutcomeConfig(hold_bars=2, entry_delay_bars=1, fee_rate=0.0, slippage_rate=0.0, leverage=1.0)
        long = compute_trade_path_outcome(market, 0, "LONG", cfg)
        short = compute_trade_path_outcome(market, 0, "SHORT", cfg)
        self.assertIsNotNone(long)
        self.assertIsNotNone(short)
        assert long is not None and short is not None
        self.assertEqual(long.entry_pos, 1)
        self.assertEqual(long.exit_pos, 3)
        self.assertAlmostEqual(long.gross_return, 0.04, places=8)
        self.assertAlmostEqual(long.mae, 0.01, places=8)
        self.assertAlmostEqual(long.mfe, 0.05, places=8)
        self.assertAlmostEqual(short.gross_return, -0.04, places=8)
        self.assertAlmostEqual(short.mae, 0.05, places=8)

    def test_record_labels_trade_gate_and_side_from_path_utility(self):
        market = _market([100, 100, 103, 104], highs=[100, 101, 105, 104], lows=[100, 99, 102, 104])
        cfg = PathOutcomeConfig(hold_bars=2, entry_delay_bars=1, fee_rate=0.0, slippage_rate=0.0, mae_penalty=0.0)
        rec = make_path_outcome_record(market, 0, cfg)
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["trade_gate_label"], "TRADE")
        self.assertEqual(rec["trade_side_label"], "LONG")
        self.assertGreater(rec["best_net_return"], 0.0)
        self.assertIn(rec["quality_label"], {"weak", "positive", "strong"})
        self.assertIn(rec["risk_label"], {"low", "medium", "high", "extreme"})
        self.assertIn("LONG", rec["outcome_summary"])

    def test_max_mae_can_suppress_trade_gate_without_changing_side_label(self):
        market = _market([100, 100, 103, 104], highs=[100, 101, 105, 104], lows=[100, 80, 102, 104])
        cfg = PathOutcomeConfig(hold_bars=2, entry_delay_bars=1, fee_rate=0.0, slippage_rate=0.0, mae_penalty=0.0, max_mae=0.05)
        rec = make_path_outcome_record(market, 0, cfg)
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["trade_side_label"], "LONG")
        self.assertEqual(rec["trade_gate_label"], "NO_TRADE")

    def test_build_and_summarize_records(self):
        market = _market([100, 100, 101, 102, 103, 104, 105, 106], highs=[101] * 8, lows=[99] * 8)
        cfg = PathOutcomeConfig(hold_bars=2, entry_delay_bars=1, fee_rate=0.0, slippage_rate=0.0, mae_penalty=0.0)
        records = build_path_outcome_records(market, cfg, window_size=2, stride_bars=2)
        self.assertGreaterEqual(len(records), 2)
        summary = summarize_records(records)
        self.assertEqual(summary["num_records"], len(records))
        self.assertIn("TRADE", summary["gate_counts"])


if __name__ == "__main__":
    unittest.main()
