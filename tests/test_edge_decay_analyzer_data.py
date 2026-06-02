import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.edge_decay_analyzer_data import (
    EdgeDecayConfig,
    build_edge_decay_records,
    classify_edge_decay,
    main,
)
from training.text_analyzer_trader_data import load_market_frame


def _market_csv(path: Path, n: int = 360) -> None:
    prices = []
    p = 100.0
    for i in range(n):
        # Up trend that later rolls over so both persist/decay labels can appear.
        step = 1.0015 if i < n // 2 else 0.9985
        p *= step
        prices.append(p)
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="5min"),
            "open": prices,
            "high": [x * 1.002 for x in prices],
            "low": [x * 0.998 for x in prices],
            "close": prices,
            "volume": [100.0 + (i % 7) for i in range(n)],
        }
    ).to_csv(path, index=False)


class TestEdgeDecayAnalyzerData(unittest.TestCase):
    def test_classifies_persist_decay_and_reversal_without_gate_labels(self):
        cfg = EdgeDecayConfig(min_edge=0.001, max_mae=0.02)
        persist = classify_edge_decay(
            trend_side="LONG",
            short_same={"net_return": 0.002, "mae": 0.003},
            long_same={"net_return": 0.004, "mae": 0.004},
            long_opposite={"net_return": -0.004, "mae": 0.004},
            cfg=cfg,
        )
        decay = classify_edge_decay(
            trend_side="LONG",
            short_same={"net_return": 0.002, "mae": 0.003},
            long_same={"net_return": -0.002, "mae": 0.01},
            long_opposite={"net_return": 0.002, "mae": 0.004},
            cfg=cfg,
        )
        self.assertEqual(persist["edge_decay_label"], "EDGE_PERSIST")
        self.assertEqual(decay["edge_decay_label"], "EDGE_DECAY")

    def test_build_records_use_past_prompt_and_future_transition_target(self):
        with tempfile.TemporaryDirectory() as td:
            market_path = Path(td) / "market.csv"
            _market_csv(market_path)
            market = load_market_frame(market_path)
            cfg = EdgeDecayConfig(window_size=48, short_hold_bars=24, long_hold_bars=72, stride_bars=24, trend_feature="trend_96", trend_threshold=0.0)
            records = build_edge_decay_records(market, cfg, max_records=5)
            self.assertEqual(len(records), 5)
            target = json.loads(records[0]["target"])
            self.assertIn(target["edge_decay_label"], {"EDGE_PERSIST", "EDGE_DECAY", "REVERSAL_RISK", "ADVERSE_STRESS", "NO_EDGE", "WEAK_PERSIST", "WEAK_DECAY", "NO_CLEAR_TREND"})
            self.assertNotIn("TRADE", records[0]["target"])
            self.assertFalse(records[0]["leakage_guard"]["prompt_uses_future_path"])
            self.assertTrue(records[0]["leakage_guard"]["target_uses_future_path"])
            self.assertTrue(records[0]["leakage_guard"]["not_gate_threshold_optimization"])

    def test_cli_writes_records_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            market_path = Path(td) / "market.csv"
            out_path = Path(td) / "edge.jsonl"
            summary_path = Path(td) / "summary.json"
            _market_csv(market_path)
            import sys

            old = sys.argv
            try:
                sys.argv = [
                    "prog",
                    "--market-csv",
                    str(market_path),
                    "--output",
                    str(out_path),
                    "--summary-output",
                    str(summary_path),
                    "--window-size",
                    "48",
                    "--short-hold-bars",
                    "24",
                    "--long-hold-bars",
                    "72",
                    "--stride-bars",
                    "24",
                    "--max-records",
                    "4",
                ]
                main()
            finally:
                sys.argv = old
            self.assertEqual(len(out_path.read_text().splitlines()), 4)
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["records"]["num_records"], 4)
            self.assertTrue(summary["records"]["leakage_guard"]["not_gate_threshold_optimization"])


if __name__ == "__main__":
    unittest.main()
