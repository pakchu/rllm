import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.text_analyzer_trader_data import (
    TextPipelineConfig,
    build_text_pipeline_records,
    load_market_frame,
    main,
)


def _market_csv(path: Path, n: int = 260) -> None:
    prices = []
    p = 100.0
    for i in range(n):
        step = 1.001 if (i // 24) % 2 == 0 else 0.999
        p *= step
        prices.append(p)
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="5min"),
            "open": prices,
            "high": [x * 1.002 for x in prices],
            "low": [x * 0.998 for x in prices],
            "close": prices,
            "volume": [100.0 + (i % 13) for i in range(n)],
        }
    ).to_csv(path, index=False)


class TestTextAnalyzerTraderData(unittest.TestCase):
    def test_builds_past_only_analyzer_and_summary_only_trader_records(self):
        with tempfile.TemporaryDirectory() as td:
            market_path = Path(td) / "market.csv"
            _market_csv(market_path)
            market = load_market_frame(market_path)
            cfg = TextPipelineConfig(window_size=24, hold_bars=12, stride_bars=17, min_net_return=0.0, max_mae=0.03)
            analyzer, trader, path_rows = build_text_pipeline_records(market, cfg, max_records=5)
            self.assertEqual(len(analyzer), 5)
            self.assertEqual(len(trader), 5)
            self.assertEqual(len(path_rows), 5)
            target = json.loads(analyzer[0]["target"])
            self.assertIn("regime", target)
            self.assertIn("risk_state", target)
            self.assertIn("recent_bar_sequence", target)
            self.assertGreater(len(target["recent_bar_sequence"]), 0)
            self.assertIn("sequence_stats", target)
            self.assertFalse(analyzer[0]["leakage_guard"]["target_uses_future_path"])
            self.assertTrue(trader[0]["leakage_guard"]["prompt_uses_analyzer_summary_only"])
            self.assertNotIn("best_net_return", trader[0]["prompt"])
            self.assertIn("recent_bar_sequence", trader[0]["prompt"])
            action = json.loads(trader[0]["target"])
            self.assertIn(action["gate"], {"TRADE", "NO_TRADE"})
            self.assertIn(action["side"], {"LONG", "SHORT", "NONE"})

    def test_cli_writes_analyzer_trader_and_summary_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            market_path = Path(td) / "market.csv"
            analyzer_path = Path(td) / "analyzer.jsonl"
            trader_path = Path(td) / "trader.jsonl"
            summary_path = Path(td) / "summary.json"
            _market_csv(market_path)
            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "prog",
                    "--market-csv",
                    str(market_path),
                    "--analyzer-output",
                    str(analyzer_path),
                    "--trader-output",
                    str(trader_path),
                    "--summary-output",
                    str(summary_path),
                    "--window-size",
                    "24",
                    "--hold-bars",
                    "12",
                    "--stride-bars",
                    "17",
                    "--max-records",
                    "3",
                ]
                main()
            finally:
                sys.argv = old_argv
            self.assertEqual(len(analyzer_path.read_text().splitlines()), 3)
            self.assertEqual(len(trader_path.read_text().splitlines()), 3)
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["records"], {"analyzer": 3, "trader": 3})
            self.assertFalse(summary["leakage_guard"]["analyzer_target_uses_future_path"])


if __name__ == "__main__":
    unittest.main()
