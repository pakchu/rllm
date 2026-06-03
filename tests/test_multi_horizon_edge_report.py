import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.multi_horizon_edge_report import best_horizon_table, main, parse_horizons, run_report
from training.strict_bar_backtest import load_market_bars


def _market(path: Path) -> None:
    prices = [100 + i for i in range(80)]
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(prices), freq="5min"),
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
        }
    ).to_csv(path, index=False)


def _row(pos: int, side: str = "LONG") -> dict:
    return {
        "date": str(pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=5 * pos)),
        "signal_pos": pos,
        "source_edge_target": {"trend_side": side},
        "target": json.dumps({"decision": "ABSTAIN", "action_side": "NONE"}),
    }


class TestMultiHorizonEdgeReport(unittest.TestCase):
    def test_parse_horizons(self):
        self.assertEqual(parse_horizons("3, 5,8"), (3, 5, 8))

    def test_run_report_summarizes_trend_fade_oracle(self):
        with tempfile.TemporaryDirectory() as td:
            market_path = Path(td) / "m.csv"
            _market(market_path)
            market = load_market_bars(str(market_path))
            splits = {"train": [_row(5, "LONG"), _row(10, "LONG")], "val": [_row(15, "LONG")]}
            report = run_report(
                splits,
                market,
                (3, 6),
                {"entry_delay_bars": 1, "fee_rate": 0.0, "slippage_rate": 0.0, "leverage": 1.0},
            )
            self.assertGreater(report["3"]["train"]["TREND"]["trade_stats"]["mean_return_pct"], 0.0)
            self.assertLess(report["3"]["train"]["FADE"]["trade_stats"]["mean_return_pct"], 0.0)
            self.assertGreater(report["3"]["val"]["ORACLE"]["trades"], 0)
            table = best_horizon_table(report)
            self.assertEqual(table[0]["hold_bars"], 3)
            self.assertEqual(table[0]["train_best_static_action"], "TREND")

    def test_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            market_path = root / "m.csv"
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            out = root / "report.json"
            _market(market_path)
            text = json.dumps(_row(5, "LONG")) + "\n"
            train.write_text(text)
            val.write_text(text)
            import sys

            old = sys.argv
            try:
                sys.argv = [
                    "prog",
                    "--market-csv",
                    str(market_path),
                    "--train-jsonl",
                    str(train),
                    "--val-jsonl",
                    str(val),
                    "--output",
                    str(out),
                    "--hold-bars-list",
                    "3,6",
                    "--fee-rate",
                    "0",
                    "--slippage-rate",
                    "0",
                    "--leverage",
                    "1",
                ]
                main()
            finally:
                sys.argv = old
            payload = json.loads(out.read_text())
            self.assertIn("summary_table", payload)
            self.assertTrue(payload["leakage_guard"]["oracle_is_upper_bound_not_deployable"])


if __name__ == "__main__":
    unittest.main()
