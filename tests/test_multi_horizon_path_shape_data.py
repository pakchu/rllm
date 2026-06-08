import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.multi_horizon_path_shape_data import (
    MultiHorizonShapeConfig,
    build_path_shape_record,
    derive_path_shape_target,
    main,
    summarize_path_shape_records,
)
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


def _record(pos: int = 5, side: str = "LONG") -> dict:
    summary = {"regime": "UPTREND", "trend_alignment": "BULL_STACK", "risk_state": "CALM"}
    return {
        "date": str(pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=5 * pos)),
        "signal_pos": pos,
        "past_summary": summary,
        "prompt": "Past-only analyzer summary: " + json.dumps(summary),
        "source_edge_target": {"trend_side": side},
        "target": json.dumps({"decision": "ABSTAIN", "action_side": "NONE"}),
    }


class TestMultiHorizonPathShapeData(unittest.TestCase):
    def test_derives_path_shape_target(self):
        with tempfile.TemporaryDirectory() as td:
            market_path = Path(td) / "m.csv"
            _market(market_path)
            market = load_market_bars(str(market_path))
            cfg = MultiHorizonShapeConfig(hold_bars_list=(3, 6), fee_rate=0.0, slippage_rate=0.0, leverage=1.0)
            target = derive_path_shape_target(_record(5, "LONG"), market, cfg)
            self.assertEqual(target["trend_side"], "LONG")
            self.assertEqual(target["direction_stability"], "TREND_STABLE")
            self.assertEqual(target["horizons"]["3"]["best_path"], "TREND")
            self.assertEqual(target["horizons"]["6"]["trend_return_bucket"], "STRONG_POSITIVE")
            self.assertIn(target["risk_profile"], {"LOW_PATH_RISK", "MIXED_PATH_RISK", "HIGH_PATH_RISK", "EXTREME_PATH_RISK"})

    def test_derives_trend_side_from_edge_decay_target_schema(self):
        with tempfile.TemporaryDirectory() as td:
            market_path = Path(td) / "m.csv"
            _market(market_path)
            market = load_market_bars(str(market_path))
            cfg = MultiHorizonShapeConfig(hold_bars_list=(3,), fee_rate=0.0, slippage_rate=0.0, leverage=1.0)
            rec = _record(5, "LONG")
            rec.pop("source_edge_target")
            rec["target"] = json.dumps({"trend_side": "LONG", "edge_decay_label": "EDGE_PERSIST"})

            target = derive_path_shape_target(rec, market, cfg)

            self.assertEqual(target["trend_side"], "LONG")
            self.assertNotEqual(target["horizons"]["3"]["relative_edge"], "NO_TREND_SIDE")

    def test_build_record_uses_past_prompt_and_future_target_guard(self):
        with tempfile.TemporaryDirectory() as td:
            market_path = Path(td) / "m.csv"
            _market(market_path)
            market = load_market_bars(str(market_path))
            cfg = MultiHorizonShapeConfig(hold_bars_list=(3,), fee_rate=0.0, slippage_rate=0.0, leverage=1.0)
            rec = build_path_shape_record(_record(), market, cfg)
            self.assertEqual(rec["task"], "multi_horizon_path_shape_analyzer")
            self.assertIn("Do not choose a final trade", rec["prompt"])
            self.assertIn("Past-only analyzer summary", rec["prompt"])
            self.assertFalse(rec["leakage_guard"]["prompt_uses_future_path"])
            self.assertTrue(rec["leakage_guard"]["target_uses_future_ohlc_paths"])
            target = json.loads(rec["target"])
            self.assertIn("horizons", target)
            summary = summarize_path_shape_records([rec])
            self.assertEqual(summary["num_records"], 1)
            self.assertIn("3", summary["horizon_best_path"])

    def test_cli_writes_records_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            market_path = root / "m.csv"
            src = root / "records.jsonl"
            out = root / "shape.jsonl"
            summary = root / "summary.json"
            _market(market_path)
            src.write_text(json.dumps(_record()) + "\n")
            import sys

            old = sys.argv
            try:
                sys.argv = [
                    "prog",
                    "--market-csv",
                    str(market_path),
                    "--records",
                    str(src),
                    "--output",
                    str(out),
                    "--summary-output",
                    str(summary),
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
            self.assertEqual(len(out.read_text().splitlines()), 1)
            payload = json.loads(summary.read_text())
            self.assertEqual(payload["records"]["num_records"], 1)
            self.assertTrue(payload["records"]["leakage_guard"]["target_is_path_shape_not_final_action"])


if __name__ == "__main__":
    unittest.main()
