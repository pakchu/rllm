import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.text_step_analyzer_data import StepAnalyzerConfig, build_step_records, parse_hold_candidates


def _market(n: int = 260) -> pd.DataFrame:
    p = 100.0
    prices = []
    for i in range(n):
        step = 1.0012 if (i // 36) % 2 == 0 else 0.9988
        p *= step
        prices.append(p)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="5min"),
            "open": prices,
            "high": [x * 1.002 for x in prices],
            "low": [x * 0.998 for x in prices],
            "close": prices,
            "volume": [100.0 + i % 7 for i in range(n)],
        }
    )


class TestTextStepAnalyzerData(unittest.TestCase):
    def test_parse_hold_candidates(self):
        self.assertEqual(parse_hold_candidates("144,48,48"), (48, 144))

    def test_build_step_records_adds_preferred_step(self):
        cfg = StepAnalyzerConfig(window_size=24, hold_candidates=(12, 24), hold_bars=24, stride_bars=13, min_net_return=0.0, max_mae=0.03)
        analyzer, trader, path = build_step_records(_market(), cfg, max_records=5)
        self.assertEqual(len(analyzer), 5)
        self.assertEqual(len(trader), 5)
        target = json.loads(analyzer[0]["target"])
        self.assertIn("preferred_step_bars", target)
        self.assertIn(target["preferred_step_bars"], {0, 12, 24})
        trader_target = json.loads(trader[0]["target"])
        self.assertIn("hold_bars", trader_target)
        self.assertTrue(analyzer[0]["leakage_guard"]["target_step_uses_future_path"])


if __name__ == "__main__":
    unittest.main()
