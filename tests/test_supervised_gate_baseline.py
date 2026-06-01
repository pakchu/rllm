import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.supervised_gate_baseline import run_baseline


def _market_csv(path: Path, n: int = 360) -> None:
    prices = []
    p = 100.0
    for i in range(n):
        phase = (i // 18) % 3
        step = 1.0015 if phase == 0 else (0.9985 if phase == 1 else 1.0003)
        p *= step
        prices.append(p)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="5min"),
            "open": prices,
            "high": [x * 1.002 for x in prices],
            "low": [x * 0.998 for x in prices],
            "close": prices,
            "volume": [100.0 + (i % 10) for i in range(n)],
        }
    )
    df.to_csv(path, index=False)


class TestSupervisedGateBaseline(unittest.TestCase):
    def test_run_baseline_produces_leakage_guarded_report(self):
        with tempfile.TemporaryDirectory() as td:
            market = Path(td) / "market.csv"
            out = Path(td) / "report.json"
            _market_csv(market)
            report = run_baseline(
                market_csv=str(market),
                output=str(out),
                train_start="2025-01-01 00:00:00",
                train_end="2025-01-01 10:00:00",
                test_start="2025-01-01 10:05:00",
                test_end="2025-01-01 20:00:00",
                eval_start="2025-01-01 20:05:00",
                eval_end="2025-01-02 05:00:00",
                window_size=8,
                hold_bars=6,
                entry_delay_bars=1,
                fee_rate=0.0,
                slippage_rate=0.0,
                leverage=1.0,
                mae_penalty=0.0,
                mfe_bonus=0.0,
                min_net_return=0.0,
                min_utility=0.0,
                max_mae=0.02,
                positive_weight=1.0,
                epochs=5,
                learning_rate=0.01,
                l2=0.0,
                cooldown_bars=0,
                min_trades=1,
            )
            self.assertTrue(out.exists())
            self.assertTrue(report["leakage_guard"]["features_are_past_only"])
            self.assertFalse(report["leakage_guard"]["eval_used_for_selection"])
            self.assertIn("eval", report["splits"])


if __name__ == "__main__":
    unittest.main()
