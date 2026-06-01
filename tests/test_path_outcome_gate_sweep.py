import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.path_outcome_gate_sweep import run_sweep


def _market_csv(path: Path, n: int = 260) -> None:
    prices = []
    p = 100.0
    for i in range(n):
        # Alternating multi-bar swings make both sides sometimes valid.
        phase = (i // 12) % 2
        p *= 1.002 if phase == 0 else 0.998
        prices.append(p)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="5min"),
            "open": prices,
            "high": [x * 1.002 for x in prices],
            "low": [x * 0.998 for x in prices],
            "close": prices,
        }
    )
    df.to_csv(path, index=False)


class TestPathOutcomeGateSweep(unittest.TestCase):
    def test_run_sweep_writes_oracle_report(self):
        with tempfile.TemporaryDirectory() as td:
            market = Path(td) / "market.csv"
            out = Path(td) / "sweep.json"
            _market_csv(market)
            report = run_sweep(
                market_csv=str(market),
                output=str(out),
                train_start="2025-01-01 00:00:00",
                train_end="2025-01-01 08:00:00",
                test_start="2025-01-01 08:05:00",
                test_end="2025-01-01 16:00:00",
                eval_start="2025-01-01 16:05:00",
                eval_end="2025-01-02 00:00:00",
                window_size=8,
                hold_bars=6,
                entry_delay_bars=1,
                fee_rate=0.0,
                slippage_rate=0.0,
                leverage=1.0,
                mae_penalty_values=[0.0],
                mfe_bonus_values=[0.0],
                min_net_return_values=[0.0],
                min_utility_values=[0.0],
                max_mae_values=[0.02],
                cooldown_bars_values=[0],
                min_train_trades=1,
                stride_bars=1,
            )
            self.assertTrue(out.exists())
            self.assertEqual(report["search_summary"]["num_candidates"], 1)
            self.assertTrue(report["leakage_note"]["oracle_future_labels"])
            self.assertIn("top", report)


if __name__ == "__main__":
    unittest.main()
