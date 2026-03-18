import unittest

from evaluation.metrics import max_drawdown_pct, summarize_metrics
from utils import log_returns, range_volatility_pct, sharpe_ratio_log


class TestMetrics(unittest.TestCase):
    def test_log_returns(self):
        rets = log_returns([100.0, 110.0, 121.0])
        self.assertEqual(len(rets), 2)
        self.assertAlmostEqual(float(rets[0]), float(rets[1]), places=12)

    def test_range_volatility_pct(self):
        vol = range_volatility_pct(highs=[110, 120], lows=[90, 95])
        self.assertGreater(vol, 0.0)
        self.assertAlmostEqual(vol, (120 - 90) / ((120 + 90) / 2))

    def test_sharpe_ratio_log_non_negative(self):
        rets = log_returns([100, 101, 102, 103, 104])
        sharpe = sharpe_ratio_log(rets, periods_per_year=365)
        self.assertGreater(sharpe, 0.0)

    def test_summarize_metrics(self):
        equity = [100.0, 105.0, 103.0, 110.0]
        underlying = [100.0, 102.0, 101.0, 103.0]
        report = summarize_metrics(equity, underlying, periods_per_year=365)
        self.assertIn("cumulative_return_pct", report)
        self.assertIn("max_drawdown_pct", report)
        self.assertIn("sharpe_ratio", report)
        self.assertIn("min_sharpe", report)
        self.assertGreaterEqual(report["max_drawdown_pct"], 0.0)

    def test_max_drawdown_pct(self):
        mdd = max_drawdown_pct([100, 120, 90, 95, 130])
        self.assertAlmostEqual(mdd, 25.0)


if __name__ == "__main__":
    unittest.main()

