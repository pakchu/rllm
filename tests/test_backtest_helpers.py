import unittest
from unittest.mock import patch

import numpy as np

from evaluation.backtest import (
    blend_weight_from_trend,
    build_regime_scores,
    build_regime_states,
    build_underlying_curve,
    normalize_action_probs,
    periods_per_year_from_timeframe,
    run_backtest_multi_seed,
)


class TestBacktestHelpers(unittest.TestCase):
    def test_build_underlying_curve(self):
        prices = np.array([100.0, 110.0, 90.0], dtype=np.float64)
        curve = build_underlying_curve(prices, initial_equity=1000.0)
        self.assertEqual(len(curve), 3)
        self.assertAlmostEqual(curve[0], 1000.0)
        self.assertAlmostEqual(curve[1], 1100.0)
        self.assertAlmostEqual(curve[2], 900.0)

    def test_periods_per_year_mapping(self):
        self.assertEqual(periods_per_year_from_timeframe("1m"), 365 * 24 * 60)
        self.assertEqual(periods_per_year_from_timeframe("5m"), 365 * 24 * 12)
        self.assertEqual(periods_per_year_from_timeframe("1d"), 365)
        self.assertEqual(periods_per_year_from_timeframe("unknown"), 365 * 24 * 60)

    def test_normalize_action_probs(self):
        p = normalize_action_probs(np.array([0.2, 0.2, 0.2], dtype=np.float64))
        self.assertAlmostEqual(float(np.sum(p)), 1.0, places=9)
        self.assertTrue(np.all(p >= 0.0))

        p2 = normalize_action_probs(np.array([np.nan, -1.0, 2.0], dtype=np.float64))
        self.assertAlmostEqual(float(np.sum(p2)), 1.0, places=9)
        self.assertAlmostEqual(float(p2[2]), 1.0, places=9)

        p3 = normalize_action_probs(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        self.assertAlmostEqual(float(np.sum(p3)), 1.0, places=9)
        self.assertTrue(np.allclose(p3, np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)))

    def test_blend_weight_from_trend(self):
        self.assertAlmostEqual(
            blend_weight_from_trend(
                mode="static",
                trend=0.01,
                weight_a=0.7,
                weight_up=0.9,
                weight_down=0.3,
                trend_threshold=0.002,
            ),
            0.7,
        )
        self.assertAlmostEqual(
            blend_weight_from_trend(
                mode="trend",
                trend=0.01,
                weight_a=0.7,
                weight_up=0.9,
                weight_down=0.3,
                trend_threshold=0.002,
            ),
            0.9,
        )
        self.assertAlmostEqual(
            blend_weight_from_trend(
                mode="trend",
                trend=-0.01,
                weight_a=0.7,
                weight_up=0.9,
                weight_down=0.3,
                trend_threshold=0.002,
            ),
            0.3,
        )

    def test_run_backtest_multi_seed_aggregation(self):
        def _fake_run_backtest(*, seed, **kwargs):
            del kwargs
            return {
                "cumulative_return_pct": float(seed),
                "sharpe_ratio": float(seed) / 10.0,
                "max_drawdown_pct": float(seed) / 100.0,
            }

        with patch("evaluation.backtest.run_backtest", side_effect=_fake_run_backtest):
            out = run_backtest_multi_seed(
                model_path="/tmp/x.zip",
                seeds=[1, 2, 3],
                source="synthetic",
            )
        self.assertIn("summary", out)
        self.assertIn("reports", out)
        self.assertEqual(len(out["reports"]), 3)
        self.assertAlmostEqual(out["summary"]["mean_cumulative_return_pct"], 2.0)

    def test_build_regime_scores_outputs(self):
        prices = np.concatenate(
            [
                np.linspace(100.0, 120.0, 200),
                np.linspace(120.0, 90.0, 200),
            ]
        ).astype(np.float64)
        out = build_regime_scores(
            prices,
            score_mode="raw",
            ret_short=5,
            ret_long=20,
            ema_fast=5,
            ema_slow=20,
            vol_lookback=10,
            z_window=50,
        )
        self.assertEqual(out["score_mode"], "raw")
        self.assertIn("score", out)
        self.assertEqual(len(out["score"]), len(prices))
        self.assertGreater(float(np.max(out["score"])), float(np.min(out["score"])))

    def test_build_regime_states_hysteresis(self):
        scores = np.asarray([0.0, 0.8, 0.9, 0.1, -0.2, -0.8, -0.9, -1.0], dtype=np.float64)
        states = build_regime_states(scores, enter_threshold=0.7, confirm_bars=2)
        self.assertTrue(np.array_equal(states, np.asarray([0, 0, 1, 1, 1, 1, -1, -1], dtype=np.int8)))


if __name__ == "__main__":
    unittest.main()
