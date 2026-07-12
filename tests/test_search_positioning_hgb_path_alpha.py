import unittest

import numpy as np
import pandas as pd

from training.search_positioning_hgb_path_alpha import (
    _feature_hash,
    causal_rolling_threshold,
    executable_path_utilities,
)
from training.search_positioning_disagreement_alpha import _simulate_no_stop


class TestPositioningHgbPathAlpha(unittest.TestCase):
    def test_path_utility_matches_next_open_and_side_mae(self):
        market = pd.DataFrame(
            {
                "open": [100.0, 100.0, 110.0, 110.0],
                "high": [100.0, 105.0, 110.0, 110.0],
                "low": [100.0, 80.0, 110.0, 110.0],
            }
        )
        utility = executable_path_utilities(
            market,
            np.array([0]),
            hold_bars=1,
            risk_lambda=0.5,
            leverage=0.5,
            side_cost=0.0,
            available_before_position=len(market),
        )
        self.assertAlmostEqual(float(utility[0, 0]), 0.5 * 0.10 - 0.5 * 0.5 * 0.20, places=6)
        self.assertAlmostEqual(float(utility[0, 1]), -0.5 * 0.10 - 0.5 * 0.5 * 0.05, places=6)

    def test_path_target_rejects_exit_at_or_after_fit_boundary(self):
        market = pd.DataFrame({"open": [100.0] * 5, "high": [100.0] * 5, "low": [100.0] * 5})
        utility = executable_path_utilities(
            market,
            np.array([0, 1]),
            hold_bars=2,
            risk_lambda=1.0,
            leverage=0.5,
            side_cost=0.0,
            available_before_position=4,
        )
        self.assertTrue(np.isfinite(utility[0]).all())
        self.assertTrue(np.isnan(utility[1]).all())

    def test_flat_path_target_cost_matches_backtest_account_cost(self):
        dates = pd.Series(pd.date_range("2023-01-01", "2023-12-31 23:55", freq="5min"))
        market = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0}, index=dates.index)
        side_cost = (0.0005 + 0.0001) * 0.5
        utility = executable_path_utilities(
            market,
            np.array([0]),
            hold_bars=1,
            risk_lambda=0.0,
            leverage=0.5,
            side_cost=side_cost,
            available_before_position=len(market),
        )
        long_active = np.zeros(len(market), dtype=bool)
        long_active[0] = True
        simulated = _simulate_no_stop(
            market,
            dates,
            long_active,
            np.zeros(len(market), dtype=bool),
            window="select2023",
            hold_bars=1,
            stride_bars=1,
            leverage=0.5,
            fee_rate=0.0005,
            slippage_rate=0.0001,
        )
        self.assertAlmostEqual(float(utility[0, 0]), simulated["return_pct"] / 100.0, places=8)

    def test_causal_threshold_does_not_use_current_or_future_scores(self):
        scores = np.arange(20.0)
        first = causal_rolling_threshold(scores, window_bars=8, quantile=0.5)
        changed = scores.copy()
        changed[12:] = 1_000.0
        second = causal_rolling_threshold(changed, window_bars=8, quantile=0.5)
        self.assertEqual(first[12], second[12])

    def test_feature_hash_changes_when_prefix_changes(self):
        frame = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]}, dtype=np.float32)
        same = frame.copy()
        changed = frame.copy()
        changed.loc[0, "a"] = 9.0
        self.assertEqual(_feature_hash(frame), _feature_hash(same))
        self.assertNotEqual(_feature_hash(frame), _feature_hash(changed))

    def test_feature_hash_includes_timestamps_when_supplied(self):
        frame = pd.DataFrame({"a": [1.0, 2.0]}, dtype=np.float32)
        first = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
        second = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-03"]))
        self.assertNotEqual(_feature_hash(frame, first), _feature_hash(frame, second))


if __name__ == "__main__":
    unittest.main()
