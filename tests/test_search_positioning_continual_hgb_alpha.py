import unittest

import numpy as np
import pandas as pd

from training.search_positioning_continual_hgb_alpha import build_continual_features, eligible_fit_mask, monthly_ranges


class TestPositioningContinualHgbAlpha(unittest.TestCase):
    def test_monthly_ranges_are_half_open(self):
        ranges = monthly_ranges("2023-01-01", "2023-04-01")
        self.assertEqual(ranges[0], (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")))
        self.assertEqual(ranges[-1], (pd.Timestamp("2023-03-01"), pd.Timestamp("2023-04-01")))

    def test_fit_mask_requires_completed_exit_before_cutoff(self):
        dates = pd.date_range("2022-12-20", periods=5, freq="1D").to_numpy(dtype="datetime64[ns]")
        exits = np.array([10, 11, 12, 13, 14])
        mask = eligible_fit_mask(
            dates,
            exits,
            cutoff_position=13,
            cutoff_date=pd.Timestamp("2023-01-01"),
            train_days=0,
        )
        self.assertEqual(mask.tolist(), [True, True, True, False, False])

    def test_rolling_train_window_excludes_old_rows(self):
        dates = pd.to_datetime(["2020-01-01", "2022-01-01", "2022-12-01"]).to_numpy(dtype="datetime64[ns]")
        mask = eligible_fit_mask(
            dates,
            np.array([1, 2, 3]),
            cutoff_position=10,
            cutoff_date=pd.Timestamp("2023-01-01"),
            train_days=730,
        )
        self.assertEqual(mask.tolist(), [False, True, True])

    def test_dvol_feature_mode_adds_option_state_and_interactions(self):
        n = 30_000
        values = np.linspace(100.0, 120.0, n)
        market = pd.DataFrame(
            {
                "date": pd.date_range("2021-01-01", periods=n, freq="5min"),
                "open": values,
                "high": values + 1,
                "low": values - 1,
                "close": values,
                "quote_asset_volume": 1_000.0,
                "taker_buy_quote": 500.0,
                "sum_open_interest": 10_000.0,
                "positioning_available": 1.0,
                "count_toptrader_long_short_ratio": 1.1,
                "sum_toptrader_long_short_ratio": 1.2,
                "count_long_short_ratio": 1.0,
                "sum_taker_long_short_vol_ratio": 1.0,
                "dvol_close": np.linspace(60.0, 80.0, n),
                "dvol_available": 1.0,
            }
        )
        features = build_continual_features(market, include_dvol=True)
        self.assertIn("option_dvol_z25920", features.columns)
        self.assertIn("dvol_stress_x_trend", features.columns)


if __name__ == "__main__":
    unittest.main()
