import unittest

import numpy as np
import pandas as pd

from training.search_positioning_continual_hgb_alpha import eligible_fit_mask, monthly_ranges


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


if __name__ == "__main__":
    unittest.main()
