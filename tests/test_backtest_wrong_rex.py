import unittest

import numpy as np
import pandas as pd

from training.backtest_wrong_rex import _expanding_positive_quantile, _weekend_fx_closed


class TestWrongRexBacktest(unittest.TestCase):
    def test_expanding_threshold_uses_only_positive_history_including_current_row(self):
        strength = np.array([0.0, 1.0, np.nan, 3.0, -2.0, 5.0])
        actual = _expanding_positive_quantile(strength, quantile=0.75, min_count=2)

        self.assertTrue(np.isnan(actual[0]))
        self.assertTrue(np.isnan(actual[1]))
        self.assertEqual(actual[3], np.quantile([1.0, 3.0], 0.75))
        self.assertEqual(actual[5], np.quantile([1.0, 3.0, 5.0], 0.75))

    def test_weekend_quality_exception_matches_old_live_contract(self):
        dates = pd.Series(pd.to_datetime([
            "2026-07-10 21:55:00Z",  # Friday before FX close
            "2026-07-10 22:00:00Z",
            "2026-07-11 12:00:00Z",
            "2026-07-12 21:55:00Z",
            "2026-07-12 22:00:00Z",
        ]))

        self.assertEqual(_weekend_fx_closed(dates).tolist(), [False, True, True, True, False])
