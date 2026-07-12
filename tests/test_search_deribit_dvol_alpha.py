import unittest

import numpy as np
import pandas as pd

from training.search_deribit_dvol_alpha import attach_dvol


class TestSearchDeribitDvolAlpha(unittest.TestCase):
    def test_dvol_is_not_visible_before_candle_close(self):
        market = pd.DataFrame({"date": pd.date_range("2024-01-01 00:55", periods=3, freq="5min"), "open": 100.0})
        dvol = pd.DataFrame({"close_time": [pd.Timestamp("2024-01-01 01:00")], "open": [60.0], "high": [61.0], "low": [59.0], "close": [60.5]})
        joined = attach_dvol(market, dvol, tolerance="65min")
        self.assertTrue(np.isnan(joined.loc[0, "dvol_close"]))
        self.assertEqual(joined.loc[1:, "dvol_close"].tolist(), [60.5, 60.5])
        self.assertTrue((joined.loc[1:, "close_time"] <= joined.loc[1:, "date"]).all())


if __name__ == "__main__":
    unittest.main()
