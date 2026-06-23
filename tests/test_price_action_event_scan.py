import unittest

import pandas as pd

from training.price_action_event_scan import build_price_action_event_features


class TestPriceActionEventScan(unittest.TestCase):
    def test_breakout_uses_prior_range_not_current_bar(self):
        market = pd.DataFrame({
            "open": [10, 10, 10, 10, 10, 10, 10],
            "high": [11, 11, 11, 11, 11, 11, 13],
            "low": [9, 9, 9, 9, 9, 9, 10],
            "close": [10, 10, 10, 10, 10, 10, 12],
            "volume": [1, 1, 1, 1, 1, 1, 2],
        })
        f = build_price_action_event_features(market, [3])
        self.assertEqual(float(f.loc[6, "pae_w3_break_above"]), 1.0)
        self.assertGreater(float(f.loc[6, "pae_w3_close_dist_prior_high"]), 0.0)

    def test_mutating_future_bar_does_not_change_past_event(self):
        market = pd.DataFrame({
            "open": [10, 10, 10, 10, 10, 10, 10],
            "high": [11, 11, 11, 11, 11, 11, 13],
            "low": [9, 9, 9, 9, 9, 9, 10],
            "close": [10, 10, 10, 10, 10, 10, 12],
            "volume": [1, 1, 1, 1, 1, 1, 2],
        })
        base = build_price_action_event_features(market, [3]).iloc[:6].copy()
        market.loc[6, ["high", "low", "close"]] = [100, 1, 50]
        changed = build_price_action_event_features(market, [3]).iloc[:6]
        pd.testing.assert_frame_equal(base, changed)


if __name__ == "__main__":
    unittest.main()
