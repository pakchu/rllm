import unittest

import numpy as np
import pandas as pd

from training.search_rex_regime_routed_alpha import router_allows, routed_prediction_rows


class TestRexRegimeRoutedAlpha(unittest.TestCase):
    def test_zero_sign_router_is_directionally_exclusive(self):
        router = {"features": ["slow"], "mode": "all_sign"}
        positive = {"feature_snapshot": {"slow": 0.2}}
        negative = {"feature_snapshot": {"slow": -0.2}}
        self.assertTrue(router_allows(positive, "long", router))
        self.assertFalse(router_allows(positive, "short", router))
        self.assertTrue(router_allows(negative, "short", router))
        self.assertFalse(router_allows(negative, "long", router))

    def test_router_uses_only_current_row_state(self):
        rows = [
            {
                "date": "2023-01-01 00:05:00",
                "signal_pos": 1,
                "side": "LONG",
                "family": "f",
                "feature_snapshot": {"slow": 0.1},
            },
            {
                "date": "2023-01-01 00:10:00",
                "signal_pos": 2,
                "side": "SHORT",
                "family": "f",
                "feature_snapshot": {"slow": -0.1},
            },
        ]
        policy_long = {
            "model": "l",
            "side": "long",
            "family": "both",
            "score_threshold": 0.0,
            "score_scale": 1.0,
        }
        policy_short = {
            "model": "s",
            "side": "short",
            "family": "both",
            "score_threshold": 0.0,
            "score_scale": 1.0,
        }
        pair = {
            "long_policy": policy_long,
            "short_policy": policy_short,
            "router": {"features": ["slow"], "mode": "all_sign"},
        }
        selected = routed_prediction_rows(
            rows,
            {"l": np.asarray([1.0, 1.0]), "s": np.asarray([1.0, 1.0])},
            pair,
            market_dates=pd.Series(pd.date_range("2023-01-01", periods=10, freq="5min")),
            start=pd.Timestamp("2023-01-01"),
            end=pd.Timestamp("2023-01-01 00:45"),
            hold_bars=1,
        )
        self.assertEqual([row["prediction"]["side"] for row in selected], ["LONG", "SHORT"])


if __name__ == "__main__":
    unittest.main()
