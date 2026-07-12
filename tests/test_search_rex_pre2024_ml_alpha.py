import unittest

import numpy as np
import pandas as pd

from training.search_rex_pre2024_ml_alpha import (
    completed_before,
    feature_matrix,
    feature_names,
    fit_medians,
    prediction_rows,
    select_highest_per_signal,
)


class TestRexPre2024MlAlpha(unittest.TestCase):
    def test_features_only_read_signal_time_snapshot(self):
        rows = [
            {"feature_snapshot": {"past_a": 1.0, "past_b": np.nan}, "reward": {"future": 999.0}, "target": "TAKE"},
            {"feature_snapshot": {"past_a": 3.0, "past_b": 2.0}, "reward": {"future": -999.0}, "target": "SKIP"},
        ]
        names = feature_names(rows)
        self.assertEqual(names, ["past_a", "past_b"])
        medians = fit_medians(rows, names)
        matrix = feature_matrix(rows, names, medians)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(matrix[0, 1], 2.0)

    def test_completed_path_must_exit_strictly_before_cutoff(self):
        dates = pd.Series(pd.date_range("2022-12-31 23:40", periods=8, freq="5min"))
        row = {"signal_pos": 1}
        self.assertTrue(completed_before(row, dates, pd.Timestamp("2023-01-01"), hold_bars=1))
        self.assertFalse(completed_before(row, dates, pd.Timestamp("2022-12-31 23:55"), hold_bars=1))

    def test_highest_score_wins_when_candidates_share_signal(self):
        rows = [{"signal_pos": 3, "side": "LONG"}, {"signal_pos": 3, "side": "SHORT"}, {"signal_pos": 9, "side": "LONG"}]
        chosen = select_highest_per_signal(rows, np.asarray([0.1, 0.8, 0.2]))
        self.assertEqual([(row["signal_pos"], row["side"]) for row in chosen], [(3, "SHORT"), (9, "LONG")])

    def test_window_excludes_trade_exiting_at_boundary(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=12, freq="5min"))
        rows = [
            {"date": "2023-01-01 00:05:00", "signal_pos": 1, "side": "LONG", "family": "f"},
            {"date": "2023-01-01 00:40:00", "signal_pos": 8, "side": "LONG", "family": "f"},
        ]
        selected = prediction_rows(
            rows,
            np.asarray([1.0, 1.0]),
            threshold=0.0,
            side="both",
            family="both",
            market_dates=dates,
            start=pd.Timestamp("2023-01-01"),
            end=pd.Timestamp("2023-01-01 00:50:00"),
            hold_bars=1,
        )
        self.assertEqual([row["signal_pos"] for row in selected], [1])


if __name__ == "__main__":
    unittest.main()
