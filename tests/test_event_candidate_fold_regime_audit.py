import unittest

import pandas as pd

from training.event_candidate_fold_regime_audit import _regime_metrics


class TestEventCandidateFoldRegimeAudit(unittest.TestCase):
    def test_regime_metrics_compute_past_window_shape(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 110.0, 105.0, 120.0] * 3,
                "high": [101.0, 112.0, 106.0, 122.0] * 3,
                "low": [99.0, 108.0, 100.0, 118.0] * 3,
                "close": [100.5, 111.0, 104.0, 121.0] * 3,
            }
        )
        m = _regime_metrics(df, "x")
        self.assertEqual(m["x_rows"], 12.0)
        self.assertIn("x_ret_pct", m)
        self.assertGreaterEqual(m["x_range_pos"], 0.0)


if __name__ == "__main__":
    unittest.main()
