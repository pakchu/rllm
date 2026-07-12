import unittest

import pandas as pd

from training.audit_rex8640_usdkrw_gate import gate_match, prediction_rows


class TestAuditRex8640UsdkrwGate(unittest.TestCase):
    def test_fixed_gate_uses_both_signal_time_conditions(self):
        base = {"feature_snapshot": {"rex_8640_range_width_pct": 0.30, "usdkrw_zscore": 0.20}}
        self.assertTrue(gate_match(base))
        self.assertFalse(gate_match({"feature_snapshot": {"rex_8640_range_width_pct": 0.20, "usdkrw_zscore": 0.20}}))
        self.assertFalse(gate_match({"feature_snapshot": {"rex_8640_range_width_pct": 0.30, "usdkrw_zscore": 0.30}}))

    def test_exit_at_window_boundary_is_rejected(self):
        market = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=8, freq="5min"),
                "open": [100.0] * 8,
                "high": [100.0] * 8,
                "low": [100.0] * 8,
                "close": [100.0] * 8,
            }
        )
        rows = [
            {
                "date": "2025-01-01 00:20:00",
                "signal_pos": 4,
                "action": {"side": "SHORT"},
                "feature_snapshot": {"rex_8640_range_width_pct": 0.30, "usdkrw_zscore": 0.20},
            }
        ]
        selected = prediction_rows(
            rows,
            market,
            start=pd.Timestamp("2025-01-01"),
            end=pd.Timestamp("2025-01-01 00:30:00"),
            side_mode="both",
            hold_bars=1,
        )
        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
