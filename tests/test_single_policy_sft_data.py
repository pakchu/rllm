import json
import unittest

from training.single_policy_sft_data import SinglePolicyConfig, exit_profile_for_hold, hold_bars_for_exit_profile, summarize_single_policy_rows


class TestSinglePolicySftData(unittest.TestCase):
    def test_exit_profile_mapping_hides_raw_hold_bars(self):
        self.assertEqual(exit_profile_for_hold(0), "AVOID")
        self.assertEqual(exit_profile_for_hold(36), "FAST")
        self.assertEqual(exit_profile_for_hold(72), "FAST")
        self.assertEqual(exit_profile_for_hold(144), "NORMAL")
        self.assertEqual(exit_profile_for_hold(288), "NORMAL")
        self.assertEqual(exit_profile_for_hold(432), "TRAIL")
        self.assertEqual(hold_bars_for_exit_profile("FAST"), 72)
        self.assertEqual(hold_bars_for_exit_profile("NORMAL"), 288)
        self.assertEqual(hold_bars_for_exit_profile("TRAIL"), 432)
        self.assertEqual(hold_bars_for_exit_profile("AVOID"), 0)

    def test_summary_counts_policy_fields(self):
        rows = [
            {"date": "2025-01-01", "prompt": "p", "target": json.dumps({"regime":"TREND_UP","edge_quality":"MODERATE","risk":"LOW","action":"LONG","exit_profile":"NORMAL","confidence":"MID"})},
            {"date": "2025-01-02", "prompt": "p", "target": json.dumps({"regime":"RANGE","edge_quality":"NONE","risk":"LOW","action":"NO_TRADE","exit_profile":"AVOID","confidence":"LOW"})},
        ]
        s = summarize_single_policy_rows(rows, cfg=SinglePolicyConfig())
        self.assertEqual(s["rows"], 2)
        self.assertEqual(s["field_counts"]["action"], {"LONG": 1, "NO_TRADE": 1})
        self.assertEqual(s["field_counts"]["exit_profile"], {"AVOID": 1, "NORMAL": 1})
        self.assertTrue(s["leakage_guard"]["targets_use_future_ohlc_utility"])


if __name__ == "__main__":
    unittest.main()
