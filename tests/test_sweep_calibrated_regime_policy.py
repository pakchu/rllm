import unittest

from training.sweep_calibrated_regime_policy import _augment_metrics, _copy_with_keys, _evaluate_rules_precomputed, _parse_key_sets, _precompute_action_rows


class TestSweepCalibratedRegimePolicy(unittest.TestCase):
    def test_parse_key_sets_uses_semicolon_groups(self):
        self.assertEqual(_parse_key_sets("a,b; c"), [("a", "b"), ("c",)])

    def test_copy_with_keys_recomputes_from_summary(self):
        rows = [{"summary": {"regime": "RANGE", "symbolic_features": {"Macro Dollar State": "RISK_ON"}}, "key": "old"}]
        keyed = _copy_with_keys(rows, ("regime", "Macro Dollar State"))
        self.assertEqual(keyed[0]["key"], "regime=RANGE|Macro Dollar State=RISK_ON")
        self.assertEqual(rows[0]["key"], "old")

    def test_evaluate_precomputed_strict_skips_overlaps(self):
        records = [
            {"signal_pos": 0, "key": "k", "actions": {"LONG_3": {"side": "LONG", "hold_bars": 3, "net_return": 0.1, "mae": 0.01, "utility": 0.09}}},
            {"signal_pos": 1, "key": "k", "actions": {"LONG_3": {"side": "LONG", "hold_bars": 3, "net_return": 0.1, "mae": 0.01, "utility": 0.09}}},
        ]
        rules = {"k": {"action": {"side": "LONG", "hold_bars": 3}}}
        out = _evaluate_rules_precomputed(records_count=2, action_rows=_precompute_action_rows(records), rules=rules, non_overlapping=True)
        self.assertEqual(out["trades"], 1)

    def test_augment_metrics_adds_ratio(self):
        out = _augment_metrics({"compounded_return": 0.21, "strict_mdd_proxy": 0.1}, years=1.0)
        self.assertAlmostEqual(out["cagr_proxy"], 0.21)
        self.assertAlmostEqual(out["cagr_to_mdd_proxy"], 2.1)


if __name__ == "__main__":
    unittest.main()
