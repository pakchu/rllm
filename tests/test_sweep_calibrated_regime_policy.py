import unittest

from training.sweep_calibrated_regime_policy import _augment_metrics, _copy_with_keys, _parse_key_sets


class TestSweepCalibratedRegimePolicy(unittest.TestCase):
    def test_parse_key_sets_uses_semicolon_groups(self):
        self.assertEqual(_parse_key_sets("a,b; c"), [("a", "b"), ("c",)])

    def test_copy_with_keys_recomputes_from_summary(self):
        rows = [{"summary": {"regime": "RANGE", "symbolic_features": {"Macro Dollar State": "RISK_ON"}}, "key": "old"}]
        keyed = _copy_with_keys(rows, ("regime", "Macro Dollar State"))
        self.assertEqual(keyed[0]["key"], "regime=RANGE|Macro Dollar State=RISK_ON")
        self.assertEqual(rows[0]["key"], "old")

    def test_augment_metrics_adds_ratio(self):
        out = _augment_metrics({"compounded_return": 0.21, "strict_mdd_proxy": 0.1}, years=1.0)
        self.assertAlmostEqual(out["cagr_proxy"], 0.21)
        self.assertAlmostEqual(out["cagr_to_mdd_proxy"], 2.1)


if __name__ == "__main__":
    unittest.main()
