import unittest

from training.alpha_inversion_audit import _invert_rule


class TestAlphaInversionAudit(unittest.TestCase):
    def test_invert_rule_swaps_sides_without_threshold_change(self):
        rule = {"low_threshold": -1.0, "high_threshold": 1.0, "high_side": "LONG", "low_side": "SHORT"}
        inv = _invert_rule(rule)
        self.assertEqual(inv["high_side"], "SHORT")
        self.assertEqual(inv["low_side"], "LONG")
        self.assertEqual(inv["low_threshold"], -1.0)
        self.assertEqual(inv["high_threshold"], 1.0)


if __name__ == "__main__":
    unittest.main()
