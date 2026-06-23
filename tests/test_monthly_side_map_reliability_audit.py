import unittest

from training.monthly_side_map_reliability_audit import _apply, _month


class TestMonthlySideMapReliabilityAudit(unittest.TestCase):
    def test_month_extracts_year_month(self):
        self.assertEqual(_month({"date": "2026-01-02 03:00:00"}), "2026-01")

    def test_apply_invert_flips_trade_side(self):
        rows = [{"prediction": {"gate": "TRADE", "side": "LONG"}}]
        self.assertEqual(_apply(rows, "invert")[0]["prediction"]["side"], "SHORT")
        self.assertEqual(_apply(rows, "block")[0]["prediction"]["gate"], "NO_TRADE")


if __name__ == "__main__":
    unittest.main()
