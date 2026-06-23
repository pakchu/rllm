import unittest

from training.nested_gate_overlay_selection import _apply_month_gate, _period_rows, _score_selection


class TestNestedGateOverlaySelection(unittest.TestCase):
    def test_period_rows_uses_month_bounds(self):
        rows = [{"date": "2024-01-01"}, {"date": "2025-12-31"}, {"date": "2026-01-01"}]
        self.assertEqual(len(_period_rows(rows, "2024-01", "2025-12")), 2)

    def test_apply_month_gate_blocks_low_score_month(self):
        rows = [{"date": "2026-01-01", "prediction": {"gate": "TRADE"}}, {"date": "2026-02-01", "prediction": {"gate": "TRADE"}}]
        out = _apply_month_gate(rows, {"2026-01": -1.0, "2026-02": 2.0}, 0.5)
        self.assertEqual(out[0]["prediction"]["gate"], "NO_TRADE")
        self.assertEqual(out[1]["prediction"]["gate"], "TRADE")

    def test_score_penalizes_too_few_trades(self):
        score = _score_selection({"sim": {"trade_entries": 2, "cagr_pct": 100, "strict_mdd_pct": 1, "cagr_to_strict_mdd": 100}, "trade_stats": {"p_value_mean_ret_approx": 0.0}}, 30)
        self.assertLess(score, -900)


if __name__ == "__main__":
    unittest.main()
