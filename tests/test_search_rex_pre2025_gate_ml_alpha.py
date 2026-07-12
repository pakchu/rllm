import unittest

from training.audit_rex8640_usdkrw_gate import parse_gates
from training.search_rex_pre2025_gate_ml_alpha import DEFAULT_GATES, filter_gate_rows, parse_execution_holds


class TestRexPre2025GateMlAlpha(unittest.TestCase):
    def test_filter_applies_fixed_taker_and_range_gate(self):
        rows = [
            {"feature_snapshot": {"taker_imbalance": -0.2, "rex_2016_range_pos": 0.5}},
            {"feature_snapshot": {"taker_imbalance": 0.2, "rex_2016_range_pos": 0.5}},
            {"feature_snapshot": {"taker_imbalance": -0.2, "rex_2016_range_pos": 0.9}},
        ]
        selected = filter_gate_rows(rows, parse_gates(DEFAULT_GATES))
        self.assertEqual(selected, [rows[0]])

    def test_execution_holds_are_positive_unique_and_sorted(self):
        self.assertEqual(parse_execution_holds("144,72,144,48"), (48, 72, 144))


if __name__ == "__main__":
    unittest.main()
