import unittest

from training.path_shape_adaptive_semantic_veto_backtest import _history_start, _months


class TestAdaptiveSemanticVetoBacktest(unittest.TestCase):
    def test_history_start_rolls_year(self):
        self.assertEqual(_history_start("2025-01", 6), "2024-07-01")

    def test_months_filters_eval_start(self):
        rows = [{"date": "2024-12-31 00:00:00"}, {"date": "2025-01-01 00:00:00"}, {"date": "2025-01-15 00:00:00"}, {"date": "2025-02-01 00:00:00"}]
        self.assertEqual(_months(rows, "2025-01-01"), ["2025-01", "2025-02"])


if __name__ == "__main__":
    unittest.main()
