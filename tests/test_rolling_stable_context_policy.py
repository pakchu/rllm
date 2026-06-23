import unittest
import pandas as pd

from training.rolling_stable_context_policy import _relabel_for_selection, month_starts


class TestRollingStableContextPolicy(unittest.TestCase):
    def test_month_starts_excludes_end_later_in_runner(self):
        months = month_starts("2025-01-15", "2025-03-01")
        self.assertEqual([str(m.date()) for m in months], ["2025-01-01", "2025-02-01", "2025-03-01"])

    def test_relabel_uses_only_rows_before_month(self):
        rows = [
            {"_dt": pd.Timestamp("2025-01-01"), "split": "old"},
            {"_dt": pd.Timestamp("2025-06-01"), "split": "old"},
            {"_dt": pd.Timestamp("2025-07-01"), "split": "old"},
        ]
        out = _relabel_for_selection(rows, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01"))
        self.assertEqual([r["split"] for r in out], ["train", "test"])
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
