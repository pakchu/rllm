import unittest

from training.sparse_setup_train_test_eval_validator import _load_folds, _score_period


class TestSparseSetupTrainTestEvalValidator(unittest.TestCase):
    def test_load_folds_sorts_by_start(self):
        folds = _load_folds('[{"name":"b","eval_start":"2024-07-01","eval_end":"2024-12-31"},{"name":"a","eval_start":"2024-01-01","eval_end":"2024-06-30"}]')
        self.assertEqual([f["name"] for f in folds], ["a", "b"])

    def test_score_period_penalizes_low_trade_count(self):
        low = {"sim": {"trade_entries": 1, "cagr_pct": 100.0, "strict_mdd_pct": 1.0, "cagr_to_strict_mdd": 100.0}}
        ok = {"sim": {"trade_entries": 30, "cagr_pct": 10.0, "strict_mdd_pct": 5.0, "cagr_to_strict_mdd": 2.0}}
        self.assertLess(_score_period(low, min_trades=20), _score_period(ok, min_trades=20))


if __name__ == "__main__":
    unittest.main()
