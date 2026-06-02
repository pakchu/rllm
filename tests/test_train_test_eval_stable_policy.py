import unittest

from training.train_test_eval_stable_policy import _period_years


class TestTrainTestEvalStablePolicy(unittest.TestCase):
    def test_period_years_is_positive(self):
        self.assertGreater(_period_years("2025-01-01", "2025-07-01"), 0)
