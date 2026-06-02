import unittest

from training.train_test_eval_regime_specialist_policy import _period_years


class TestTrainTestEvalRegimeSpecialistPolicy(unittest.TestCase):
    def test_period_years_is_positive(self):
        self.assertGreater(_period_years("2025-01-01", "2025-07-01"), 0)
