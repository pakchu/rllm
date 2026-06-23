import unittest

from training.eval_event_side_rationale_preference import _adjust_scores


class TestEvalEventSideRationalePrior(unittest.TestCase):
    def test_adjust_scores_subtracts_prior(self):
        adjusted = _adjust_scores({"normal": 3.0, "inverse": 4.0}, {"normal": 2.0, "inverse": 3.5}, 1.0)
        self.assertEqual(adjusted, {"normal": 1.0, "inverse": 0.5})

    def test_adjust_scores_without_prior_returns_raw(self):
        self.assertEqual(_adjust_scores({"normal": 1.0}, None, 1.0), {"normal": 1.0})


if __name__ == "__main__":
    unittest.main()
