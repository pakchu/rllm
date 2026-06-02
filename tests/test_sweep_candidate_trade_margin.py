import unittest

from training.sweep_candidate_trade_margin import extract_margins, summarize_margins


class TestSweepCandidateTradeMargin(unittest.TestCase):
    def test_extract_margins_from_preview_scores(self):
        report = {"generated_preview": [{"scores": [{"mean_logprob": -0.4}, {"mean_logprob": -0.1}]}]}
        self.assertAlmostEqual(extract_margins(report)[0], 0.3)

    def test_summarize_empty_and_non_empty(self):
        self.assertEqual(summarize_margins([])["count"], 0)
        self.assertEqual(summarize_margins([-1.0, 0.0, 1.0])["count"], 3)


if __name__ == "__main__":
    unittest.main()
