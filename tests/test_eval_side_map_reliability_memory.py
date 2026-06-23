import unittest

from training.eval_side_map_reliability_memory import _aux_signature, _history_labels, _majority, _market_signature


class TestSideMapReliabilityMemoryEval(unittest.TestCase):
    def test_history_labels(self):
        row = {"prompt": "- month=2025-01 label=inverse pass_cagr_bucket=negative\n- month=2025-02 label=normal"}
        self.assertEqual(_history_labels(row), ["inverse", "normal"])

    def test_majority(self):
        self.assertEqual(_majority(["normal", "inverse", "normal"]), "normal")
        self.assertEqual(_majority([], "unreliable"), "unreliable")

    def test_aux_signature(self):
        row = {"source": {"prior_binance_aux_tokens": {"prior_btc_premium_mean": "negative", "prior_btc_funding_mean": "positive", "prior_btc_funding_abs": "neutral"}}}
        self.assertIn("prior_btc_premium_mean=negative", _aux_signature(row))

    def test_market_signature(self):
        row = {"source": {"prior_market_structure_tokens": {"prior_market_return": "strong_down", "prior_market_drawdown": "high", "prior_market_close_pos": "near_low"}}}
        self.assertIn("prior_market_return=strong_down", _market_signature(row))


if __name__ == "__main__":
    unittest.main()
