import unittest

from training.augment_side_map_sft_with_binance_aux import _bucket, _prev_month, _tokens_for


class TestAugmentSideMapSftWithBinanceAux(unittest.TestCase):
    def test_prev_month(self):
        self.assertEqual(_prev_month("2026-01"), "2025-12")

    def test_bucket(self):
        self.assertEqual(_bucket(0.001, 0.0001, 0.0005), "high_positive")
        self.assertEqual(_bucket(-0.001, 0.0001, 0.0005), "high_negative")
        self.assertEqual(_bucket(0.0, 0.0001, 0.0005), "neutral")

    def test_tokens_for_missing(self):
        toks = _tokens_for(None)
        self.assertEqual(toks["prior_btc_funding_mean"], "unknown")


if __name__ == "__main__":
    unittest.main()
