import unittest

from training.augment_side_map_sft_with_market_structure import _bucket_signed, _close_pos_bucket, _prev_month


class TestAugmentSideMapSftWithMarketStructure(unittest.TestCase):
    def test_prev_month(self):
        self.assertEqual(_prev_month("2022-01"), "2021-12")

    def test_bucket_signed(self):
        self.assertEqual(_bucket_signed(0.12, 0.03, 0.10), "strong_up")
        self.assertEqual(_bucket_signed(-0.12, 0.03, 0.10), "strong_down")
        self.assertEqual(_bucket_signed(0.01, 0.03, 0.10), "flat")

    def test_close_pos_bucket(self):
        self.assertEqual(_close_pos_bucket(0.9), "near_high")
        self.assertEqual(_close_pos_bucket(0.1), "near_low")
        self.assertEqual(_close_pos_bucket(0.5), "mid_range")


if __name__ == "__main__":
    unittest.main()
