import unittest

from training.data_sources import make_synthetic_market_df
from training.train_sb3 import _augment_with_mirror_market


class TestTrainingHelpers(unittest.TestCase):
    def test_make_synthetic_market_df_schema(self):
        df = make_synthetic_market_df(num_rows=128, seed=7)

        self.assertEqual(len(df), 128)
        self.assertEqual(
            list(df.columns),
            ["date", "open", "high", "low", "close", "volume", "tic"],
        )
        self.assertTrue((df["high"] >= df[["open", "close"]].max(axis=1)).all())
        self.assertTrue((df["low"] <= df[["open", "close"]].min(axis=1)).all())
        self.assertEqual(df["tic"].nunique(), 1)
        self.assertEqual(df["tic"].iloc[0], "BTCUSDT")

    def test_augment_with_mirror_market(self):
        df = make_synthetic_market_df(num_rows=64, seed=3)
        aug = _augment_with_mirror_market(df)
        self.assertEqual(len(aug), 128)
        self.assertTrue((aug["high"] >= aug[["open", "close"]].max(axis=1)).all())
        self.assertTrue((aug["low"] <= aug[["open", "close"]].min(axis=1)).all())


if __name__ == "__main__":
    unittest.main()


from utils import disable_transformers_allocator_warmup

class TestAllocatorWarmupDisable(unittest.TestCase):
    def test_disable_transformers_allocator_warmup_context_restores(self):
        try:
            from transformers import modeling_utils
        except Exception:
            self.skipTest('transformers unavailable')
        original = getattr(modeling_utils, 'caching_allocator_warmup', None)
        self.assertIsNotNone(original)
        with disable_transformers_allocator_warmup():
            self.assertIsNot(modeling_utils.caching_allocator_warmup, original)
            self.assertIsNone(modeling_utils.caching_allocator_warmup())
        self.assertIs(modeling_utils.caching_allocator_warmup, original)
