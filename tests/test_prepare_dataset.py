import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from training.prepare_dataset import (
    prepare_observation_dataset,
    save_dataset_npz,
)


def _market_df(n: int = 70) -> pd.DataFrame:
    base = np.linspace(100, 120, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="1min"),
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base * 1.001,
            "volume": np.ones(n),
        }
    )


class TestPrepareDataset(unittest.TestCase):
    def test_prepare_observation_dataset_shapes(self):
        df = _market_df(70)
        dataset = prepare_observation_dataset(
            market_df=df,
            window_size=32,
            resolution=32,
            cache_dir=None,
            image_render_backend="fast",
        )
        expected_n = 70 - 32 + 1

        self.assertEqual(dataset["images"].shape, (expected_n, 3, 32, 32))
        self.assertEqual(dataset["scalars"].shape, (expected_n, 3))
        self.assertEqual(len(dataset["dates"]), expected_n)

    def test_save_dataset_npz(self):
        df = _market_df(40)
        dataset = prepare_observation_dataset(
            market_df=df,
            window_size=16,
            resolution=32,
            cache_dir=None,
            image_render_backend="fast",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "obs.npz"
            saved = save_dataset_npz(dataset, str(output))
            self.assertTrue(Path(saved).exists())

            loaded = np.load(saved)
            self.assertEqual(loaded["images"].shape[0], dataset["images"].shape[0])
            self.assertEqual(loaded["scalars"].shape[1], 3)


if __name__ == "__main__":
    unittest.main()
