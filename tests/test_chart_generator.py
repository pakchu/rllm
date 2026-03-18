import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.chart_generator import ChartGenerator, ChartGeneratorConfig


def _sample_market_df(n: int = 140) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, size=n)))
    open_ = base
    close = base * np.exp(rng.normal(0, 0.001, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.002, 0.0005, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.002, 0.0005, size=n)))
    volume = rng.lognormal(mean=2.0, sigma=0.2, size=n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="1min"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestChartGenerator(unittest.TestCase):
    def test_render_window_shape_and_range(self):
        df = _sample_market_df()
        window = df.iloc[20:116].copy()
        gen = ChartGenerator(ChartGeneratorConfig(resolution=64))
        image = gen.render_window(window)

        self.assertEqual(image.shape, (3, 64, 64))
        self.assertEqual(image.dtype, np.float32)
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_fast_render_window_shape_and_range(self):
        df = _sample_market_df()
        window = df.iloc[20:116].copy()
        gen = ChartGenerator(ChartGeneratorConfig(resolution=64, backend="fast"))
        image = gen.render_window(window)

        self.assertEqual(image.shape, (3, 64, 64))
        self.assertEqual(image.dtype, np.float32)
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)
        self.assertGreater(float(image.sum()), 0.0)

    def test_render_at_is_leak_safe(self):
        df = _sample_market_df(n=170)
        t = 120
        w = 96
        gen = ChartGenerator(ChartGeneratorConfig(resolution=64))

        image_a = gen.render_at(df, t=t, w=w)
        df_extended = pd.concat([df, _sample_market_df(n=20)], ignore_index=True)
        image_b = gen.render_at(df_extended, t=t, w=w)

        self.assertTrue(np.allclose(image_a, image_b))

    def test_fast_render_at_is_leak_safe(self):
        df = _sample_market_df(n=170)
        t = 120
        w = 96
        gen = ChartGenerator(ChartGeneratorConfig(resolution=64, backend="fast"))

        image_a = gen.render_at(df, t=t, w=w)
        df_extended = pd.concat([df, _sample_market_df(n=20)], ignore_index=True)
        image_b = gen.render_at(df_extended, t=t, w=w)

        self.assertTrue(np.allclose(image_a, image_b))

    def test_cache_writes_and_reuses(self):
        df = _sample_market_df()
        window = df.iloc[10:106].copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ChartGenerator(
                ChartGeneratorConfig(resolution=64, cache_dir=tmpdir)
            )
            image1 = gen.render_window(window)
            files = list(Path(tmpdir).glob("*.npy"))
            self.assertGreaterEqual(len(files), 1)

            image2 = gen.render_window(window)
            self.assertTrue(np.array_equal(image1, image2))


if __name__ == "__main__":
    unittest.main()
