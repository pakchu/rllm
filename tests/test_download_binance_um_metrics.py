import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from training.download_binance_um_metrics import METRIC_COLUMNS, MetricsDownloadConfig, _parse_archive, run


def _frame(day: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "create_time": [f"{day} 00:00:00", f"{day} 00:05:00"],
            "symbol": [symbol, symbol],
            "sum_open_interest": [100.0, 101.0],
            "sum_open_interest_value": [1_000.0, 1_010.0],
            "count_toptrader_long_short_ratio": [1.1, 1.2],
            "sum_toptrader_long_short_ratio": [1.0, 1.1],
            "count_long_short_ratio": [0.9, 1.0],
            "sum_taker_long_short_vol_ratio": [0.8, 1.3],
        }
    )


def _archive(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("metrics.csv", frame.to_csv(index=False))
    return buf.getvalue()


class TestDownloadBinanceUmMetrics(unittest.TestCase):
    def test_parse_archive_validates_and_normalises(self):
        parsed = _parse_archive(_archive(_frame("2024-01-01")), symbol="BTCUSDT", day="2024-01-01")
        self.assertEqual(list(parsed.columns), list(METRIC_COLUMNS))
        self.assertEqual(str(parsed.loc[0, "create_time"]), "2024-01-01 00:00:00")
        self.assertEqual(float(parsed.loc[1, "sum_taker_long_short_vol_ratio"]), 1.3)

    def test_run_orders_days_and_writes_causality_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "metrics.csv.gz"
            cfg = MetricsDownloadConfig(output_csv=str(output), start="2024-01-01", end="2024-01-02", workers=2)

            def fake_download(_: MetricsDownloadConfig, day: str) -> pd.DataFrame:
                return _frame(day).assign(create_time=lambda x: pd.to_datetime(x["create_time"]))

            with patch("training.download_binance_um_metrics._download_day", side_effect=fake_download):
                report = run(cfg)
            written = pd.read_csv(output, parse_dates=["create_time"])

        self.assertEqual(len(written), 4)
        self.assertTrue(written["create_time"].is_monotonic_increasing)
        self.assertEqual(report["days"], 2)
        self.assertIn("shift by >=1", report["causality_requirement"])


if __name__ == "__main__":
    unittest.main()
