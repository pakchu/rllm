import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.download_deribit_volatility_index import DeribitVolatilityConfig, download, run


class TestDownloadDeribitVolatilityIndex(unittest.TestCase):
    def test_paginates_backward_and_uses_close_time(self):
        pages = [
            {"result": {"data": [[3_600_000, 10, 12, 9, 11], [7_200_000, 11, 13, 10, 12]], "continuation": 3_599_999}},
            {"result": {"data": [[0, 9, 11, 8, 10], [3_600_000, 10, 12, 9, 11]]}},
        ]

        def fetch(_: dict):
            return pages.pop(0)

        cfg = DeribitVolatilityConfig(output_csv="unused.csv", start="1970-01-01", end="1970-01-01 02:00:00")
        frame = download(cfg, fetch=fetch)
        self.assertEqual(len(frame), 3)
        self.assertEqual(str(frame.loc[0, "date"]), "1970-01-01 00:00:00")
        self.assertEqual(str(frame.loc[0, "close_time"]), "1970-01-01 01:00:00")

    def test_run_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dvol.csv.gz"
            cfg = DeribitVolatilityConfig(output_csv=str(output), start="2024-01-01", end="2024-01-01 01:00:00")
            from unittest.mock import patch

            with patch(
                "training.download_deribit_volatility_index.get_json",
                return_value={"result": {"data": [[1704067200000, 60, 61, 59, 60.5], [1704070800000, 60.5, 62, 60, 61]]}},
            ):
                report = run(cfg)
            written = pd.read_csv(output)
            self.assertEqual(len(written), 2)
            self.assertEqual(report["rows"], 2)
            self.assertTrue(Path(str(output) + ".summary.json").exists())


if __name__ == "__main__":
    unittest.main()
