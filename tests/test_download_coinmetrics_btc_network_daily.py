import gzip
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from training.download_coinmetrics_btc_network_daily import CoinMetricsBTCNetworkDailyConfig, download_rows, run


def _unix(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def _row(day: str, *, adr: int = 100, tx: int = 200, transfers: int = 300, available: str | None = None) -> dict:
    available = available or f"{day[:8]}{int(day[8:10]) + 1:02d}T00:00:00"
    return {
        "asset": "btc",
        "time": f"{day}T00:00:00.000000000Z",
        "AdrActCnt": str(adr),
        "TxCnt": str(tx),
        "TxTfrCnt": str(transfers),
        "AssetEODCompletionTime": str(_unix(available)),
    }


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class TestDownloadCoinMetricsBTCNetworkDaily(unittest.TestCase):
    def test_paginates_sorts_and_dedupes_rows(self):
        cfg = CoinMetricsBTCNetworkDailyConfig(
            output="unused.csv.gz",
            manifest="unused.json",
            start="2024-01-01",
            end="2024-01-03",
        )
        pages = [
            {
                "data": [_row("2024-01-02", adr=222), _row("2024-01-01", adr=111)],
                "next_page_url": "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?page_token=next",
            },
            {"data": [_row("2024-01-02", adr=999), _row("2024-01-03", adr=333)]},
        ]
        requested_urls = []

        def fetch(url: str) -> dict:
            requested_urls.append(url)
            return pages.pop(0)

        rows, first_url = download_rows(cfg, fetch=fetch)

        self.assertIn("assets=btc", first_url)
        self.assertIn("AdrActCnt%2CTxCnt%2CTxTfrCnt%2CAssetEODCompletionTime", first_url)
        self.assertEqual(len(requested_urls), 2)
        self.assertEqual([row["observation_date"] for row in rows], ["2024-01-01 00:00:00", "2024-01-02 00:00:00", "2024-01-03 00:00:00"])
        self.assertEqual(rows[1]["AdrActCnt"], "999")
        self.assertEqual(rows[0]["available_at"], "2024-01-02 00:00:00")

    def test_rejects_rows_available_before_observation_plus_one_day(self):
        cfg = CoinMetricsBTCNetworkDailyConfig(output="unused.csv.gz", manifest="unused.json")

        with self.assertRaisesRegex(ValueError, "availability precedes required daily lag"):
            download_rows(cfg, fetch=lambda _: {"data": [_row("2024-01-01", available="2024-01-01T23:59:59")]})

    def test_rejects_non_positive_metrics(self):
        cfg = CoinMetricsBTCNetworkDailyConfig(output="unused.csv.gz", manifest="unused.json")

        with self.assertRaisesRegex(ValueError, "TxCnt must be positive"):
            download_rows(cfg, fetch=lambda _: {"data": [_row("2024-01-01", tx=0)]})

    def test_run_uses_mocked_urlopen_and_writes_deterministic_gzip_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "coinmetrics_btc_network_daily.csv.gz"
            manifest_path = Path(tmp) / "coinmetrics_btc_network_daily.manifest.json"
            cfg = CoinMetricsBTCNetworkDailyConfig(
                output=str(output),
                manifest=str(manifest_path),
                start="2024-01-01",
                end="2024-01-02",
                page_size=1,
            )
            payloads = [
                {
                    "data": [_row("2024-01-02", adr=20)],
                    "next_page_url": "/v4/timeseries/asset-metrics?page_token=abc",
                },
                {"data": [_row("2024-01-01", adr=10)]},
            ]
            calls = []

            def fake_urlopen(request, timeout):
                calls.append((request.full_url, timeout, request.headers.get("User-agent")))
                return _Response(payloads.pop(0))

            with patch("training.download_coinmetrics_btc_network_daily.urllib.request.urlopen", side_effect=fake_urlopen):
                report1 = run(cfg)
            gzip_bytes_1 = output.read_bytes()
            manifest_1 = manifest_path.read_text()

            payloads = [
                {
                    "data": [_row("2024-01-02", adr=20)],
                    "next_page_url": "/v4/timeseries/asset-metrics?page_token=abc",
                },
                {"data": [_row("2024-01-01", adr=10)]},
            ]
            with patch("training.download_coinmetrics_btc_network_daily.urllib.request.urlopen", side_effect=fake_urlopen):
                report2 = run(cfg)

            self.assertEqual(gzip_bytes_1, output.read_bytes())
            self.assertEqual(manifest_1, manifest_path.read_text())
            self.assertEqual(report1, report2)
            manifest = json.loads(manifest_1)
            self.assertEqual(manifest["rows"], 2)
            self.assertEqual(manifest["row_range"], {"start": "2024-01-01 00:00:00", "end": "2024-01-02 00:00:00"})
            self.assertEqual(manifest["sha256"], hashlib.sha256(gzip_bytes_1).hexdigest())
            self.assertIn("source_url", manifest)
            with gzip.open(output, "rt", encoding="utf-8") as fh:
                csv_text = fh.read()
            self.assertEqual(
                csv_text,
                "observation_date,available_at,AdrActCnt,TxCnt,TxTfrCnt\n"
                "2024-01-01 00:00:00,2024-01-02 00:00:00,10,200,300\n"
                "2024-01-02 00:00:00,2024-01-03 00:00:00,20,200,300\n",
            )
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[0][1], 30.0)
            self.assertEqual(calls[0][2], "rllm-research/1.0")


if __name__ == "__main__":
    unittest.main()
