from __future__ import annotations

import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.download_coinmetrics_stablecoin_supply_daily import (
    ASSETS,
    CoinMetricsStablecoinSupplyConfig,
    download_rows,
    run,
)


def row(asset: str, day: str, completion: int, supply: str = "1000000") -> dict[str, str]:
    return {
        "asset": asset,
        "time": f"{day}T00:00:00.000000000Z",
        "AssetEODCompletionTime": str(completion),
        "SplyCur": supply,
    }


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class TestDownloadCoinMetricsStablecoinSupplyDaily(unittest.TestCase):
    def test_paginates_sorts_and_dedupes_rows(self) -> None:
        cfg = CoinMetricsStablecoinSupplyConfig(output="unused.csv.gz", manifest="unused.json")
        pages = iter(
            [
                {
                    "data": [row("usdt_trx", "2023-01-02", 1672707000)],
                    "next_page_url": "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?page_token=next",
                },
                {
                    "data": [
                        row("usdt_eth", "2023-01-01", 1672619400),
                        row("usdt_trx", "2023-01-02", 1672707001, "1000001"),
                    ]
                },
            ]
        )
        rows, url = download_rows(cfg, fetch=lambda _: next(pages))
        self.assertIn("usdt_eth%2Cusdt_trx", url)
        self.assertEqual(
            [(item["observation_date"][:10], item["asset"]) for item in rows],
            [("2023-01-01", "usdt_eth"), ("2023-01-02", "usdt_trx")],
        )
        self.assertEqual(rows[1]["supply"], "1000001")

    def test_rejects_unknown_asset(self) -> None:
        cfg = CoinMetricsStablecoinSupplyConfig(output="unused.csv.gz", manifest="unused.json")
        with self.assertRaisesRegex(ValueError, "unexpected stablecoin asset"):
            download_rows(cfg, fetch=lambda _: {"data": [row("usdt", "2023-01-01", 1672619400)]})

    def test_rejects_precompletion_or_nonpositive_supply(self) -> None:
        cfg = CoinMetricsStablecoinSupplyConfig(output="unused.csv.gz", manifest="unused.json")
        with self.assertRaisesRegex(ValueError, "before the UTC day ended"):
            download_rows(cfg, fetch=lambda _: {"data": [row("usdt_eth", "2023-01-01", 1672534800)]})
        with self.assertRaisesRegex(ValueError, "SplyCur must be positive"):
            download_rows(
                cfg,
                fetch=lambda _: {"data": [row("usdt_eth", "2023-01-01", 1672619400, "0")]},
            )

    def test_run_writes_deterministic_gzip_and_lag_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "stablecoin.csv.gz"
            manifest = Path(tmp) / "manifest.json"
            cfg = CoinMetricsStablecoinSupplyConfig(
                output=str(output), manifest=str(manifest), start="2023-01-01", end="2023-01-01"
            )
            data = [row(asset, "2023-01-01", 1672619400, str(1_000_000 + i)) for i, asset in enumerate(ASSETS)]
            payload = {"data": data}
            with patch(
                "training.download_coinmetrics_stablecoin_supply_daily.urllib.request.urlopen",
                return_value=FakeResponse(payload),
            ):
                first = run(cfg)
            first_bytes = output.read_bytes()
            with patch(
                "training.download_coinmetrics_stablecoin_supply_daily.urllib.request.urlopen",
                return_value=FakeResponse(payload),
            ):
                second = run(cfg)
            self.assertEqual(first_bytes, output.read_bytes())
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["row_counts"], {asset: 1 for asset in ASSETS})
            self.assertEqual(first["excluded_composites"], ["usdt", "usdc"])
            with gzip.open(output, "rt") as fh:
                lines = fh.read().splitlines()
            self.assertEqual(len(lines), len(ASSETS) + 1)
            self.assertTrue(lines[0].startswith("asset,observation_date,available_at,supply"))


if __name__ == "__main__":
    unittest.main()
