from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from training.export_delta_neutral_carry_sources import (
    CarrySourceExportConfig,
    funding_query,
    normalise_funding_csv,
    normalise_spot_csv,
    perp_query,
    run,
    spot_query,
)


MARKET_HEADER = "date,open,high,low,close,max_updated_at\n"
FUNDING_HEADER = "date,funding_rate,mark_price,max_updated_at\n"


def market_row(date: str = "2023-01-01 00:00:00") -> str:
    return f"{date},100,102,99,101,2026-07-16 00:00:00.39068\n"


def funding_row(date: str = "2023-01-01 00:00:00", rate: float = 0.0001) -> str:
    return f"{date},{rate},101,2026-07-16 00:00:00.39068\n"


class TestExportDeltaNeutralCarrySources(unittest.TestCase):
    def test_queries_freeze_tables_symbol_interval_and_cutoff(self) -> None:
        cfg = CarrySourceExportConfig(
            perp_output="perp", spot_output="spot", funding_output="funding", manifest="manifest", end="2024-01-01"
        )
        perp = perp_query(cfg)
        spot = spot_query(cfg)
        funding = funding_query(cfg)
        self.assertIn("FROM bars_binance_spot", spot)
        self.assertIn("symbol = 'BTCUSDT'", spot)
        self.assertIn("interval = '1m'", spot)
        self.assertIn("FROM bars_binance", perp)
        self.assertIn("interval = '1m'", perp)
        self.assertIn("FROM funding_rates_binance", funding)
        self.assertIn("TIMESTAMPTZ '2024-01-01 00:00:00+00'", spot)
        self.assertIn("TIMESTAMPTZ '2024-01-01 00:00:00+00'", funding)

    def test_normalisers_fail_closed_on_duplicates_and_bad_market_rows(self) -> None:
        self.assertEqual(normalise_spot_csv(MARKET_HEADER + market_row())[0]["close"], "101")
        with self.assertRaisesRegex(ValueError, "duplicate spot"):
            normalise_spot_csv(MARKET_HEADER + market_row() + market_row())
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            normalise_spot_csv(MARKET_HEADER + market_row().replace(",102,99,", ",100,101,"))
        with self.assertRaisesRegex(ValueError, "one-minute boundary"):
            normalise_spot_csv(
                MARKET_HEADER + market_row("2023-01-01 00:00:01")
            )
        self.assertEqual(
            normalise_funding_csv(FUNDING_HEADER + funding_row())[0]["funding_rate"], "0.0001"
        )
        missing_mark = normalise_funding_csv(
            FUNDING_HEADER + funding_row().replace(",101,", ",,")
        )
        self.assertEqual(missing_mark[0]["mark_price"], "")
        with self.assertRaisesRegex(ValueError, "duplicate funding"):
            normalise_funding_csv(FUNDING_HEADER + funding_row() + funding_row())

    def test_run_writes_deterministic_outputs_and_redacts_credentials(self) -> None:
        market_payload = MARKET_HEADER + market_row()
        funding_payload = FUNDING_HEADER + funding_row()

        def runner(_cfg: CarrySourceExportConfig, query: str) -> str:
            return funding_payload if "funding_rates_binance" in query else market_payload

        with tempfile.TemporaryDirectory() as tmp:
            perp = Path(tmp) / "perp.csv.gz"
            spot = Path(tmp) / "spot.csv.gz"
            funding = Path(tmp) / "funding.csv.gz"
            manifest = Path(tmp) / "manifest.json"
            cfg = CarrySourceExportConfig(
                perp_output=str(perp),
                spot_output=str(spot),
                funding_output=str(funding),
                manifest=str(manifest),
                env_file="/secret/database.env",
                start="2023-01-01",
                end="2023-01-02",
            )
            first = run(cfg, query_runner=runner)
            perp_bytes = perp.read_bytes()
            spot_bytes = spot.read_bytes()
            funding_bytes = funding.read_bytes()
            second = run(cfg, query_runner=runner)
            self.assertEqual(perp_bytes, perp.read_bytes())
            self.assertEqual(spot_bytes, spot.read_bytes())
            self.assertEqual(funding_bytes, funding.read_bytes())
            self.assertEqual(first["outputs"]["spot"]["sha256"], second["outputs"]["spot"]["sha256"])
            self.assertEqual(first["outputs"]["perp"]["sha256"], second["outputs"]["perp"]["sha256"])
            self.assertEqual(first["outputs"]["funding"]["missing_mark_prices"], 0)
            written = json.loads(manifest.read_text())
            self.assertEqual(written["config"]["env_file"], "<redacted>")
            self.assertNotIn("secret", manifest.read_text())
            with gzip.open(spot, "rt") as fh:
                self.assertEqual(len(fh.read().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
