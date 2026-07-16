from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from training.export_emfx_daily_from_postgres import (
    SYMBOLS,
    EmfxDailyExportConfig,
    export_query,
    normalise_csv,
    run,
)


HEADER = "symbol,observation_date,observations,close,last_ts,max_updated_at\n"


def row(
    symbol: str,
    *,
    day: str = "2023-01-03",
    observations: int = 1_440,
    close: float = 1.0,
    last_ts: str | None = None,
) -> str:
    last = last_ts or f"{day} 23:59:00"
    return f"{symbol},{day} 00:00:00,{observations},{close},{last},2026-07-16 00:00:00\n"


class TestExportEmfxDailyFromPostgres(unittest.TestCase):
    def test_query_freezes_table_universe_dates_and_utc_day(self) -> None:
        cfg = EmfxDailyExportConfig(
            output="out.csv.gz",
            manifest="manifest.json",
            start="2019-01-01",
            end="2024-01-01",
        )
        query = export_query(cfg)
        self.assertIn("FROM bars_polygon", query)
        self.assertIn("ts AT TIME ZONE 'UTC'", query)
        self.assertIn("TIMESTAMPTZ '2019-01-01 00:00:00+00'", query)
        self.assertIn("TIMESTAMPTZ '2024-01-01 00:00:00+00'", query)
        for symbol in SYMBOLS:
            self.assertIn(f"'{symbol}'", query)
        with self.assertRaisesRegex(ValueError, "audited bars_polygon"):
            export_query(EmfxDailyExportConfig(output="x", manifest="y", table="other"))

    def test_normalise_sorts_and_rejects_duplicate_or_invalid_rows(self) -> None:
        text = HEADER + row("USDMXN", close=17.0) + row("USDAUD", close=1.5)
        rows = normalise_csv(text)
        self.assertEqual([item["symbol"] for item in rows], ["USDAUD", "USDMXN"])
        with self.assertRaisesRegex(ValueError, "duplicate EM-FX"):
            normalise_csv(text + row("USDMXN", close=18.0))
        with self.assertRaisesRegex(ValueError, "outside its UTC day"):
            normalise_csv(HEADER + row("USDAUD", last_ts="2023-01-04 00:00:00"))
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            normalise_csv(HEADER + row("USDAUD", close=0.0))

    def test_normalise_accepts_postgres_variable_fractional_seconds(self) -> None:
        text = HEADER + row("USDAUD").replace(
            "2026-07-16 00:00:00", "2026-07-16 00:00:00.39068"
        )
        rows = normalise_csv(text)
        self.assertEqual(rows[0]["max_updated_at"], "2026-07-16 00:00:00")

    def test_run_writes_deterministic_gzip_and_redacted_manifest(self) -> None:
        payload = HEADER + "".join(
            row(symbol, close=1.0 + index) for index, symbol in enumerate(reversed(SYMBOLS))
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "emfx.csv.gz"
            manifest = Path(tmp) / "manifest.json"
            cfg = EmfxDailyExportConfig(
                output=str(output),
                manifest=str(manifest),
                env_file="/secret/database.env",
                start="2023-01-03",
                end="2023-01-04",
            )
            first = run(cfg, query_runner=lambda _cfg, _query: payload)
            first_bytes = output.read_bytes()
            second = run(cfg, query_runner=lambda _cfg, _query: payload)
            self.assertEqual(first_bytes, output.read_bytes())
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["row_counts"], {symbol: 1 for symbol in SYMBOLS})
            written = json.loads(manifest.read_text())
            self.assertEqual(written["config"]["env_file"], "<redacted>")
            self.assertNotIn("secret", manifest.read_text())
            with gzip.open(output, "rt") as fh:
                lines = fh.read().splitlines()
            self.assertEqual(len(lines), len(SYMBOLS) + 1)
            self.assertEqual([line.split(",", 1)[0] for line in lines[1:]], sorted(SYMBOLS))


if __name__ == "__main__":
    unittest.main()
