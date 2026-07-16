"""Export a deterministic daily EM-FX close panel from the live PostgreSQL mirror.

The exporter uses ``psql`` directly so the research environment does not need a
new Python database dependency. Credentials are loaded from an env file but are
never written to the manifest. The exported day is complete UTC market data;
research deliberately waits until 00:05 UTC on the following day before using it.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable


SYMBOLS = ("USDAUD", "USDCNY", "USDHKD", "USDINR", "USDMXN")
OUTPUT_COLUMNS = (
    "symbol",
    "observation_date",
    "observations",
    "close",
    "last_ts",
    "max_updated_at",
)


@dataclass(frozen=True)
class EmfxDailyExportConfig:
    output: str
    manifest: str
    start: str = "2019-01-01"
    end: str = "2026-06-02"
    env_file: str = ".env"
    table: str = "bars_polygon"
    psql_binary: str = "psql"


def _validate_iso_date(value: str, name: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD, got {value!r}") from exc
    return parsed.strftime("%Y-%m-%d")


def load_env_file(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value
    required = ("PG_USER", "PG_PASSWORD", "PG_HOST", "PG_PORT", "PG_DB_NAME")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(f"env file missing PostgreSQL keys: {missing}")
    return values


def _parse_pg_timestamp(value: str) -> datetime:
    """Parse PostgreSQL timestamps on Python versions strict about fractional width."""
    text = value.strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        match = re.fullmatch(r"(.+?)\.(\d+)([+-]\d\d(?::?\d\d)?)?", text)
        if match is None:
            raise
        base, fraction, offset = match.groups()
        normalized = f"{base}.{fraction[:6].ljust(6, '0')}{offset or ''}"
        return datetime.fromisoformat(normalized)


def export_query(cfg: EmfxDailyExportConfig) -> str:
    start = _validate_iso_date(cfg.start, "start")
    end = _validate_iso_date(cfg.end, "end")
    if end <= start:
        raise ValueError("end must be after start")
    if cfg.table != "bars_polygon":
        raise ValueError("only the audited bars_polygon table is allowed")
    symbols = ",".join(f"'{symbol}'" for symbol in SYMBOLS)
    return f"""COPY (
SELECT
    symbol,
    date_trunc('day', ts AT TIME ZONE 'UTC') AS observation_date,
    count(*)::bigint AS observations,
    (array_agg(close ORDER BY ts DESC))[1]::double precision AS close,
    max(ts) AT TIME ZONE 'UTC' AS last_ts,
    max(updated_at) AT TIME ZONE 'UTC' AS max_updated_at
FROM {cfg.table}
WHERE symbol IN ({symbols})
  AND ts >= TIMESTAMPTZ '{start} 00:00:00+00'
  AND ts < TIMESTAMPTZ '{end} 00:00:00+00'
GROUP BY symbol, date_trunc('day', ts AT TIME ZONE 'UTC')
ORDER BY observation_date, symbol
) TO STDOUT WITH (FORMAT CSV, HEADER TRUE);"""


def run_psql_query(cfg: EmfxDailyExportConfig, query: str) -> str:
    values = load_env_file(cfg.env_file)
    env = os.environ.copy()
    env["PGPASSWORD"] = values["PG_PASSWORD"]
    env["PGTZ"] = "UTC"
    command = [
        cfg.psql_binary,
        "--no-psqlrc",
        "--set",
        "ON_ERROR_STOP=1",
        "--host",
        values["PG_HOST"],
        "--port",
        values["PG_PORT"],
        "--username",
        values["PG_USER"],
        "--dbname",
        values["PG_DB_NAME"],
        "--command",
        query,
    ]
    completed = subprocess.run(
        command,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def normalise_csv(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
        raise ValueError(f"unexpected psql columns: {reader.fieldnames!r}")
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for raw in reader:
        symbol = str(raw["symbol"]).strip().upper()
        if symbol not in SYMBOLS:
            raise ValueError(f"unexpected EM-FX symbol: {symbol!r}")
        observed = _parse_pg_timestamp(raw["observation_date"])
        last_ts = _parse_pg_timestamp(raw["last_ts"])
        updated_at = _parse_pg_timestamp(raw["max_updated_at"])
        count = int(raw["observations"])
        close = float(raw["close"])
        if count <= 0:
            raise ValueError("observations must be positive")
        if not math.isfinite(close) or close <= 0.0:
            raise ValueError("daily close must be positive and finite")
        if not (observed <= last_ts < observed + timedelta(days=1)):
            raise ValueError(f"last quote falls outside its UTC day: {symbol} {observed} {last_ts}")
        row = {
            "symbol": symbol,
            "observation_date": observed.isoformat(sep=" ", timespec="seconds"),
            "observations": str(count),
            "close": format(close, ".15g"),
            "last_ts": last_ts.isoformat(sep=" ", timespec="seconds"),
            "max_updated_at": updated_at.isoformat(sep=" ", timespec="seconds"),
        }
        key = (row["observation_date"], symbol)
        if key in by_key:
            raise ValueError(f"duplicate EM-FX UTC day/symbol row: {key}")
        by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def _write_csv_gz(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as wrapper:
                writer = csv.DictWriter(
                    wrapper, fieldnames=list(OUTPUT_COLUMNS), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    cfg: EmfxDailyExportConfig,
    *,
    query_runner: Callable[[EmfxDailyExportConfig, str], str] = run_psql_query,
) -> dict[str, object]:
    query = export_query(cfg)
    rows = normalise_csv(query_runner(cfg, query))
    observed_symbols = {row["symbol"] for row in rows}
    if observed_symbols != set(SYMBOLS):
        raise RuntimeError(
            f"EM-FX source universe mismatch: expected={sorted(SYMBOLS)} observed={sorted(observed_symbols)}"
        )
    output = Path(cfg.output)
    _write_csv_gz(output, rows)
    counts = {symbol: sum(row["symbol"] == symbol for row in rows) for symbol in SYMBOLS}
    manifest: dict[str, object] = {
        "config": {**asdict(cfg), "env_file": "<redacted>"},
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": "Polygon minute FX bars mirrored in PostgreSQL",
        "source_table": cfg.table,
        "fixed_symbol_universe": list(SYMBOLS),
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "output": str(output),
        "rows": len(rows),
        "row_counts": counts,
        "row_range": {
            "start": rows[0]["observation_date"] if rows else None,
            "end": rows[-1]["observation_date"] if rows else None,
        },
        "columns": list(OUTPUT_COLUMNS),
        "sha256": _sha256(output),
        "semantic_availability_rule": "UTC day d is usable no earlier than d+1 00:05 UTC",
        "database_snapshot_is_point_in_time": False,
        "revision_note": (
            "database ingestion timestamps are later backfills; the fixed-panel market quotes are "
            "timestamped observations but this export is not a point-in-time database snapshot"
        ),
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return manifest


def parse_args() -> EmfxDailyExportConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--start", default=EmfxDailyExportConfig.start)
    parser.add_argument("--end", default=EmfxDailyExportConfig.end)
    parser.add_argument("--env-file", default=EmfxDailyExportConfig.env_file)
    parser.add_argument("--table", default=EmfxDailyExportConfig.table)
    parser.add_argument("--psql-binary", default=EmfxDailyExportConfig.psql_binary)
    return EmfxDailyExportConfig(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
