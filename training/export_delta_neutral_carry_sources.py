"""Export deterministic Binance spot bars and funding for delta-neutral carry research."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from training.export_emfx_daily_from_postgres import _parse_pg_timestamp, load_env_file


SPOT_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "observations",
    "last_ts",
    "max_updated_at",
)
FUNDING_COLUMNS = ("date", "funding_rate", "mark_price", "max_updated_at")


@dataclass(frozen=True)
class CarrySourceExportConfig:
    spot_output: str
    funding_output: str
    manifest: str
    start: str = "2020-01-01"
    end: str = "2026-06-02"
    env_file: str = ".env"
    psql_binary: str = "psql"


def _validate_iso_date(value: str, name: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD, got {value!r}") from exc
    return parsed.strftime("%Y-%m-%d")


def _bounds(cfg: CarrySourceExportConfig) -> tuple[str, str]:
    start = _validate_iso_date(cfg.start, "start")
    end = _validate_iso_date(cfg.end, "end")
    if end <= start:
        raise ValueError("end must be after start")
    return start, end


def spot_query(cfg: CarrySourceExportConfig) -> str:
    start, end = _bounds(cfg)
    return f"""COPY (
SELECT
    date_bin('5 minutes', ts, TIMESTAMPTZ '1970-01-01 00:00:00+00') AT TIME ZONE 'UTC' AS date,
    (array_agg(open ORDER BY ts ASC))[1]::double precision AS open,
    max(high)::double precision AS high,
    min(low)::double precision AS low,
    (array_agg(close ORDER BY ts DESC))[1]::double precision AS close,
    count(*)::bigint AS observations,
    max(ts) AT TIME ZONE 'UTC' AS last_ts,
    max(updated_at) AT TIME ZONE 'UTC' AS max_updated_at
FROM bars_binance_spot
WHERE symbol = 'BTCUSDT'
  AND interval = '1m'
  AND ts >= TIMESTAMPTZ '{start} 00:00:00+00'
  AND ts < TIMESTAMPTZ '{end} 00:00:00+00'
GROUP BY date_bin('5 minutes', ts, TIMESTAMPTZ '1970-01-01 00:00:00+00')
ORDER BY date
) TO STDOUT WITH (FORMAT CSV, HEADER TRUE);"""


def funding_query(cfg: CarrySourceExportConfig) -> str:
    start, end = _bounds(cfg)
    return f"""COPY (
SELECT
    funding_time AT TIME ZONE 'UTC' AS date,
    funding_rate::double precision AS funding_rate,
    NULLIF(mark_price, 0)::double precision AS mark_price,
    updated_at AT TIME ZONE 'UTC' AS max_updated_at
FROM funding_rates_binance
WHERE symbol = 'BTCUSDT'
  AND funding_time >= TIMESTAMPTZ '{start} 00:00:00+00'
  AND funding_time < TIMESTAMPTZ '{end} 00:00:00+00'
ORDER BY funding_time
) TO STDOUT WITH (FORMAT CSV, HEADER TRUE);"""


def run_psql_query(cfg: CarrySourceExportConfig, query: str) -> str:
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


def _read_rows(text: str, columns: tuple[str, ...]) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != columns:
        raise ValueError(f"unexpected psql columns: {reader.fieldnames!r}")
    return [dict(row) for row in reader]


def normalise_spot_csv(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in _read_rows(text, SPOT_COLUMNS):
        date = _parse_pg_timestamp(raw["date"])
        last_ts = _parse_pg_timestamp(raw["last_ts"])
        updated_at = _parse_pg_timestamp(raw["max_updated_at"])
        if date.minute % 5 or date.second or date.microsecond:
            raise ValueError(f"spot bar is not aligned to a five-minute boundary: {date}")
        if not (date <= last_ts < date + timedelta(minutes=5)):
            raise ValueError(f"last spot minute falls outside its five-minute bin: {date} {last_ts}")
        key = date.isoformat(sep=" ", timespec="seconds")
        if key in seen:
            raise ValueError(f"duplicate spot five-minute bar: {key}")
        seen.add(key)
        values = {name: float(raw[name]) for name in ("open", "high", "low", "close")}
        if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
            raise ValueError("spot OHLC must be positive and finite")
        if values["high"] < max(values["open"], values["close"], values["low"]):
            raise ValueError("spot high is inconsistent with OHLC")
        if values["low"] > min(values["open"], values["close"], values["high"]):
            raise ValueError("spot low is inconsistent with OHLC")
        observations = int(raw["observations"])
        if observations < 1 or observations > 5:
            raise ValueError("spot observations must be in [1, 5]")
        rows.append(
            {
                "date": key,
                "open": format(values["open"], ".15g"),
                "high": format(values["high"], ".15g"),
                "low": format(values["low"], ".15g"),
                "close": format(values["close"], ".15g"),
                "observations": str(observations),
                "last_ts": last_ts.isoformat(sep=" ", timespec="seconds"),
                "max_updated_at": updated_at.isoformat(sep=" ", timespec="seconds"),
            }
        )
    return sorted(rows, key=lambda row: row["date"])


def normalise_funding_csv(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in _read_rows(text, FUNDING_COLUMNS):
        date = _parse_pg_timestamp(raw["date"])
        updated_at = _parse_pg_timestamp(raw["max_updated_at"])
        key = date.isoformat(sep=" ", timespec="microseconds")
        if key in seen:
            raise ValueError(f"duplicate funding event: {key}")
        seen.add(key)
        rate = float(raw["funding_rate"])
        mark_text = raw["mark_price"].strip()
        mark = float(mark_text) if mark_text else None
        if not math.isfinite(rate):
            raise ValueError("funding rate must be finite")
        if mark is not None and (not math.isfinite(mark) or mark <= 0.0):
            raise ValueError("funding mark price must be positive and finite")
        rows.append(
            {
                "date": key,
                "funding_rate": format(rate, ".15g"),
                "mark_price": "" if mark is None else format(mark, ".15g"),
                "max_updated_at": updated_at.isoformat(sep=" ", timespec="seconds"),
            }
        )
    return sorted(rows, key=lambda row: row["date"])


def _write_csv_gz(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as wrapper:
                writer = csv.DictWriter(wrapper, fieldnames=list(columns), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    cfg: CarrySourceExportConfig,
    *,
    query_runner: Callable[[CarrySourceExportConfig, str], str] = run_psql_query,
) -> dict[str, object]:
    spot_sql = spot_query(cfg)
    funding_sql = funding_query(cfg)
    spot_rows = normalise_spot_csv(query_runner(cfg, spot_sql))
    funding_rows = normalise_funding_csv(query_runner(cfg, funding_sql))
    if not spot_rows or not funding_rows:
        raise RuntimeError("carry source export is empty")
    spot_path = Path(cfg.spot_output)
    funding_path = Path(cfg.funding_output)
    _write_csv_gz(spot_path, SPOT_COLUMNS, spot_rows)
    _write_csv_gz(funding_path, FUNDING_COLUMNS, funding_rows)
    manifest: dict[str, object] = {
        "config": {**asdict(cfg), "env_file": "<redacted>"},
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "spot": "bars_binance_spot BTCUSDT 1m aggregated to 5m",
            "funding": "funding_rates_binance BTCUSDT",
        },
        "query_sha256": {
            "spot": hashlib.sha256(spot_sql.encode()).hexdigest(),
            "funding": hashlib.sha256(funding_sql.encode()).hexdigest(),
        },
        "outputs": {
            "spot": {
                "path": str(spot_path),
                "rows": len(spot_rows),
                "range": [spot_rows[0]["date"], spot_rows[-1]["date"]],
                "partial_five_minute_bars": sum(int(row["observations"]) < 5 for row in spot_rows),
                "sha256": _sha256(spot_path),
            },
            "funding": {
                "path": str(funding_path),
                "rows": len(funding_rows),
                "range": [funding_rows[0]["date"], funding_rows[-1]["date"]],
                "missing_mark_prices": sum(not row["mark_price"] for row in funding_rows),
                "sha256": _sha256(funding_path),
            },
        },
        "database_snapshot_is_point_in_time": False,
        "revision_note": (
            "historical rows were backfilled into PostgreSQL; exchange timestamps define semantic "
            "availability, source-prefix hashes detect later revisions, and live forward proof is required"
        ),
        "missing_mark_price_policy": (
            "preserve unavailable historical funding mark prices as empty; the simulator must use "
            "the last fully completed USD-M futures five-minute close strictly before funding_time"
        ),
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return manifest


def parse_args() -> CarrySourceExportConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spot-output", required=True)
    parser.add_argument("--funding-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--start", default=CarrySourceExportConfig.start)
    parser.add_argument("--end", default=CarrySourceExportConfig.end)
    parser.add_argument("--env-file", default=CarrySourceExportConfig.env_file)
    parser.add_argument("--psql-binary", default=CarrySourceExportConfig.psql_binary)
    return CarrySourceExportConfig(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
