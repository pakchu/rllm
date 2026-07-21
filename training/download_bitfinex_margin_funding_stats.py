"""Freeze pre-2024 Bitfinex public margin-funding statistics.

This source-only builder is physically bounded to 2020-2023.  It downloads
hourly ``fUSD`` and ``fBTC`` funding-stat snapshots from Bitfinex's public V2
API, keeps the official observation fields, assigns a conservative live
availability clock, and writes deterministic compressed artifacts.  It never
loads BTC prices, returns, funding paid on Binance, labels, positions, or PnL.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


ENDPOINT = "https://api-pub.bitfinex.com/v2/funding/stats/{symbol}/hist"
DOCUMENTATION = "https://docs.bitfinex.com/reference/rest-public-funding-stats"
API_REQUIREMENTS = "https://docs.bitfinex.com/docs/requirements-and-limitations"
PROTOCOL_VERSION = "bitfinex_margin_funding_stats_source_v1"
SOURCE_BUILDER = "training/download_bitfinex_margin_funding_stats.py"
SYMBOLS = ("fUSD", "fBTC")


@dataclass(frozen=True)
class Config:
    start: str = "2020-01-01T00:00:00Z"
    end_exclusive: str = "2024-01-01T00:00:00Z"
    output: str = "data/bitfinex_margin_funding_stats_2020_2023.csv.gz"
    raw_output: str = "data/bitfinex_margin_funding_stats_raw_2020_2023.jsonl.gz"
    manifest: str = (
        "results/bitfinex_margin_funding_stats_source_manifest_2026-07-20.json"
    )
    cache_dir: str = "data/bitfinex_margin_funding_stats_2020_2023_cache"
    request_pause_seconds: float = 4.05
    request_timeout_seconds: float = 30.0
    maximum_attempts: int = 7
    page_limit: int = 250
    maximum_gap_hours: float = 6.0
    keep_cache: bool = False


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise ValueError("timestamp must not be NaT")
    if timestamp.tzinfo is None:
        return cast(pd.Timestamp, timestamp.tz_localize("UTC"))
    return cast(pd.Timestamp, timestamp.tz_convert("UTC"))


def _milliseconds(value: pd.Timestamp) -> int:
    return int(value.timestamp() * 1_000)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _write_deterministic_gzip(payload: bytes, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as stream:
        stream.write(payload)
    compressed = buffer.getvalue()
    path.write_bytes(compressed)
    return hashlib.sha256(compressed).hexdigest()


def _write_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _request_json(url: str, cfg: Config) -> list[list[Any]]:
    last_error: Exception | None = None
    for attempt in range(cfg.maximum_attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "rllm-bitfinex-source-audit/1.0"},
            )
            with urllib.request.urlopen(
                request,
                timeout=cfg.request_timeout_seconds,
            ) as response:
                payload = json.loads(response.read())
            if not isinstance(payload, list):
                raise ValueError("Bitfinex funding-stat response must be a list")
            time.sleep(cfg.request_pause_seconds)
            return payload
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
            if attempt + 1 < cfg.maximum_attempts:
                time.sleep(max(cfg.request_pause_seconds, min(2.0**attempt, 60.0)))
    raise RuntimeError(f"Bitfinex request failed: {url}") from last_error


def parse_row(symbol: str, row: list[Any]) -> dict[str, Any]:
    if symbol not in SYMBOLS:
        raise ValueError(f"unsupported Bitfinex funding symbol: {symbol}")
    if not isinstance(row, list) or len(row) < 12:
        raise ValueError("Bitfinex funding-stat row must contain at least 12 fields")
    required = {
        0: "timestamp_ms",
        3: "frr",
        4: "average_period_days",
        7: "funding_amount",
        8: "funding_amount_used",
    }
    values: dict[str, float | int] = {}
    for index, name in required.items():
        value = row[index]
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Bitfinex funding-stat field is non-finite: {name}")
        values[name] = int(value) if index == 0 else float(value)
    below = row[11]
    if below is not None and not math.isfinite(float(below)):
        raise ValueError("Bitfinex funding-below-threshold field is non-finite")
    observed = pd.to_datetime(values["timestamp_ms"], unit="ms", utc=True)
    # Historical snapshots are normally timestamped near HH:05.  Waiting until
    # HH:15 makes the research clock reproducible under a live hourly poll and
    # prevents treating the provider timestamp as an instantaneous local read.
    available = observed.floor("h") + pd.Timedelta(minutes=15)
    return {
        "symbol": symbol,
        "observation_time": observed,
        "available_at": available,
        "timestamp_ms": int(values["timestamp_ms"]),
        "frr": float(values["frr"]),
        "average_period_days": float(values["average_period_days"]),
        "funding_amount": float(values["funding_amount"]),
        "funding_amount_used": float(values["funding_amount_used"]),
        "funding_below_threshold": None if below is None else float(below),
    }


def validate_page(
    symbol: str,
    payload: list[list[Any]],
    *,
    requested_start_ms: int,
    requested_end_ms: int,
    page_limit: int,
) -> list[dict[str, Any]]:
    if len(payload) > page_limit:
        raise ValueError("Bitfinex page exceeds the requested limit")
    parsed = [parse_row(symbol, row) for row in payload]
    timestamps = [int(item["timestamp_ms"]) for item in parsed]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("Bitfinex page contains duplicate timestamps")
    if timestamps != sorted(timestamps, reverse=True):
        raise ValueError("Bitfinex funding-stat page is not reverse chronological")
    if any(
        value < requested_start_ms or value > requested_end_ms for value in timestamps
    ):
        raise ValueError("Bitfinex page crossed the requested source boundary")
    return parsed


def _page_cache_path(cache_dir: Path, symbol: str, end_ms: int) -> Path:
    return cache_dir / symbol / f"end_{end_ms}.json"


def fetch_symbol(symbol: str, cfg: Config) -> tuple[list[list[Any]], dict[str, int]]:
    start = _utc(cfg.start)
    end_exclusive = _utc(cfg.end_exclusive)
    start_ms = _milliseconds(start)
    cursor_end_ms = _milliseconds(end_exclusive) - 1
    raw_rows: list[list[Any]] = []
    cache_dir = Path(cfg.cache_dir)
    network_requests = 0
    cached_pages = 0

    while cursor_end_ms >= start_ms:
        cache_path = _page_cache_path(cache_dir, symbol, cursor_end_ms)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            cached_pages += 1
        else:
            query = urllib.parse.urlencode(
                {
                    "start": start_ms,
                    "end": cursor_end_ms,
                    "limit": cfg.page_limit,
                }
            )
            url = ENDPOINT.format(symbol=symbol) + "?" + query
            payload = _request_json(url, cfg)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(payload, cache_path)
            network_requests += 1
        parsed = validate_page(
            symbol,
            payload,
            requested_start_ms=start_ms,
            requested_end_ms=cursor_end_ms,
            page_limit=cfg.page_limit,
        )
        if not parsed:
            break
        raw_rows.extend(payload)
        oldest_ms = min(int(item["timestamp_ms"]) for item in parsed)
        next_cursor = oldest_ms - 1
        if next_cursor >= cursor_end_ms:
            raise RuntimeError("Bitfinex pagination cursor did not move backward")
        cursor_end_ms = next_cursor

    return raw_rows, {
        "network_requests": network_requests,
        "cached_pages": cached_pages,
    }


def validate_frame(
    frame: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        raise ValueError("Bitfinex funding-stat source returned no rows")
    output = frame.sort_values(["symbol", "observation_time"]).reset_index(drop=True)
    if output.duplicated(["symbol", "timestamp_ms"]).any():
        raise ValueError("Bitfinex source contains duplicate symbol timestamps")
    start = _utc(cfg.start)
    end = _utc(cfg.end_exclusive)
    before_start = output["observation_time"].lt(start).to_numpy(dtype=bool)
    after_end = output["observation_time"].ge(end).to_numpy(dtype=bool)
    if bool(before_start.any()) or bool(after_end.any()):
        raise ValueError("Bitfinex source escaped the physical pre-2024 boundary")
    if set(output["symbol"]) != set(SYMBOLS):
        raise ValueError("Bitfinex source is missing a frozen funding symbol")
    numeric = output[
        ["frr", "average_period_days", "funding_amount", "funding_amount_used"]
    ]
    if not bool(np.isfinite(numeric.to_numpy(dtype=float)).all()):
        raise ValueError("Bitfinex source contains non-finite required values")
    if bool(output["funding_amount"].le(0.0).to_numpy(dtype=bool).any()):
        raise ValueError("Bitfinex total funding must be positive")
    if bool(output["funding_amount_used"].lt(0.0).to_numpy(dtype=bool).any()):
        raise ValueError("Bitfinex used funding must be non-negative")
    used_above_total = output["funding_amount_used"].gt(output["funding_amount"] + 1e-9)
    if bool(used_above_total.to_numpy(dtype=bool).any()):
        raise ValueError("Bitfinex used funding exceeds total funding")
    invalid_clock = output["available_at"].lt(output["observation_time"])
    if bool(invalid_clock.to_numpy(dtype=bool).any()):
        raise ValueError("Bitfinex conservative availability precedes observation")

    diagnostics: dict[str, Any] = {}
    for symbol_value, group in output.groupby("symbol", sort=True):
        symbol = str(symbol_value)
        times = [
            cast(pd.Timestamp, pd.Timestamp(value))
            for value in group["observation_time"].tolist()
        ]
        gaps = [
            (current.to_pydatetime() - previous.to_pydatetime()).total_seconds()
            / 3_600.0
            for previous, current in zip(times, times[1:])
        ]
        maximum_gap = max(gaps, default=0.0)
        if maximum_gap > cfg.maximum_gap_hours:
            raise ValueError(
                f"Bitfinex {symbol} source gap exceeds {cfg.maximum_gap_hours} hours"
            )
        expected_hours = pd.date_range(
            start.floor("h"), end.floor("h"), freq="h", inclusive="left"
        )
        observed_hours = pd.DatetimeIndex(
            [value.floor("h") for value in times]
        ).unique()
        diagnostics[symbol] = {
            "rows": int(len(group)),
            "first_observation": min(times).isoformat(),
            "last_observation": max(times).isoformat(),
            "missing_utc_hours": int(len(expected_hours.difference(observed_hours))),
            "maximum_gap_hours": maximum_gap,
        }
    return output, diagnostics


def _raw_jsonl(rows_by_symbol: dict[str, list[list[Any]]]) -> bytes:
    records: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for row in rows_by_symbol[symbol]:
            records.append({"symbol": symbol, "row": row})
    records.sort(key=lambda item: (item["symbol"], int(item["row"][0])))
    return b"".join(_canonical_json(item) for item in records)


def build(cfg: Config) -> dict[str, Any]:
    start = _utc(cfg.start)
    end = _utc(cfg.end_exclusive)
    if start != pd.Timestamp("2020-01-01T00:00:00Z"):
        raise ValueError("source contract requires the frozen 2020 start")
    if end != pd.Timestamp("2024-01-01T00:00:00Z"):
        raise ValueError("source contract forbids opening 2024 or later")
    if cfg.page_limit != 250:
        raise ValueError("source contract requires the documented maximum page size")
    if cfg.request_pause_seconds < 4.0:
        raise ValueError("source contract must respect the 15 requests/minute limit")

    rows_by_symbol: dict[str, list[list[Any]]] = {}
    request_diagnostics: dict[str, Any] = {}
    parsed_rows: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        raw_rows, requests = fetch_symbol(symbol, cfg)
        rows_by_symbol[symbol] = raw_rows
        request_diagnostics[symbol] = requests
        parsed_rows.extend(parse_row(symbol, row) for row in raw_rows)

    frame, coverage = validate_frame(pd.DataFrame(parsed_rows), cfg)
    output_path = Path(cfg.output)
    raw_path = Path(cfg.raw_output)
    csv_payload = frame.to_csv(index=False, lineterminator="\n").encode()
    output_sha = _write_deterministic_gzip(csv_payload, output_path)
    raw_sha = _write_deterministic_gzip(_raw_jsonl(rows_by_symbol), raw_path)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_contract": {
            "endpoint": ENDPOINT,
            "documentation": DOCUMENTATION,
            "api_requirements": API_REQUIREMENTS,
            "symbols": list(SYMBOLS),
            "start_inclusive": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "page_limit": cfg.page_limit,
            "minimum_request_pause_seconds": 4.0,
            "availability_rule": "floor(observation_time, 1h) + 15m",
            "outcomes_opened": False,
            "market_or_pnl_columns_loaded": False,
            "post_2023_rows_requested": False,
        },
        "coverage": coverage,
        "requests": request_diagnostics,
        "files": {
            "canonical": {
                "path": cfg.output,
                "sha256": output_sha,
                "rows": int(len(frame)),
                "columns": frame.columns.tolist(),
            },
            "raw": {
                "path": cfg.raw_output,
                "sha256": raw_sha,
                "rows": int(sum(len(rows) for rows in rows_by_symbol.values())),
            },
            "builder": {
                "path": SOURCE_BUILDER,
                "sha256": _sha256(SOURCE_BUILDER),
            },
        },
        "config": asdict(cfg),
    }
    _write_json_atomic(manifest, Path(cfg.manifest))

    if not cfg.keep_cache:
        for cache_path in Path(cfg.cache_dir).glob("*/*.json"):
            cache_path.unlink()
        for symbol_dir in Path(cfg.cache_dir).glob("*"):
            if symbol_dir.is_dir() and not any(symbol_dir.iterdir()):
                symbol_dir.rmdir()
        cache_root = Path(cfg.cache_dir)
        if cache_root.exists() and not any(cache_root.iterdir()):
            cache_root.rmdir()
    return manifest


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-pause-seconds", type=float, default=4.05)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-attempts", type=int, default=7)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    return Config(
        request_pause_seconds=args.request_pause_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        maximum_attempts=args.maximum_attempts,
        keep_cache=args.keep_cache,
    )


def main() -> None:
    manifest = build(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
