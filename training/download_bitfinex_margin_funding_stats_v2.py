"""Build the BFMWD Bitfinex source with a fail-closed availability clock.

Transport v2 changes only the availability timestamp for the rare historical
row observed after HH:15.  All fields, symbols, date bounds, pagination, cache,
and source validation remain delegated to the hash-bound v1 builder.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

from training import download_bitfinex_margin_funding_stats as v1


PROTOCOL_VERSION = "bitfinex_margin_funding_stats_source_v2"
SOURCE_BUILDER = "training/download_bitfinex_margin_funding_stats_v2.py"
TRANSPORT_AMENDMENT = (
    "results/bitfinex_margin_funding_stats_transport_v2_amendment_2026-07-20.json"
)
Config = v1.Config
SYMBOLS = v1.SYMBOLS


def conservative_availability(observed: pd.Timestamp) -> pd.Timestamp:
    regular_poll = observed.floor("h") + pd.Timedelta(minutes=15)
    observed_bar = observed.ceil("5min")
    return cast(pd.Timestamp, max(regular_poll, observed_bar))


def parse_row(symbol: str, row: list[Any]) -> dict[str, Any]:
    parsed = v1.parse_row(symbol, row)
    observed = pd.Timestamp(parsed["observation_time"])
    if observed is pd.NaT:
        raise ValueError("Bitfinex observation timestamp must not be NaT")
    parsed["available_at"] = conservative_availability(cast(pd.Timestamp, observed))
    return parsed


def build(cfg: Config) -> dict[str, Any]:
    start = v1._utc(cfg.start)
    end = v1._utc(cfg.end_exclusive)
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
        raw_rows, requests = v1.fetch_symbol(symbol, cfg)
        rows_by_symbol[symbol] = raw_rows
        request_diagnostics[symbol] = requests
        parsed_rows.extend(parse_row(symbol, row) for row in raw_rows)

    frame, coverage = v1.validate_frame(pd.DataFrame(parsed_rows), cfg)
    regular = frame["observation_time"].dt.floor("h") + pd.Timedelta(minutes=15)
    late_rows = int(frame["available_at"].gt(regular).sum())
    output_path = Path(cfg.output)
    raw_path = Path(cfg.raw_output)
    csv_payload = frame.to_csv(index=False, lineterminator="\n").encode()
    output_sha = v1._write_deterministic_gzip(csv_payload, output_path)
    raw_sha = v1._write_deterministic_gzip(v1._raw_jsonl(rows_by_symbol), raw_path)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "transport_amendment": {
            "path": TRANSPORT_AMENDMENT,
            "sha256": v1._sha256(TRANSPORT_AMENDMENT),
            "v1_builder": v1.SOURCE_BUILDER,
            "v1_builder_sha256": v1._sha256(v1.SOURCE_BUILDER),
        },
        "source_contract": {
            "endpoint": v1.ENDPOINT,
            "documentation": v1.DOCUMENTATION,
            "api_requirements": v1.API_REQUIREMENTS,
            "symbols": list(SYMBOLS),
            "start_inclusive": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "page_limit": cfg.page_limit,
            "minimum_request_pause_seconds": 4.0,
            "availability_rule": (
                "max(floor(observation_time, 1h) + 15m, "
                "ceil(observation_time, 5m))"
            ),
            "late_observation_fallback_rows": late_rows,
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
                "sha256": v1._sha256(SOURCE_BUILDER),
            },
        },
        "config": asdict(cfg),
    }
    v1._write_json_atomic(manifest, Path(cfg.manifest))

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
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
