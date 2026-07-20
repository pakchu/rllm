"""Freeze published checksums for the outcome-blind DLPD premium prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from training.build_binance_aggtrade_microstructure import (
    _fetch_bytes,
    expected_sha256,
)
from training.build_binance_stablecoin_quote_flow import _month_starts
from training.build_binance_um_premium_path import archive_url, checksum_url


SCHEMA_VERSION = 1
SYMBOLS = ("BTCUSDT", "BTCDOMUSDT")
INTERVAL = "1h"
START = date(2021, 7, 1)
END = date(2024, 1, 1)
SOURCE_DECISION = Path(
    "docs/btcdom-leverage-polarity-decomposition-source-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "ed402ef2a91e400b29b902154646637987318d57ab0543312f383f1193be3cf6"
)
DEFAULT_OUTPUT = Path(
    "data/binance_btcdom_premium_decomposition_2021_2023/"
    "archive_checksums.json"
)


@dataclass(frozen=True)
class Config:
    output: str = str(DEFAULT_OUTPUT)
    workers: int = 8
    retries: int = 5
    timeout_seconds: int = 60


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_keys() -> tuple[tuple[str, str], ...]:
    return tuple(
        (symbol, f"{month:%Y-%m}")
        for symbol in SYMBOLS
        for month in _month_starts(START, END)
    )


def _fetch_checksum(
    symbol: str,
    month: date,
    cfg: Config,
    *,
    fetcher: Callable[..., bytes],
) -> dict[str, Any]:
    url = checksum_url(symbol, INTERVAL, month)
    payload = fetcher(
        url,
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    return {
        "symbol": symbol,
        "month": f"{month:%Y-%m}",
        "interval": INTERVAL,
        "archive_url": archive_url(symbol, INTERVAL, month),
        "checksum_url": url,
        "archive_sha256": expected_sha256(payload),
    }


def build_inventory(
    cfg: Config = Config(),
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    if sha256_file(SOURCE_DECISION) != SOURCE_DECISION_SHA256:
        raise ValueError("DLPD source decision changed before checksum freeze")

    months = _month_starts(START, END)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(
                _fetch_checksum,
                symbol,
                month,
                cfg,
                fetcher=fetcher,
            ): (symbol, month)
            for symbol in SYMBOLS
            for month in months
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: (item["symbol"], item["month"]))

    observed = tuple((item["symbol"], item["month"]) for item in records)
    if observed != tuple(sorted(expected_keys())):
        raise ValueError("DLPD checksum inventory coverage changed")
    if len({item["checksum_url"] for item in records}) != len(records):
        raise ValueError("DLPD checksum inventory contains duplicate URLs")

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "DLPD published-checksum freeze before archive build",
        "as_of_date": "2026-07-20",
        "source_decision": str(SOURCE_DECISION),
        "source_decision_sha256": SOURCE_DECISION_SHA256,
        "config": {
            **asdict(cfg),
            "symbols": list(SYMBOLS),
            "interval": INTERVAL,
            "start_month": START.isoformat(),
            "end_month_exclusive": END.isoformat(),
        },
        "source_only": True,
        "outcomes_opened": False,
        "archive_bytes_downloaded": False,
        "post_2023_rows_requested": False,
        "records": records,
    }


def write_inventory(payload: dict[str, Any], output: str | Path) -> None:
    target = Path(output)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if target.exists() and target.read_bytes() != encoded:
        raise FileExistsError(f"existing frozen checksum inventory differs: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--workers", type=int, default=Config.workers)
    parser.add_argument("--retries", type=int, default=Config.retries)
    parser.add_argument("--timeout-seconds", type=int, default=Config.timeout_seconds)
    cfg = Config(**vars(parser.parse_args()))
    payload = build_inventory(cfg)
    write_inventory(payload, cfg.output)
    print(
        json.dumps(
            {"output": cfg.output, "records": len(payload["records"])},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
