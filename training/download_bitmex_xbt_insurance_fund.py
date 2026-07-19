"""Download and audit the daily BitMEX XBt insurance-fund history.

Official endpoint:
https://docs.bitmex.com/api-explorer/get-insurances

The downloader is intentionally narrow.  It requests one frozen currency,
normalizes the legacy ``currency`` and current ``symbol`` response keys, and
refuses gaps, duplicates, off-noon timestamps, or non-positive balances.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ENDPOINT = "https://www.bitmex.com/api/v1/insurance"
OFFICIAL_DOCS = "https://docs.bitmex.com/api-explorer/get-insurances"


@dataclass(frozen=True)
class Config:
    output_csv: str = "data/bitmex_xbt_insurance_fund_2018_2022.csv.gz"
    manifest_output: str = (
        "results/bitmex_xbt_insurance_fund_source_manifest_2026-07-20.json"
    )
    start: str = "2018-01-01"
    end_exclusive: str = "2023-01-01"
    currency: str = "XBt"
    page_size: int = 500
    timeout_sec: float = 30.0


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _api_time(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def get_json(params: dict[str, Any], *, timeout_sec: float) -> list[dict[str, Any]]:
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "rllm-private-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("BitMEX insurance response is not a list")
    return payload


def _currency(row: dict[str, Any]) -> str:
    legacy = row.get("currency")
    current = row.get("symbol")
    if legacy is not None and current is not None and legacy != current:
        raise ValueError("BitMEX insurance row has conflicting currency keys")
    value = legacy if legacy is not None else current
    if not isinstance(value, str) or not value:
        raise ValueError("BitMEX insurance row has no currency/symbol")
    return value


def download(
    cfg: Config,
    *,
    fetch: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 1 <= cfg.page_size <= 500:
        raise ValueError("BitMEX insurance page_size must be in [1, 500]")
    start = _utc(cfg.start)
    end = _utc(cfg.end_exclusive)
    if start >= end:
        raise ValueError("source interval must be non-empty")
    end_inclusive = end - pd.Timedelta(days=1)
    if end_inclusive.hour or end_inclusive.minute or end_inclusive.second:
        raise ValueError("end_exclusive must be a UTC calendar-day boundary")
    end_inclusive += pd.Timedelta(hours=12)
    request_start = start + pd.Timedelta(hours=12)
    fetch = fetch or (
        lambda params: get_json(params, timeout_sec=cfg.timeout_sec)
    )

    rows: list[dict[str, Any]] = []
    offset = 0
    page_lengths: list[int] = []
    while True:
        params = {
            "currency": cfg.currency,
            "count": cfg.page_size,
            "start": offset,
            "reverse": "false",
            "startTime": _api_time(request_start),
            "endTime": _api_time(end_inclusive),
        }
        batch = fetch(params)
        page_lengths.append(len(batch))
        if len(batch) > cfg.page_size:
            raise RuntimeError("BitMEX insurance page exceeds frozen page size")
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < cfg.page_size:
            break
        offset += len(batch)

    if not rows:
        raise RuntimeError("BitMEX insurance source returned no rows")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if _currency(row) != cfg.currency:
            raise ValueError("BitMEX insurance currency filter returned another asset")
        timestamp = _utc(str(row["timestamp"]))
        balance = row.get("walletBalance")
        if isinstance(balance, bool) or not isinstance(balance, int):
            raise ValueError("BitMEX walletBalance must be an integer")
        normalized.append(
            {
                "date": timestamp,
                "wallet_balance_satoshi": balance,
            }
        )

    frame = pd.DataFrame.from_records(normalized).sort_values("date")
    frame = frame[
        frame["date"].ge(request_start)
        & frame["date"].lt(end)
    ].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("BitMEX insurance rows are outside frozen interval")
    if frame["date"].duplicated().any():
        raise RuntimeError("BitMEX insurance source contains duplicate timestamps")
    expected = pd.date_range(request_start, end_inclusive, freq="1D", tz="UTC")
    if not frame["date"].reset_index(drop=True).equals(
        pd.Series(expected, name="date")
    ):
        missing = expected.difference(pd.DatetimeIndex(frame["date"]))
        extra = pd.DatetimeIndex(frame["date"]).difference(expected)
        raise RuntimeError(
            "BitMEX insurance source is not a complete daily 12:00 UTC grid: "
            f"missing={len(missing)} extra={len(extra)}"
        )
    if frame["wallet_balance_satoshi"].le(0).any():
        raise RuntimeError("BitMEX XBt insurance balance must stay positive")

    audit = {
        "endpoint": ENDPOINT,
        "official_docs": OFFICIAL_DOCS,
        "request_pages": len(page_lengths),
        "page_lengths": page_lengths,
        "rows_received": len(rows),
        "rows_selected": len(frame),
        "expected_days": len(expected),
        "complete_daily_noon_utc_grid": True,
        "start": str(frame["date"].iloc[0]),
        "end": str(frame["date"].iloc[-1]),
        "response_currency": cfg.currency,
    }
    frame["date"] = frame["date"].dt.tz_convert(None)
    return frame, audit


def run(cfg: Config) -> dict[str, Any]:
    frame, audit = download(cfg)
    output = Path(cfg.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, compression="infer")
    core = {
        "protocol_version": "bitmex_xbt_insurance_source_v1",
        "config": asdict(cfg),
        "source_audit": audit,
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
            "bytes": output.stat().st_size,
            "columns": list(frame.columns),
        },
        "data_use": (
            "private internal research; raw source is ignored and is not "
            "redistributed by the repository"
        ),
    }
    manifest = {
        **core,
        "manifest_hash": _canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", default=Config.output_csv)
    parser.add_argument("--manifest-output", default=Config.manifest_output)
    parser.add_argument("--start", default=Config.start)
    parser.add_argument("--end-exclusive", default=Config.end_exclusive)
    parser.add_argument("--currency", default=Config.currency)
    parser.add_argument("--page-size", type=int, default=Config.page_size)
    parser.add_argument("--timeout-sec", type=float, default=Config.timeout_sec)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
