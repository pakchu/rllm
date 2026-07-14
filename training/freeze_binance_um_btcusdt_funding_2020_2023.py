"""Download and freeze official BTCUSDT USD-M realized funding through 2023."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable


API_ENDPOINT = "https://fapi.binance.com/fapi/v1/fundingRate"
API_DOCUMENTATION = (
    "https://developers.binance.com/en/docs/catalog/"
    "core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data"
    "#get-funding-rate-history"
)
FUNDING_EXPLANATION = (
    "https://academy.binance.com/en/articles/what-are-funding-rates-in-crypto-markets"
)
START_MS = 1_577_836_800_000  # 2020-01-01T00:00:00.000Z, inclusive
END_MS = 1_704_067_199_999  # 2023-12-31T23:59:59.999Z, inclusive
COLUMNS = (
    "funding_time_ms",
    "funding_time_utc",
    "symbol",
    "funding_rate",
    "mark_price",
)
OpenJson = Callable[[str], Any]


@dataclass(frozen=True)
class FreezeConfig:
    symbol: str = "BTCUSDT"
    start_ms: int = START_MS
    end_ms: int = END_MS
    limit: int = 1_000
    timeout_seconds: float = 30.0
    retry_attempts: int = 5
    retry_backoff_seconds: float = 1.0
    output: str = "results/binance_um_btcusdt_realized_funding_2020_2023.csv"
    manifest: str = (
        "results/binance_um_btcusdt_realized_funding_2020_2023_manifest.json"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_iso(milliseconds: int) -> str:
    value = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
        milliseconds=milliseconds
    )
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _default_open_json(url: str, *, timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "rllm-frozen-funding/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def _request_page(
    cfg: FreezeConfig,
    *,
    start_ms: int,
    open_json: OpenJson | None,
) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "symbol": cfg.symbol,
            "startTime": start_ms,
            "endTime": cfg.end_ms,
            "limit": cfg.limit,
        }
    )
    url = f"{API_ENDPOINT}?{params}"
    opener = open_json or (
        lambda value: _default_open_json(value, timeout=cfg.timeout_seconds)
    )
    last_error: Exception | None = None
    for attempt in range(cfg.retry_attempts):
        try:
            payload = opener(url)
            if not isinstance(payload, list):
                raise ValueError("Binance funding response is not a JSON array")
            if not all(isinstance(row, dict) for row in payload):
                raise ValueError("Binance funding response contains a non-object row")
            return payload
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < cfg.retry_attempts:
                time.sleep(cfg.retry_backoff_seconds * (2**attempt))
    raise RuntimeError("failed to download Binance funding history") from last_error


def fetch_records(
    cfg: FreezeConfig,
    *,
    open_json: OpenJson | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if cfg.symbol != "BTCUSDT":
        raise ValueError("this frozen source is restricted to BTCUSDT")
    if cfg.start_ms != START_MS or cfg.end_ms != END_MS:
        raise ValueError("this frozen source is restricted to calendar 2020-2023")
    if not 1 <= cfg.limit <= 1_000:
        raise ValueError("Binance funding page limit must be in [1, 1000]")

    records: list[dict[str, Any]] = []
    cursor = cfg.start_ms
    pages = 0
    while cursor <= cfg.end_ms:
        page = _request_page(cfg, start_ms=cursor, open_json=open_json)
        pages += 1
        if not page:
            break
        records.extend(page)
        last_time = int(page[-1]["fundingTime"])
        if last_time < cursor:
            raise ValueError("Binance funding pagination moved backward")
        cursor = last_time + 1
        if len(page) < cfg.limit:
            break
    return records, pages


def validate_records(
    records: list[dict[str, Any]],
    cfg: FreezeConfig,
) -> list[dict[str, str | int]]:
    normalized: list[dict[str, str | int]] = []
    previous_time: int | None = None
    for row in records:
        required = {"symbol", "fundingTime", "fundingRate", "markPrice"}
        if not required.issubset(row):
            raise ValueError("Binance funding row is missing a required field")
        symbol = str(row["symbol"])
        funding_time = int(row["fundingTime"])
        funding_rate = str(row["fundingRate"])
        mark_price = str(row["markPrice"])
        if symbol != cfg.symbol:
            raise ValueError("Binance funding row has the wrong symbol")
        if not cfg.start_ms <= funding_time <= cfg.end_ms:
            raise ValueError("Binance funding row opens the sealed interval")
        if previous_time is not None and funding_time <= previous_time:
            raise ValueError("Binance funding timestamps are not strictly increasing")
        try:
            if not Decimal(funding_rate).is_finite():
                raise ValueError("Binance funding rate is not finite")
            if mark_price and not Decimal(mark_price).is_finite():
                raise ValueError("Binance mark price is not finite")
        except InvalidOperation as exc:
            raise ValueError("Binance funding row has an invalid decimal") from exc
        normalized.append(
            {
                "funding_time_ms": funding_time,
                "funding_time_utc": _utc_iso(funding_time),
                "symbol": symbol,
                "funding_rate": funding_rate,
                "mark_price": mark_price,
            }
        )
        previous_time = funding_time
    if not normalized:
        raise ValueError("Binance funding history is empty")
    return normalized


def _serialize_csv(records: list[dict[str, str | int]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode()


def run_freeze(
    cfg: FreezeConfig,
    *,
    open_json: OpenJson | None = None,
) -> dict[str, Any]:
    downloaded, pages = fetch_records(cfg, open_json=open_json)
    records = validate_records(downloaded, cfg)
    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_serialize_csv(records))

    years = Counter(str(row["funding_time_utc"])[:4] for row in records)
    manifest = {
        "protocol": {
            "name": "Binance BTCUSDT USD-M realized funding, 2020-2023",
            "stage": "pre_outcome_funding_source_freeze",
            "luri_outcomes_opened": False,
            "selection_end_exclusive": "2024-01-01T00:00:00.000Z",
            "endpoint_semantics": "startTime and endTime inclusive; ascending order",
            "settlement_policy": "use exact returned fundingTime; do not infer cadence",
        },
        "config": asdict(cfg),
        "official_source": {
            "api_endpoint": API_ENDPOINT,
            "api_documentation": API_DOCUMENTATION,
            "funding_explanation": FUNDING_EXPLANATION,
            "pages": pages,
        },
        "data": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "rows": len(records),
            "columns": list(COLUMNS),
            "first_funding_time_ms": records[0]["funding_time_ms"],
            "first_funding_time_utc": records[0]["funding_time_utc"],
            "last_funding_time_ms": records[-1]["funding_time_ms"],
            "last_funding_time_utc": records[-1]["funding_time_utc"],
            "rows_by_year": dict(sorted(years.items())),
        },
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=FreezeConfig.output)
    parser.add_argument("--manifest", default=FreezeConfig.manifest)
    args = parser.parse_args()
    manifest = run_freeze(FreezeConfig(output=args.output, manifest=args.manifest))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
