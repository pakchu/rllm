"""Freeze complete BTCUSDT funding-event mark prices for 2020-2023."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


BASE_URL = "https://fapi.binance.com"
ENDPOINT = "/fapi/v1/markPriceKlines"
OFFICIAL_DOCUMENTATION = (
    "https://developers.binance.com/docs/derivatives/usds-margined-futures/"
    "market-data/rest-api/Mark-Price-Kline-Candlestick-Data"
)
START = pd.Timestamp("2020-01-01 00:00:00")
END = pd.Timestamp("2024-01-01 00:00:00")
INTERVAL = "8h"
STEP_MS = 8 * 60 * 60 * 1_000
LIMIT = 1_500
MAXIMUM_FUNDING_TIME_OFFSET_MS = 60_000
MAXIMUM_PROXY_FUNDING_CASH_ERROR_BP_NOTIONAL = 0.1
PARENT_FUNDING = Path("results/binance_um_btcusdt_realized_funding_2020_2023.csv")
PARENT_FUNDING_SHA256 = "c19829fa085a50f29c13762373a2b6db1c62025d657be1f5a3fbb9ce254482f7"
PARENT_MANIFEST = Path(
    "results/binance_um_btcusdt_realized_funding_2020_2023_manifest.json"
)
PARENT_MANIFEST_SHA256 = "c70280e46bcbc2410cc59c2bcc93780c40997dbc5d0edb82d82127b59593250c"
DEFAULT_OUTPUT = "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
DEFAULT_MANIFEST = (
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
DEFAULT_DOCS = "docs/binance-um-btcusdt-funding-marks-2020-2023-2026-07-17.md"
FREEZER_PATH = Path("training/freeze_binance_um_btcusdt_funding_marks_2020_2023.py")
TEST_PATH = Path("tests/test_freeze_binance_um_btcusdt_funding_marks_2020_2023.py")
RequestJson = Callable[[str, dict[str, Any]], Any]


@dataclass(frozen=True)
class FreezeConfig:
    symbol: str = "BTCUSDT"
    interval: str = INTERVAL
    start: str = str(START)
    end: str = str(END)
    limit: int = LIMIT
    timeout_seconds: float = 30.0
    retry_attempts: int = 5
    retry_backoff_seconds: float = 1.0
    sleep_seconds: float = 0.05
    output: str = DEFAULT_OUTPUT
    manifest: str = DEFAULT_MANIFEST
    docs: str = DEFAULT_DOCS


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_json(path: str, params: dict[str, Any], cfg: FreezeConfig) -> Any:
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "rllm-funding-mark-freeze/1.0"},
    )
    error: Exception | None = None
    for attempt in range(cfg.retry_attempts):
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=cfg.timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network failure only
            error = exc
            if attempt + 1 < cfg.retry_attempts:
                time.sleep(cfg.retry_backoff_seconds * (2**attempt))
    assert error is not None
    raise RuntimeError("failed to download Binance mark-price history") from error


def download_mark_klines(
    cfg: FreezeConfig,
    *,
    request_json: RequestJson | None = None,
) -> tuple[pd.DataFrame, int]:
    if cfg.symbol != "BTCUSDT" or cfg.interval != INTERVAL:
        raise ValueError("funding-mark freeze is restricted to BTCUSDT 8h")
    if pd.Timestamp(cfg.start) != START or pd.Timestamp(cfg.end) != END:
        raise ValueError("funding-mark freeze is restricted to calendar 2020-2023")
    if not 1 <= cfg.limit <= 1_500:
        raise ValueError("mark-price page limit must be in [1, 1500]")
    opener = request_json or (lambda path, params: _request_json(path, params, cfg))
    start_ms = int(START.tz_localize("UTC").timestamp() * 1_000)
    end_ms = int(END.tz_localize("UTC").timestamp() * 1_000)
    cursor = start_ms
    rows: list[list[Any]] = []
    pages = 0
    while cursor < end_ms:
        batch = opener(
            ENDPOINT,
            {
                "symbol": cfg.symbol,
                "interval": cfg.interval,
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": cfg.limit,
            },
        )
        pages += 1
        if not isinstance(batch, list):
            raise ValueError("Binance mark-price response is not an array")
        if not batch:
            break
        if not all(isinstance(row, list) and len(row) >= 7 for row in batch):
            raise ValueError("Binance mark-price response has an invalid row")
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + STEP_MS
        if next_cursor <= cursor:
            raise ValueError("Binance mark-price pagination stalled")
        cursor = next_cursor
        if cursor < end_ms and cfg.sleep_seconds:
            time.sleep(cfg.sleep_seconds)
    if not rows:
        raise ValueError("Binance mark-price history is empty")
    frame = pd.DataFrame(
        {
            "open_time_ms": [int(row[0]) for row in rows],
            "open": pd.to_numeric([row[1] for row in rows], errors="raise"),
            "high": pd.to_numeric([row[2] for row in rows], errors="raise"),
            "low": pd.to_numeric([row[3] for row in rows], errors="raise"),
            "close": pd.to_numeric([row[4] for row in rows], errors="raise"),
            "close_time_ms": [int(row[6]) for row in rows],
        }
    )
    frame = frame.drop_duplicates("open_time_ms").sort_values("open_time_ms")
    frame = frame.loc[
        frame["open_time_ms"].ge(start_ms) & frame["open_time_ms"].lt(end_ms)
    ].reset_index(drop=True)
    expected = np.arange(start_ms, end_ms, STEP_MS, dtype=np.int64)
    actual = frame["open_time_ms"].to_numpy(np.int64)
    if not np.array_equal(actual, expected):
        raise ValueError("Binance 8h mark-price grid is incomplete")
    expected_close = actual + STEP_MS - 1
    if not np.array_equal(frame["close_time_ms"].to_numpy(np.int64), expected_close):
        raise ValueError("Binance 8h mark-price close times are invalid")
    prices = frame[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("Binance mark-price values are invalid")
    if (
        (frame["high"] < frame[["open", "close"]].max(axis=1)).any()
        or (frame["low"] > frame[["open", "close"]].min(axis=1)).any()
    ):
        raise ValueError("Binance mark-price OHLC is invalid")
    return frame, pages


def load_parent_funding() -> pd.DataFrame:
    if sha256_file(PARENT_FUNDING) != PARENT_FUNDING_SHA256:
        raise ValueError("parent funding data hash changed")
    if sha256_file(PARENT_MANIFEST) != PARENT_MANIFEST_SHA256:
        raise ValueError("parent funding manifest hash changed")
    manifest = json.loads(PARENT_MANIFEST.read_text())
    if manifest.get("protocol", {}).get("luri_outcomes_opened") is not False:
        raise ValueError("parent funding provenance changed")
    if manifest.get("data", {}).get("sha256") != PARENT_FUNDING_SHA256:
        raise ValueError("parent funding manifest points to another file")
    frame = pd.read_csv(
        PARENT_FUNDING,
        usecols=[
            "funding_time_ms",
            "funding_time_utc",
            "symbol",
            "funding_rate",
            "mark_price",
        ],
        dtype={"funding_rate": str, "mark_price": str, "symbol": str},
    )
    frame["funding_time_ms"] = pd.to_numeric(
        frame["funding_time_ms"], errors="raise"
    ).astype(np.int64)
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
    if frame["funding_time_ms"].duplicated().any() or not frame[
        "funding_time_ms"
    ].is_monotonic_increasing:
        raise ValueError("parent funding timestamps are invalid")
    if not frame["symbol"].eq("BTCUSDT").all():
        raise ValueError("parent funding contains another symbol")
    if not np.isfinite(frame["funding_rate"].to_numpy(float)).all():
        raise ValueError("parent funding rates are invalid")
    return frame


def compose_settlement_marks(
    funding: pd.DataFrame,
    mark_klines: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = funding.copy()
    frame["mark_open_time_ms"] = (
        frame["funding_time_ms"].to_numpy(np.int64) // STEP_MS * STEP_MS
    )
    frame["funding_time_offset_ms"] = (
        frame["funding_time_ms"] - frame["mark_open_time_ms"]
    )
    offsets = frame["funding_time_offset_ms"].to_numpy(np.int64)
    if (offsets < 0).any() or (offsets > MAXIMUM_FUNDING_TIME_OFFSET_MS).any():
        raise ValueError("funding timestamp is too far from its canonical 8h boundary")
    if frame["mark_open_time_ms"].duplicated().any():
        raise ValueError("multiple funding events map to one 8h mark interval")
    mark_open = mark_klines.set_index("open_time_ms")["open"]
    frame["settlement_mark_price"] = frame["mark_open_time_ms"].map(mark_open)
    if frame["settlement_mark_price"].isna().any():
        raise ValueError("funding event lacks an 8h mark-price open")

    recorded = pd.to_numeric(frame["mark_price"], errors="coerce")
    overlap = recorded.notna()
    if overlap.any():
        mark_error_bp = (
            frame.loc[overlap, "settlement_mark_price"] / recorded.loc[overlap] - 1.0
        ).abs() * 10_000.0
        cash_error_bp = mark_error_bp * frame.loc[overlap, "funding_rate"].abs()
        maximum_mark_error_bp = float(mark_error_bp.max())
        maximum_cash_error_bp = float(cash_error_bp.max())
        if maximum_cash_error_bp > MAXIMUM_PROXY_FUNDING_CASH_ERROR_BP_NOTIONAL:
            raise ValueError("8h mark-open funding-cash proxy error exceeds frozen limit")
    else:
        maximum_mark_error_bp = 0.0
        maximum_cash_error_bp = 0.0

    event_time = pd.to_datetime(frame["funding_time_ms"], unit="ms", utc=True)
    recorded_time = pd.to_datetime(frame["funding_time_utc"], utc=True, errors="raise")
    if not pd.DatetimeIndex(event_time).equals(pd.DatetimeIndex(recorded_time)):
        raise ValueError("parent funding timestamp representations disagree")
    output = pd.DataFrame(
        {
            "funding_time_ms": frame["funding_time_ms"].to_numpy(np.int64),
            "funding_time_utc": recorded_time.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "symbol": frame["symbol"],
            "funding_rate": frame["funding_rate"],
            "settlement_mark_price": frame["settlement_mark_price"],
            "mark_open_time_ms": frame["mark_open_time_ms"].to_numpy(np.int64),
            "mark_open_time_utc": pd.to_datetime(
                frame["mark_open_time_ms"], unit="ms", utc=True
            ).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "funding_time_offset_ms": offsets,
            "mark_source": "binance_8h_mark_price_kline_open",
        }
    )
    stats = {
        "events": int(len(output)),
        "recorded_mark_overlap_events": int(overlap.sum()),
        "backfilled_events": int((~overlap).sum()),
        "maximum_funding_time_offset_ms": int(offsets.max()),
        "maximum_recorded_vs_8h_open_mark_error_bp": maximum_mark_error_bp,
        "maximum_proxy_funding_cash_error_bp_notional": maximum_cash_error_bp,
    }
    return output, stats


def deterministic_csv_gz(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as handle:
        handle.write(csv_bytes)
    target.write_bytes(buffer.getvalue())


def _markdown(result: dict[str, Any]) -> str:
    stats = result["quality"]
    return f"""# BTCUSDT funding settlement marks, 2020-2023

Outcome-blind execution-source freeze only. No position return, PnL, CAGR,
drawdown, gate, or 2024+ row was calculated.

- funding events: {stats['events']}
- exact funding-record mark overlap: {stats['recorded_mark_overlap_events']}
- mark-open backfills: {stats['backfilled_events']}
- maximum funding timestamp jitter: {stats['maximum_funding_time_offset_ms']} ms
- maximum overlap mark error: {stats['maximum_recorded_vs_8h_open_mark_error_bp']:.9f} bp
- maximum implied funding-cash error: {stats['maximum_proxy_funding_cash_error_bp_notional']:.12f} bp/notional

All settlement marks use one uniform rule: the open of the official Binance
BTCUSDT USD-M 8h mark-price kline whose canonical boundary contains the
returned funding timestamp. The 185 non-empty historical
`fundingRate.markPrice` values are validation overlaps only. Returned funding
timestamps are retained exactly for settlement inclusion.

Official endpoint: <{OFFICIAL_DOCUMENTATION}>

Manifest hash: `{result['manifest_hash']}`
"""


def run(
    cfg: FreezeConfig,
    *,
    request_json: RequestJson | None = None,
) -> dict[str, Any]:
    funding = load_parent_funding()
    mark_klines, pages = download_mark_klines(cfg, request_json=request_json)
    events, quality = compose_settlement_marks(funding, mark_klines)
    deterministic_csv_gz(events, cfg.output)
    core: dict[str, Any] = {
        "protocol_version": "btc_um_funding_settlement_marks_2020_2023_v1",
        "outcomes_opened": False,
        "strategy_outcomes_calculated": [],
        "selection_end_exclusive": str(END),
        "config": asdict(cfg),
        "implementation": {
            "freezer_path": str(FREEZER_PATH),
            "freezer_sha256": sha256_file(FREEZER_PATH),
            "test_path": str(TEST_PATH),
            "test_sha256": sha256_file(TEST_PATH),
        },
        "parent": {
            "funding_path": str(PARENT_FUNDING),
            "funding_sha256": PARENT_FUNDING_SHA256,
            "manifest_path": str(PARENT_MANIFEST),
            "manifest_sha256": PARENT_MANIFEST_SHA256,
        },
        "official_source": {
            "endpoint": f"{BASE_URL}{ENDPOINT}",
            "documentation": OFFICIAL_DOCUMENTATION,
            "interval": INTERVAL,
            "pages": pages,
        },
        "mapping": {
            "funding_time": "exact returned fundingTime retained",
            "mark": "open of floor(fundingTime, 8h) official mark-price kline",
            "maximum_allowed_timestamp_offset_ms": MAXIMUM_FUNDING_TIME_OFFSET_MS,
            "maximum_allowed_proxy_funding_cash_error_bp_notional": (
                MAXIMUM_PROXY_FUNDING_CASH_ERROR_BP_NOTIONAL
            ),
        },
        "quality": quality,
        "data": {
            "path": cfg.output,
            "sha256": sha256_file(cfg.output),
            "rows": int(len(events)),
            "columns": list(events.columns),
            "first_funding_time_ms": int(events["funding_time_ms"].iloc[0]),
            "last_funding_time_ms": int(events["funding_time_ms"].iloc[-1]),
        },
        "sealed": ["all_strategy_returns", "2024", "2025", "2026_ytd"],
    }
    result = {
        **core,
        "manifest_hash": _canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(cfg.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.manifest).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    Path(cfg.docs).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.docs).write_text(_markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--docs", default=DEFAULT_DOCS)
    parser.add_argument("--sleep-seconds", type=float, default=FreezeConfig.sleep_seconds)
    args = parser.parse_args()
    cfg = FreezeConfig(
        output=args.output,
        manifest=args.manifest,
        docs=args.docs,
        sleep_seconds=args.sleep_seconds,
    )
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
