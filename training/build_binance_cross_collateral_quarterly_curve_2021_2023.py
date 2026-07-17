"""Freeze a pre-2024 USD-M/COIN-M current-quarter execution panel.

The panel is a new source axis for a market-neutral cross-collateral basis
alpha.  It downloads only public Binance continuous-contract candles, rejects
post-2023 rows, preserves the completed-candle clock, and never loads a BTC
return or another strategy outcome.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ENDPOINT = "https://fapi.binance.com/fapi/v1/continuousKlines"
PAIRS = {"um": "BTCUSDT", "cm": "BTCUSD"}
DEFAULT_OUTPUT_DIR = "data/binance_cross_collateral_quarterly_curve_2021_2023"
DEFAULT_UM_SNAPSHOT = (
    f"{DEFAULT_OUTPUT_DIR}/BTCUSDT_CURRENT_QUARTER_5m_2021_2023.raw.json.gz"
)
DEFAULT_CM_SNAPSHOT = (
    f"{DEFAULT_OUTPUT_DIR}/BTCUSD_CURRENT_QUARTER_5m_2021_2023.raw.json.gz"
)
FIELDS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)


@dataclass(frozen=True)
class Config:
    start: str = "2021-01-01"
    end: str = "2024-01-01"
    contract_type: str = "CURRENT_QUARTER"
    interval: str = "5m"
    limit: int = 1_500
    output_dir: str = DEFAULT_OUTPUT_DIR
    manifest: str = (
        "results/binance_cross_collateral_quarterly_curve_2021_2023_manifest.json"
    )
    retries: int = 5
    timeout_seconds: int = 60
    request_pause_seconds: float = 0.015
    staged_um_json: str | None = DEFAULT_UM_SNAPSHOT
    staged_cm_json: str | None = DEFAULT_CM_SNAPSHOT


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _milliseconds(value: pd.Timestamp) -> int:
    return int(value.timestamp() * 1_000)


def _request_json(
    url: str,
    *,
    retries: int,
    timeout: int,
) -> list[list[Any]]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "rllm-quarterly-curve-research/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            if not isinstance(payload, list):
                raise ValueError(f"unexpected Binance response: {payload!r}")
            return payload
        except Exception as error:  # pragma: no cover - live retry path
            last_error = error
            if attempt + 1 < retries:
                retry_after = 0.0
                if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                    retry_after = float(error.headers.get("Retry-After", 30.0))
                time.sleep(max(retry_after, min(2.0**attempt, 8.0)))
    raise RuntimeError(f"Binance request failed: {url}") from last_error


def _query_url(pair: str, cursor: pd.Timestamp, end: pd.Timestamp, cfg: Config) -> str:
    query = urllib.parse.urlencode(
        {
            "pair": pair,
            "contractType": cfg.contract_type,
            "interval": cfg.interval,
            "startTime": _milliseconds(cursor),
            "endTime": _milliseconds(end) - 1,
            "limit": cfg.limit,
        }
    )
    return f"{ENDPOINT}?{query}"


def fetch_pair(
    pair: str,
    cfg: Config,
    *,
    fetcher: Callable[..., list[list[Any]]] = _request_json,
) -> tuple[list[list[Any]], int]:
    start = _utc(cfg.start)
    end = _utc(cfg.end)
    cursor = start
    rows: list[list[Any]] = []
    requests = 0
    while cursor < end:
        page = fetcher(
            _query_url(pair, cursor, end, cfg),
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
        requests += 1
        if not page:
            break
        if any(len(row) != len(FIELDS) for row in page):
            raise ValueError(f"{pair} returned an incomplete continuous kline")
        rows.extend(page)
        last_open = pd.to_datetime(int(page[-1][0]), unit="ms", utc=True)
        next_cursor = last_open + pd.Timedelta(minutes=5)
        if next_cursor <= cursor:
            raise ValueError(f"{pair} pagination did not advance")
        cursor = next_cursor
        if cfg.request_pause_seconds > 0.0:
            time.sleep(cfg.request_pause_seconds)

    bounded = {
        int(row[0]): row
        for row in rows
        if _milliseconds(start) <= int(row[0]) < _milliseconds(end)
    }
    return [bounded[key] for key in sorted(bounded)], requests


def rows_to_frame(rows: list[list[Any]], pair: str) -> pd.DataFrame:
    if not rows:
        raise ValueError(f"{pair} returned no pre-2024 rows")
    frame = pd.DataFrame(rows, columns=FIELDS)
    frame["date"] = pd.to_datetime(frame.pop("open_time"), unit="ms", utc=True)
    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("close_time", "trades"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError(f"{pair} timestamps are not unique and sorted")
    intervals = frame["date"].diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta(minutes=5)).all():
        raise ValueError(f"{pair} continuous current-quarter grid has gaps")
    expected_close = frame["date"].astype("int64") // 1_000_000 + 299_999
    if not frame["close_time"].eq(expected_close).all():
        raise ValueError(f"{pair} close times do not match completed five-minute bars")
    positive = frame[["open", "high", "low", "close"]].gt(0.0).all(axis=1)
    finite = frame[["open", "high", "low", "close"]].notna().all(axis=1)
    envelope = frame["high"].ge(frame[["open", "close"]].max(axis=1)) & frame[
        "low"
    ].le(frame[["open", "close"]].min(axis=1))
    frame["ohlc_valid"] = positive & finite & envelope
    return frame


def quarter_delivery_times(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Return UTC 08:00 delivery times through the first delivery after end."""
    start = _utc(start)
    end = _utc(end)
    output: list[pd.Timestamp] = []
    for year in range(start.year - 1, end.year + 2):
        for month in (3, 6, 9, 12):
            delivery = pd.Timestamp(year, month, 1, tz="UTC") + pd.offsets.MonthEnd(0)
            while delivery.weekday() != 4:
                delivery -= pd.Timedelta(days=1)
            delivery = delivery.normalize() + pd.Timedelta(hours=8)
            if delivery >= start - pd.Timedelta(days=100):
                output.append(delivery)
    output = sorted(set(output))
    if not any(timestamp > end for timestamp in output):
        raise ValueError("delivery calendar does not extend beyond source end")
    return output


def combine_pairs(um: pd.DataFrame, cm: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "open", "high", "low", "close", "close_time", "ohlc_valid"]
    left = um[columns].rename(columns={name: f"um_{name}" for name in columns[1:]})
    right = cm[columns].rename(columns={name: f"cm_{name}" for name in columns[1:]})
    panel = left.merge(right, on="date", how="inner", validate="one_to_one")
    panel["source_complete"] = panel["um_ohlc_valid"] & panel["cm_ohlc_valid"]
    if panel.empty or panel["date"].duplicated().any():
        raise ValueError("cross-collateral panel is empty or duplicated")
    intervals = panel["date"].diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta(minutes=5)).all():
        raise ValueError("cross-collateral common grid has gaps")
    if not panel["um_close_time"].eq(panel["cm_close_time"]).all():
        raise ValueError("cross-collateral legs do not share a close clock")
    panel = panel.rename(columns={"date": "open_time"})
    panel.insert(
        1,
        "available_time",
        pd.to_datetime(panel["um_close_time"] + 1, unit="ms", utc=True),
    )
    if not panel["available_time"].eq(
        panel["open_time"] + pd.Timedelta(minutes=5)
    ).all():
        raise ValueError("completed-candle availability is not the next bar boundary")

    deliveries = quarter_delivery_times(
        panel["open_time"].iloc[0], panel["open_time"].iloc[-1]
    )
    delivery_ns = [timestamp.value for timestamp in deliveries]
    open_ns = panel["open_time"].astype("int64").to_numpy()
    positions = pd.Series(open_ns).map(
        lambda value: next(index for index, item in enumerate(delivery_ns) if item > value)
    )
    panel["delivery_time"] = pd.DatetimeIndex(
        [deliveries[index] for index in positions]
    )
    panel["contract_segment"] = panel["delivery_time"].dt.strftime("%Y%m%d")
    panel["bars_to_delivery"] = (
        (panel["delivery_time"] - panel["open_time"]) / pd.Timedelta(minutes=5)
    ).astype(int)
    delivery_set = set(deliveries)
    panel["is_roll_boundary"] = panel["open_time"].isin(delivery_set)
    panel["is_pre_roll_final_bar"] = panel["available_time"].isin(delivery_set)
    return panel


def _write_deterministic_gzip(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as stream:
        stream.write(csv_bytes)
    payload = buffer.getvalue()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_raw_rows(rows: list[list[Any]], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(rows, separators=(",", ":")).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as stream:
        stream.write(raw)
    payload = buffer.getvalue()
    path.write_bytes(payload)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": int(len(payload)),
        "uncompressed_json_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _load_staged_rows(path: str, pair: str) -> tuple[list[list[Any]], dict[str, Any]]:
    source = Path(path)
    payload = source.read_bytes()
    raw = gzip.decompress(payload) if payload.startswith(b"\x1f\x8b") else payload
    rows = json.loads(raw)
    if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
        raise ValueError(f"staged {pair} source is not a JSON row array")
    return rows, {
        "staged_input_sha256": hashlib.sha256(payload).hexdigest(),
        "staged_input_bytes": int(len(payload)),
        "staged_uncompressed_json_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _pair_summary(frame: pd.DataFrame, requests: int) -> dict[str, Any]:
    invalid = frame.loc[~frame["ohlc_valid"], "date"]
    return {
        "rows": int(len(frame)),
        "first_open_time": frame["date"].iloc[0].isoformat(),
        "last_open_time": frame["date"].iloc[-1].isoformat(),
        "requests": int(requests),
        "invalid_ohlc_rows": int(len(invalid)),
        "invalid_ohlc_timestamps": [timestamp.isoformat() for timestamp in invalid],
    }


def build(cfg: Config) -> dict[str, Any]:
    start = _utc(cfg.start)
    end = _utc(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start < pd.Timestamp("2021-01-01", tz="UTC") or end > pd.Timestamp(
        "2024-01-01", tz="UTC"
    ):
        raise ValueError("quarterly-curve source build is physically bounded to 2021-2023")
    if cfg.contract_type != "CURRENT_QUARTER" or cfg.interval != "5m":
        raise ValueError("frozen source contract is CURRENT_QUARTER at five minutes")
    if cfg.limit != 1_500:
        raise ValueError("frozen source pagination limit must remain 1500")

    staged_paths = {"um": cfg.staged_um_json, "cm": cfg.staged_cm_json}
    if bool(cfg.staged_um_json) != bool(cfg.staged_cm_json):
        raise ValueError("both staged pair sources must be supplied together")
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}
    output_dir = Path(cfg.output_dir)
    for venue, pair in PAIRS.items():
        staging: dict[str, Any] = {}
        if staged_paths[venue]:
            rows, staging = _load_staged_rows(str(staged_paths[venue]), pair)
            requests = 0
        else:
            rows, requests = fetch_pair(pair, cfg)
        frame = rows_to_frame(rows, pair)
        if frame["date"].min() < start or frame["date"].max() >= end:
            raise ValueError(f"{pair} opened a row outside the frozen source window")
        frames[venue] = frame
        raw_file = _write_raw_rows(
            rows,
            output_dir / f"{pair}_CURRENT_QUARTER_5m_2021_2023.raw.json.gz",
        )
        summaries[venue] = {
            "pair": pair,
            **_pair_summary(frame, requests),
            **staging,
            "raw_snapshot": raw_file,
        }

    panel = combine_pairs(frames["um"], frames["cm"])
    output = output_dir / (
        "BTCUSDT_BTCUSD_CURRENT_QUARTER_5m_2021_2023.csv.gz"
    )
    file_hash = _write_deterministic_gzip(panel, output)
    body = {
        "protocol": {
            "name": "Binance cross-collateral current-quarter five-minute panel",
            "outcomes_opened": False,
            "requested_start_inclusive": start.isoformat(),
            "requested_end_exclusive": end.isoformat(),
            "post_2023_rows_requested": False,
            "source": "official public Binance USD-M Futures REST market data",
            "endpoint": ENDPOINT,
            "availability": "each row becomes usable only after close_time",
            "execution_contracts": (
                "USD-M BTCUSDT CURRENT_QUARTER and COIN-M BTCUSD "
                "CURRENT_QUARTER; do not cross a delivery roll"
            ),
        },
        "config": {
            key: value
            for key, value in asdict(cfg).items()
            if key not in {"staged_um_json", "staged_cm_json"}
        },
        "source_mode": (
            "offline_official_api_snapshot" if cfg.staged_um_json else "live_official_api"
        ),
        "pairs": summaries,
        "combined": {
            "rows": int(len(panel)),
            "first_open_time": panel["open_time"].iloc[0].isoformat(),
            "last_open_time": panel["open_time"].iloc[-1].isoformat(),
            "source_complete_rows": int(panel["source_complete"].sum()),
            "incomplete_rows": int((~panel["source_complete"]).sum()),
            "contract_segments": int(panel["contract_segment"].nunique()),
            "roll_boundary_rows": int(panel["is_roll_boundary"].sum()),
            "pre_roll_final_rows": int(panel["is_pre_roll_final_bar"].sum()),
        },
        "file": {
            "path": str(output),
            "sha256": file_hash,
            "bytes": int(output.stat().st_size),
        },
    }
    manifest = {
        **body,
        "manifest_hash": _canonical_hash(body),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=Config.start)
    parser.add_argument("--end", default=Config.end)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--manifest", default=Config.manifest)
    parser.add_argument("--request-pause-seconds", type=float, default=0.015)
    parser.add_argument("--staged-um-json", default=Config.staged_um_json)
    parser.add_argument("--staged-cm-json", default=Config.staged_cm_json)
    parser.add_argument(
        "--live-api",
        action="store_true",
        help="refresh from the mutable public API instead of replaying tracked snapshots",
    )
    args = parser.parse_args()
    if args.live_api and (
        args.staged_um_json != Config.staged_um_json
        or args.staged_cm_json != Config.staged_cm_json
    ):
        parser.error("--live-api cannot be combined with custom staged snapshots")
    manifest = build(
        Config(
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            manifest=args.manifest,
            request_pause_seconds=args.request_pause_seconds,
            staged_um_json=None if args.live_api else args.staged_um_json,
            staged_cm_json=None if args.live_api else args.staged_cm_json,
        )
    )
    print(
        json.dumps(
            {
                "manifest": args.manifest,
                "manifest_hash": manifest["manifest_hash"],
                "file": manifest["file"],
                "combined": manifest["combined"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
