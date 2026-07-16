"""Freeze 2020-2022 Coinbase and Binance inputs without opening trade outcomes."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training.export_wikimedia_attention_source import (
    deterministic_gzip_csv,
    sha256_file,
)
from training.preregister_coinbase_spot_leadership_alpha import (
    DEFAULT_OUTPUT as DEFAULT_PREREGISTRATION,
    SELECTION_END,
    canonical_hash,
    validate_manifest as validate_preregistration,
)


DEFAULT_BINANCE_INPUT = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
DEFAULT_FUNDING_INPUT = (
    "/home/pakchu/rllm/data/binance_um_aux_btc_2020_2026/"
    "BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
)
DEFAULT_COINBASE_OUTPUT = "data/coinbase_btcusd_5m_2020_2022.csv.gz"
DEFAULT_BINANCE_OUTPUT = "data/coinbase_leadership_binance_5m_2020_2022.csv.gz"
DEFAULT_FUNDING_OUTPUT = "data/coinbase_leadership_funding_2020_2022.csv.gz"
DEFAULT_MANIFEST = "results/coinbase_spot_leadership_source_manifest_2026-07-16.json"
DEFAULT_START = "2020-01-01"
DEFAULT_END = SELECTION_END
COINBASE_ENDPOINT = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
USER_AGENT = "rllm-coinbase-venue-alpha/1.0 (https://github.com/pakchu/rllm)"
GRANULARITY_SECONDS = 300
MAX_BUCKETS_PER_REQUEST = 299


@dataclass(frozen=True)
class Config:
    binance_input: str = DEFAULT_BINANCE_INPUT
    funding_input: str = DEFAULT_FUNDING_INPUT
    coinbase_output: str = DEFAULT_COINBASE_OUTPUT
    binance_output: str = DEFAULT_BINANCE_OUTPUT
    funding_output: str = DEFAULT_FUNDING_OUTPUT
    manifest_output: str = DEFAULT_MANIFEST
    preregistration: str = DEFAULT_PREREGISTRATION
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    timeout_seconds: float = 30.0
    retries: int = 5
    requests_per_second: float = 8.0
    attest_existing: bool = False


def resolve_existing(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(path)


def raw_input_metadata(supplied_path: str | Path) -> dict[str, Any]:
    supplied = Path(supplied_path).expanduser()
    resolved = resolve_existing(supplied)
    return {
        "supplied_path": str(supplied),
        "resolved_path": str(resolved),
        "supplied_is_symlink": supplied.is_symlink(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def git_commit_for(path: str | Path) -> str:
    candidate = Path(path)
    try:
        pathspec = str(candidate.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        pathspec = str(candidate)
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", pathspec],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(result) != 40:
        raise RuntimeError(f"no Git commit anchor found for {path}")
    return result


def expected_grid(start: str, end: str) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_convert("UTC").tz_localize(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_convert("UTC").tz_localize(None)
    if start_ts.floor("5min") != start_ts or end_ts.floor("5min") != end_ts:
        raise ValueError("source boundaries must align to five-minute buckets")
    if start_ts >= end_ts:
        raise ValueError("source start must precede end")
    return pd.date_range(start_ts, end_ts, freq="5min", inclusive="left")


def request_windows(
    grid: pd.DatetimeIndex, *, chunk_size: int = MAX_BUCKETS_PER_REQUEST
) -> list[pd.DatetimeIndex]:
    if chunk_size < 1 or chunk_size > MAX_BUCKETS_PER_REQUEST:
        raise ValueError(f"chunk_size must be in [1, {MAX_BUCKETS_PER_REQUEST}]")
    return [grid[offset : offset + chunk_size] for offset in range(0, len(grid), chunk_size)]


def _coinbase_time(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def candle_url(window: pd.DatetimeIndex) -> str:
    if len(window) == 0:
        raise ValueError("Coinbase request window cannot be empty")
    query = urllib.parse.urlencode(
        {
            "granularity": GRANULARITY_SECONDS,
            "start": _coinbase_time(window[0]),
            "end": _coinbase_time(window[-1]),
        }
    )
    return f"{COINBASE_ENDPOINT}?{query}"


def fetch_payload(url: str, cfg: Config) -> tuple[list[Any], str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(cfg.retries):
        try:
            with urllib.request.urlopen(request, timeout=cfg.timeout_seconds) as handle:
                raw = handle.read()
            payload = json.loads(raw)
            if not isinstance(payload, list):
                raise ValueError("Coinbase candle response must be a JSON list")
            return payload, hashlib.sha256(raw).hexdigest()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 and exc.code < 500:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt + 1 < cfg.retries:
            time.sleep(min(30.0, 1.5 * (2**attempt)))
    raise RuntimeError(f"Coinbase request failed after {cfg.retries} attempts: {url}") from last_error


def _parse_coinbase_row(raw: Any) -> tuple[pd.Timestamp, tuple[float, float, float, float, float]]:
    if not isinstance(raw, list) or len(raw) != 6:
        raise ValueError(f"unexpected Coinbase candle row: {raw!r}")
    epoch = int(raw[0])
    if epoch % GRANULARITY_SECONDS:
        raise ValueError(f"Coinbase candle is not five-minute aligned: {epoch}")
    timestamp = pd.Timestamp(epoch, unit="s", tz="UTC").tz_localize(None)
    low, high, open_price, close, volume = (float(value) for value in raw[1:])
    values = np.asarray([low, high, open_price, close, volume], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Coinbase candle contains non-finite values")
    if min(low, high, open_price, close) <= 0 or volume < 0:
        raise ValueError("Coinbase candle contains non-positive price or negative volume")
    if low > min(open_price, close) or high < max(open_price, close) or low > high:
        raise ValueError("Coinbase candle has incoherent OHLC")
    return timestamp, (open_price, high, low, close, volume)


def parse_coinbase_payload(
    payload: list[Any], expected: Iterable[pd.Timestamp]
) -> tuple[dict[pd.Timestamp, tuple[float, float, float, float, float]], int]:
    expected_set = set(expected)
    accepted: dict[pd.Timestamp, tuple[float, float, float, float, float]] = {}
    outside = 0
    for raw in payload:
        timestamp, candle = _parse_coinbase_row(raw)
        if timestamp not in expected_set:
            outside += 1
            continue
        prior = accepted.get(timestamp)
        if prior is not None and prior != candle:
            raise RuntimeError(f"conflicting Coinbase duplicate candle: {timestamp}")
        accepted[timestamp] = candle
    return accepted, outside


def coinbase_frame(
    grid: pd.DatetimeIndex,
    candles: dict[pd.Timestamp, tuple[float, float, float, float, float]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for timestamp in grid:
        candle = candles.get(timestamp)
        records.append(
            {
                "date": timestamp,
                "open": candle[0] if candle is not None else np.nan,
                "high": candle[1] if candle is not None else np.nan,
                "low": candle[2] if candle is not None else np.nan,
                "close": candle[3] if candle is not None else np.nan,
                "volume": candle[4] if candle is not None else np.nan,
                "source_complete": int(candle is not None),
            }
        )
    frame = pd.DataFrame.from_records(records)
    complete = frame["source_complete"].eq(1)
    if complete.any():
        values = frame.loc[complete, ["open", "high", "low", "close", "volume"]].to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError("complete Coinbase rows contain non-finite values")
    if frame.loc[~complete, ["open", "high", "low", "close", "volume"]].notna().any().any():
        raise RuntimeError("missing Coinbase rows must not contain imputed values")
    return frame


def range_frame(
    path: str | Path,
    *,
    date_column: str,
    start: str,
    end: str,
    usecols: list[str],
) -> pd.DataFrame:
    """Stream a chronological range and stop before reading future non-date values."""
    source = Path(path)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    rows: list[dict[str, Any]] = []
    previous: pd.Timestamp | None = None
    sentinel: pd.Timestamp | None = None
    opener = gzip.open if source.suffix == ".gz" else Path.open
    with opener(source, "rt", encoding="utf-8", newline="") as handle:
        header_line = handle.readline()
        fieldnames = next(csv.reader([header_line]))
        missing = set(usecols) - set(fieldnames)
        if missing:
            raise ValueError(f"source columns missing: {sorted(missing)}")
        if not fieldnames or fieldnames[0] != date_column:
            raise ValueError("date column must be first to enforce the cutoff audit boundary")
        for raw_line in handle:
            date_token = raw_line.split(",", 1)[0]
            timestamp = pd.Timestamp(date_token)
            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_convert("UTC").tz_localize(None)
            if previous is not None and timestamp < previous:
                raise RuntimeError("range source is not chronological")
            previous = timestamp
            if timestamp >= end_ts:
                sentinel = timestamp
                break
            if timestamp < start_ts:
                continue
            values = next(csv.reader([raw_line]))
            if len(values) != len(fieldnames):
                raise ValueError("malformed CSV row inside frozen range")
            raw = dict(zip(fieldnames, values))
            row = {column: raw[column] for column in usecols}
            row[date_column] = timestamp
            rows.append(row)
    if not rows:
        raise ValueError(f"no source rows in frozen range: {path}")
    frame = pd.DataFrame(rows, columns=usecols)
    frame.attrs["cutoff_sentinel_date"] = str(sentinel) if sentinel is not None else None
    frame.attrs["future_non_date_fields_csv_parsed"] = 0
    return frame


def validate_binance(frame: pd.DataFrame, expected: pd.DatetimeIndex) -> pd.DataFrame:
    source_attrs = dict(frame.attrs)
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any() or not frame["date"].equals(pd.Series(expected)):
        raise RuntimeError("Binance prefix is not the exact complete five-minute grid")
    numeric = ["open", "high", "low", "close", "quote_asset_volume"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"invalid Binance prefix column: {column}")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Binance prices must be positive")
    if (frame["quote_asset_volume"] < 0).any():
        raise ValueError("Binance quote volume must be nonnegative")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise ValueError("Binance low is incoherent")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise ValueError("Binance high is incoherent")
    frame.attrs.update(source_attrs)
    return frame


def validate_funding(frame: pd.DataFrame) -> pd.DataFrame:
    source_attrs = dict(frame.attrs)
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
    if not np.isfinite(frame["funding_rate"]).all():
        raise ValueError("funding prefix contains non-finite rates")
    duplicated = frame[frame["date"].duplicated(keep=False)]
    if not duplicated.empty:
        distinct = duplicated.groupby("date")["funding_rate"].nunique()
        if (distinct > 1).any():
            raise RuntimeError("funding prefix has conflicting duplicate timestamps")
        frame = frame.drop_duplicates("date", keep="first").reset_index(drop=True)
    timestamps_ns = frame["date"].astype("int64").to_numpy()
    period_ns = 8 * 60 * 60 * 1_000_000_000
    remainder = timestamps_ns % period_ns
    grid_distance_seconds = np.minimum(remainder, period_ns - remainder) / 1e9
    if (grid_distance_seconds > 1.0).any():
        raise RuntimeError("funding timestamp is more than one second from the 8h grid")
    spacing = frame["date"].diff().dropna().dt.total_seconds()
    if (np.abs(spacing - 8 * 60 * 60) > 2.0).any():
        raise RuntimeError("funding prefix is not a complete approximate 8h sequence")
    frame.attrs.update(source_attrs)
    frame.attrs["maximum_grid_offset_seconds"] = float(grid_distance_seconds.max())
    return frame


def validate_config(cfg: Config) -> None:
    if cfg.end != SELECTION_END:
        raise RuntimeError("selection source freezer is locked to end before 2023")
    if pd.Timestamp(cfg.start) != pd.Timestamp(DEFAULT_START):
        raise RuntimeError("selection source freezer is locked to start at 2020")
    if cfg.retries < 1 or not (0 < cfg.requests_per_second <= 8.0):
        raise ValueError("invalid retry or Coinbase request-rate configuration")


def audit_fields(cfg: Config, binance: pd.DataFrame, funding: pd.DataFrame) -> dict[str, Any]:
    return {
        "raw_inputs": {
            "binance_market": raw_input_metadata(cfg.binance_input),
            "binance_funding": raw_input_metadata(cfg.funding_input),
        },
        "git_anchors": {
            "preregistration_manifest_commit": git_commit_for(cfg.preregistration),
            "source_freezer_code_commit": git_commit_for(__file__),
        },
        "source_freezer_code_sha256": sha256_file(__file__),
        "prefix_materialization_contract": {
            "reader": "chronological raw-line stream with first date field parsed alone",
            "stop": "before CSV parsing any non-date field at or after 2023-01-01",
            "future_non_date_fields_csv_parsed": 0,
            "binance_cutoff_sentinel_date": binance.attrs.get("cutoff_sentinel_date"),
            "funding_cutoff_sentinel_date": funding.attrs.get("cutoff_sentinel_date"),
        },
        "funding_timestamp_contract": {
            "normalization": "none; preserve exact source milliseconds",
            "maximum_distance_from_8h_grid_seconds": funding.attrs.get(
                "maximum_grid_offset_seconds"
            ),
            "application": "causal as-of; funding timestamp <= event/position timestamp",
            "exact_timestamp_join_forbidden": True,
        },
    }


def run(cfg: Config) -> dict[str, Any]:
    validate_config(cfg)
    prereg_path = resolve_existing(cfg.preregistration)
    prereg = json.loads(prereg_path.read_text())
    validate_preregistration(prereg)
    if prereg.get("outcomes_opened") is not False:
        raise RuntimeError("source freeze requires unopened outcomes")

    grid = expected_grid(cfg.start, cfg.end)
    candles: dict[pd.Timestamp, tuple[float, float, float, float, float]] = {}
    requests: list[dict[str, Any]] = []
    delay = 1.0 / cfg.requests_per_second
    last_started = 0.0
    for number, window in enumerate(request_windows(grid), start=1):
        sleep_for = delay - (time.monotonic() - last_started)
        if sleep_for > 0:
            time.sleep(sleep_for)
        url = candle_url(window)
        last_started = time.monotonic()
        payload, payload_sha256 = fetch_payload(url, cfg)
        parsed, outside = parse_coinbase_payload(payload, window)
        overlap = set(candles).intersection(parsed)
        if overlap:
            raise RuntimeError(f"Coinbase request windows overlapped: {min(overlap)}")
        candles.update(parsed)
        requests.append(
            {
                "request_number": number,
                "start_inclusive": str(window[0]),
                "end_inclusive": str(window[-1]),
                "expected_rows": int(len(window)),
                "returned_rows": int(len(payload)),
                "accepted_rows": int(len(parsed)),
                "outside_rows": int(outside),
                "payload_sha256": payload_sha256,
            }
        )

    coinbase = coinbase_frame(grid, candles)
    binance_input = resolve_existing(cfg.binance_input)
    funding_input = resolve_existing(cfg.funding_input)
    binance_raw = range_frame(
        binance_input,
        date_column="date",
        start=cfg.start,
        end=cfg.end,
        usecols=["date", "open", "high", "low", "close", "quote_asset_volume"],
    )
    binance = validate_binance(binance_raw, grid)
    funding_raw = range_frame(
        funding_input,
        date_column="date",
        start=cfg.start,
        end=cfg.end,
        usecols=["date", "funding_rate"],
    )
    funding = validate_funding(funding_raw)

    deterministic_gzip_csv(coinbase, cfg.coinbase_output)
    deterministic_gzip_csv(binance, cfg.binance_output)
    deterministic_gzip_csv(funding, cfg.funding_output)
    complete = int(coinbase["source_complete"].sum())
    missing = int(len(coinbase) - complete)
    request_log_hash = canonical_hash(requests)
    core: dict[str, Any] = {
        "protocol_version": "coinbase_spot_leadership_source_freeze_v1",
        "phase": "selection_inputs_only_2020_2022",
        "forward_trade_outcomes_opened": False,
        "start_inclusive": cfg.start,
        "end_exclusive": cfg.end,
        "preregistration_path": str(prereg_path),
        "preregistration_file_sha256": sha256_file(prereg_path),
        "preregistration_manifest_hash": prereg["manifest_hash"],
        "coinbase_request_contract": {
            "endpoint": COINBASE_ENDPOINT,
            "product": "BTC-USD",
            "granularity_seconds": GRANULARITY_SECONDS,
            "maximum_expected_buckets_per_request": MAX_BUCKETS_PER_REQUEST,
            "requests_per_second_max": cfg.requests_per_second,
            "user_agent": USER_AGENT,
            "request_count": len(requests),
            "request_log_hash": request_log_hash,
            "requests": requests,
        },
        "quality": {
            "expected_five_minute_rows": int(len(grid)),
            "coinbase_complete_rows": complete,
            "coinbase_missing_rows": missing,
            "coinbase_missing_fraction": missing / len(grid),
            "binance_complete_rows": int(len(binance)),
            "funding_rows": int(len(funding)),
        },
        **audit_fields(cfg, binance, funding),
        "outputs": {
            "coinbase": {
                "path": cfg.coinbase_output,
                "bytes": Path(cfg.coinbase_output).stat().st_size,
                "sha256": sha256_file(cfg.coinbase_output),
            },
            "binance": {
                "path": cfg.binance_output,
                "bytes": Path(cfg.binance_output).stat().st_size,
                "sha256": sha256_file(cfg.binance_output),
            },
            "funding": {
                "path": cfg.funding_output,
                "bytes": Path(cfg.funding_output).stat().st_size,
                "sha256": sha256_file(cfg.funding_output),
            },
        },
        "historical_snapshot_is_point_in_time": False,
        "future_data_requested": False,
    }
    manifest = {
        **core,
        "manifest_hash": canonical_hash(core),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(cfg.manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def attest_existing(cfg: Config) -> dict[str, Any]:
    """Rebuild local prefixes with hardened cutoff parsing and anchor existing download."""
    manifest_path = Path(cfg.manifest_output)
    existing = json.loads(manifest_path.read_text())
    existing_core = {
        key: value
        for key, value in existing.items()
        if key not in {"manifest_hash", "retrieved_at"}
    }
    if canonical_hash(existing_core) != existing.get("manifest_hash"):
        raise RuntimeError("existing source manifest hash mismatch")
    if existing.get("forward_trade_outcomes_opened") is not False:
        raise RuntimeError("cannot attest after forward outcomes were opened")
    prereg = json.loads(Path(cfg.preregistration).read_text())
    validate_preregistration(prereg)
    if prereg["manifest_hash"] != existing["preregistration_manifest_hash"]:
        raise RuntimeError("existing source freeze points at a different preregistration")

    grid = expected_grid(cfg.start, cfg.end)
    binance_raw = range_frame(
        resolve_existing(cfg.binance_input),
        date_column="date",
        start=cfg.start,
        end=cfg.end,
        usecols=["date", "open", "high", "low", "close", "quote_asset_volume"],
    )
    binance = validate_binance(binance_raw, grid)
    funding_raw = range_frame(
        resolve_existing(cfg.funding_input),
        date_column="date",
        start=cfg.start,
        end=cfg.end,
        usecols=["date", "funding_rate"],
    )
    funding = validate_funding(funding_raw)
    with tempfile.TemporaryDirectory(prefix="coinbase-source-attest-") as directory:
        rebuilt_binance = Path(directory) / "binance.csv.gz"
        rebuilt_funding = Path(directory) / "funding.csv.gz"
        deterministic_gzip_csv(binance, rebuilt_binance)
        deterministic_gzip_csv(funding, rebuilt_funding)
        if sha256_file(rebuilt_binance) != existing["outputs"]["binance"]["sha256"]:
            raise RuntimeError("hardened Binance prefix differs from frozen output")
        if sha256_file(rebuilt_funding) != existing["outputs"]["funding"]["sha256"]:
            raise RuntimeError("hardened funding prefix differs from frozen output")
    for metadata in existing["outputs"].values():
        if sha256_file(metadata["path"]) != metadata["sha256"]:
            raise RuntimeError("frozen source output failed attestation hash check")

    core = dict(existing_core)
    core.update(audit_fields(cfg, binance, funding))
    core["audit_amendment"] = {
        "after_source_download": True,
        "before_forward_trade_outcomes": True,
        "changed_source_values": False,
        "reason": (
            "remove implicit external fallback, hash raw inputs, enforce date-only "
            "cutoff parsing, and preserve funding milliseconds under causal as-of"
        ),
    }
    payload = {
        **core,
        "manifest_hash": canonical_hash(core),
        "retrieved_at": existing["retrieved_at"],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    parser.add_argument("--attest-existing", action="store_true")
    for name in asdict(defaults):
        if name == "attest_existing":
            continue
        option = f"--{name.replace('_', '-')}"
        value = getattr(defaults, name)
        parser.add_argument(option, type=type(value), default=value)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = attest_existing(Config(**vars(args))) if args.attest_existing else run(Config(**vars(args)))
    print(
        json.dumps(
            {
                "manifest_hash": manifest["manifest_hash"],
                "quality": manifest["quality"],
                "outputs": manifest["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
