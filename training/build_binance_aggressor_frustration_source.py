"""Build outcome-blind BAFR features from official Binance USD-M aggTrades."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from preprocessing.aggressor_frustration import (
    AggTradeTickState,
    BAR_COLUMNS,
    aggregate_frustration_five_minute,
    classify_tick_arrays,
)
from training import build_binance_aggtrade_microstructure as base


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BuildConfig:
    symbol: str = "BTCUSDT"
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_um_aggressor_frustration_btc_2020_2023"
    workers: int = 4
    retries: int = 5
    timeout_seconds: int = 60
    overwrite: bool = False


def _state_dict(state: AggTradeTickState) -> dict[str, int | float | None]:
    return {
        "previous_price": state.previous_price,
        "last_nonzero_tick": state.last_nonzero_tick,
        "previous_agg_trade_id": state.previous_agg_trade_id,
    }


def _fetch_verified_day(
    day: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes],
) -> tuple[bytes, str]:
    checksum_payload = fetcher(
        base.checksum_url(cfg.symbol, day),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    expected = base.expected_sha256(checksum_payload)
    payload = fetcher(
        base.archive_url(cfg.symbol, day),
        retries=cfg.retries,
        timeout=cfg.timeout_seconds,
    )
    return payload, base.verify_sha256(payload, expected)


def _warmup_state(
    day: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes],
) -> tuple[AggTradeTickState, dict[str, Any]]:
    try:
        payload, archive_hash = _fetch_verified_day(day, cfg, fetcher=fetcher)
    except FileNotFoundError:
        return AggTradeTickState(), {"date": day.isoformat(), "status": "unavailable"}

    raw = base.read_archive(payload)
    _, _, reset, state = classify_tick_arrays(
        raw["agg_trade_id"].to_numpy(dtype="int64", copy=False),
        raw["price"].to_numpy(dtype="float64", copy=False),
    )
    return state, {
        "date": day.isoformat(),
        "status": "verified",
        "archive_sha256": archive_hash,
        "agg_trade_rows": int(len(raw)),
        "first_agg_trade_id": int(raw["agg_trade_id"].iloc[0]),
        "last_agg_trade_id": int(raw["agg_trade_id"].iloc[-1]),
        "state_reset_count": int(reset.sum()),
        "state_out": _state_dict(state),
    }


def _resume_metadata_is_current(
    metadata: dict[str, Any],
    *,
    cfg: BuildConfig,
    month: date,
    expected_days: list[date],
    output_path: Path,
    fetcher: Callable[..., bytes],
) -> bool:
    expected_dates = [day.isoformat() for day in expected_days]
    archives = metadata.get("archives")
    warmup = metadata.get("warmup")
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("month") != f"{month:%Y-%m}"
        or metadata.get("symbol") != cfg.symbol
        or metadata.get("requested_dates") != expected_dates
        or not isinstance(archives, list)
        or [item.get("date") for item in archives] != expected_dates
        or not isinstance(warmup, dict)
        or warmup.get("date") != (expected_days[0] - timedelta(days=1)).isoformat()
    ):
        return False
    if hashlib.sha256(output_path.read_bytes()).hexdigest() != metadata.get("output_sha256"):
        raise ValueError(f"resume artifact hash mismatch: {output_path}")

    for day, archive in zip(expected_days, archives, strict=True):
        current = base.expected_sha256(
            fetcher(
                base.checksum_url(cfg.symbol, day),
                retries=cfg.retries,
                timeout=cfg.timeout_seconds,
            )
        )
        if current != archive.get("archive_sha256"):
            return False

    warmup_day = expected_days[0] - timedelta(days=1)
    try:
        current_warmup = base.expected_sha256(
            fetcher(
                base.checksum_url(cfg.symbol, warmup_day),
                retries=cfg.retries,
                timeout=cfg.timeout_seconds,
            )
        )
    except FileNotFoundError:
        return warmup.get("status") == "unavailable"
    return warmup.get("status") == "verified" and current_warmup == warmup.get(
        "archive_sha256"
    )


def _process_month(
    month: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = base._fetch_bytes,
) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    expected_days = base._month_days(month, start, end)
    if not expected_days:
        raise ValueError(f"month {month:%Y-%m} has no requested days")

    output_dir = Path(cfg.output_dir)
    monthly_dir = output_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.symbol}_aggressor_frustration_5m_{month:%Y-%m}"
    output_path = monthly_dir / f"{stem}.csv.gz"
    metadata_path = monthly_dir / f"{stem}.json"
    if output_path.exists() and metadata_path.exists() and not cfg.overwrite:
        metadata = json.loads(metadata_path.read_text())
        if _resume_metadata_is_current(
            metadata,
            cfg=cfg,
            month=month,
            expected_days=expected_days,
            output_path=output_path,
            fetcher=fetcher,
        ):
            return metadata

    warmup_day = expected_days[0] - timedelta(days=1)
    state, warmup = _warmup_state(warmup_day, cfg, fetcher=fetcher)
    frames: list[pd.DataFrame] = []
    archives: list[dict[str, Any]] = []
    for day in expected_days:
        payload, archive_hash = _fetch_verified_day(day, cfg, fetcher=fetcher)
        raw = base.read_archive(payload)
        state_in = state
        bars, state = aggregate_frustration_five_minute(raw, initial_state=state)
        day_start = pd.Timestamp(day)
        day_end = day_start + pd.Timedelta("1d")
        if not ((bars["date"] >= day_start) & (bars["date"] < day_end)).all():
            raise ValueError(f"archive contains timestamps outside {day}")
        frames.append(bars)
        archives.append(
            {
                "date": day.isoformat(),
                "archive_sha256": archive_hash,
                "agg_trade_rows": int(len(raw)),
                "five_minute_rows": int(len(bars)),
                "first_agg_trade_id": int(raw["agg_trade_id"].iloc[0]),
                "last_agg_trade_id": int(raw["agg_trade_id"].iloc[-1]),
                "state_reset_count": int(bars["state_reset_count"].sum()),
                "unavailable_tick_count": int(bars["unavailable_tick_count"].sum()),
                "state_in": _state_dict(state_in),
                "state_out": _state_dict(state),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    if combined["date"].duplicated().any() or not combined["date"].is_monotonic_increasing:
        raise ValueError(f"month {month:%Y-%m} has duplicate or unordered bins")
    base._write_gzip_csv(combined, output_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "month": f"{month:%Y-%m}",
        "symbol": cfg.symbol,
        "requested_dates": [day.isoformat() for day in expected_days],
        "warmup": warmup,
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "first_date": str(combined["date"].min()),
        "last_date": str(combined["date"].max()),
        "columns": list(combined.columns),
        "archives": archives,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def build(
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = base._fetch_bytes,
) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")

    months = base._month_starts(start, end)
    metadata: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        future_map = {
            executor.submit(_process_month, month, cfg, fetcher=fetcher): month
            for month in months
        }
        for future in as_completed(future_map):
            month = future_map[future]
            result = future.result()
            metadata.append(result)
            print(f"completed {month:%Y-%m}: rows={result['rows']}", flush=True)
    metadata.sort(key=lambda item: item["month"])

    monthly_frames = [
        pd.read_csv(item["output"], compression="gzip", parse_dates=["date"])
        for item in metadata
    ]
    combined = (
        pd.concat(monthly_frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    if combined["date"].duplicated().any() or not combined["date"].is_monotonic_increasing:
        raise ValueError("combined BAFR output has duplicate or unordered timestamps")
    if tuple(combined.columns) != BAR_COLUMNS:
        raise ValueError("combined BAFR output schema changed")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_day = end - timedelta(days=1)
    combined_path = output_dir / (
        f"{cfg.symbol}_aggressor_frustration_5m_{cfg.start}_{last_day}.csv.gz"
    )
    base._write_gzip_csv(combined, combined_path)
    manifest = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "config": asdict(cfg),
        "protocol": {
            "source": "official Binance USD-M Futures daily aggTrades archives",
            "archive_checksums_verified": True,
            "end_is_exclusive": True,
            "five_minute_bin": "UTC floor of aggregate-trade transaction timestamp",
            "buyer_maker_semantics": "true = buyer passive / seller aggressive",
            "tick_rule": "last nonzero trade-price tick carried only across contiguous aggregate IDs",
            "gap_rule": "aggregate-ID discontinuity resets price and tick state",
            "month_warmup": "verified prior-day archive, or unavailable state",
            "raw_archives_persisted": False,
            "outcomes_opened": False,
        },
        "combined_output": str(combined_path),
        "combined_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "first_date": str(combined["date"].min()),
        "last_date": str(combined["date"].max()),
        "columns": list(combined.columns),
        "months": metadata,
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=BuildConfig.symbol)
    parser.add_argument("--start", default=BuildConfig.start)
    parser.add_argument("--end", default=BuildConfig.end)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--workers", type=int, default=BuildConfig.workers)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build(BuildConfig(**vars(args)))
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in ("combined_output", "combined_sha256", "rows", "first_date", "last_date")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
