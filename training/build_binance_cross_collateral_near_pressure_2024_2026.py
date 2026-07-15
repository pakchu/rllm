#!/usr/bin/env python3
"""Build an outcome-blind future USD-M/COIN-M near-pressure panel.

Daily Binance Vision ``bookDepth`` archives are checksum-verified, reduced in
memory to five-minute near-book net-flow pressure, and discarded.  The builder
never imports a market-price or return source and retains no raw archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_binance_cross_collateral_book_depth_2023 as base
from training import build_binance_cross_collateral_book_shells_2023 as shells


DEFAULT_OUTPUT_DIR = "data/binance_cross_collateral_near_pressure_btc_2024_2026"
DEFAULT_MANIFEST = "results/binance_cross_collateral_near_pressure_btc_2024_2026_manifest.json"


@dataclass(frozen=True)
class Config:
    start: str = "2024-01-01"
    end: str = "2026-06-02"
    output_dir: str = DEFAULT_OUTPUT_DIR
    manifest: str = DEFAULT_MANIFEST
    workers: int = 8
    retries: int = 5
    timeout_seconds: int = 60
    minimum_snapshots_per_bar: int = 8
    maximum_first_snapshot_offset_seconds: float = 60.0
    minimum_last_snapshot_offset_seconds: float = 240.0


def aggregate_near_pressure(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    panel = shells.aggregate_shells(frame, cfg)
    if panel.empty:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "near_pressure": pd.Series(dtype=float),
                "snapshot_count": pd.Series(dtype=float),
                "first_offset_seconds": pd.Series(dtype=float),
                "last_offset_seconds": pd.Series(dtype=float),
            }
        )
    bid = panel["shell_flow_net_m1"] + 0.5 * panel["shell_flow_net_m2"]
    ask = panel["shell_flow_net_p1"] + 0.5 * panel["shell_flow_net_p2"]
    output = panel[["date", "snapshot_count", "first_offset_seconds", "last_offset_seconds"]].copy()
    output["near_pressure"] = bid - ask
    if output["near_pressure"].isna().any():
        raise ValueError("accepted near-pressure bar is non-finite")
    return output


def process_day(
    venue: str,
    symbol: str,
    day: date,
    cfg: Config,
    *,
    fetcher: Callable[..., bytes] = base._fetch_bytes,
) -> dict[str, Any]:
    try:
        checksum = base.expected_sha256(
            fetcher(base.checksum_url(venue, symbol, day), retries=cfg.retries, timeout=cfg.timeout_seconds)
        )
        payload = fetcher(
            base.archive_url(venue, symbol, day), retries=cfg.retries, timeout=cfg.timeout_seconds
        )
    except FileNotFoundError:
        return base._empty_day(venue, symbol, day)
    archive_hash = base.verify_sha256(payload, checksum)
    raw = base.read_archive(payload)
    day_start = pd.Timestamp(day)
    day_end = day_start + pd.Timedelta(days=1)
    if raw["timestamp"].lt(day_start).any() or raw["timestamp"].ge(day_end).any():
        raise ValueError(f"{venue} archive {day} contains another UTC date")
    bars = aggregate_near_pressure(raw, cfg)
    return {
        "venue": venue,
        "symbol": symbol,
        "date": day.isoformat(),
        "available": True,
        "archive_sha256": archive_hash,
        "raw_rows": int(len(raw)),
        "snapshot_count": int(raw["timestamp"].nunique()),
        "accepted_bar_count": int(len(bars)),
        "first_timestamp": str(raw["timestamp"].min()),
        "last_timestamp": str(raw["timestamp"].max()),
        "frame": bars,
    }


def validate_config(cfg: Config) -> tuple[date, date]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start < date(2024, 1, 1) or end > date(2026, 6, 2):
        raise ValueError("future near-pressure build is physically bounded to 2024-01-01..2026-06-02")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    if not 1 <= cfg.minimum_snapshots_per_bar <= 10:
        raise ValueError("minimum snapshots per bar must be in [1, 10]")
    return start, end


def build(cfg: Config) -> dict[str, Any]:
    start, end = validate_config(cfg)
    tasks = [
        (venue, symbol, day)
        for venue, symbol in base.VENUES.items()
        for day in base._days(start, end)
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(process_day, venue, symbol, day, cfg): (venue, day)
            for venue, symbol, day in tasks
        }
        for completed, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if completed % 100 == 0:
                print(f"processed {completed}/{len(tasks)} venue-days", flush=True)
    results.sort(key=lambda item: (item["venue"], item["date"]))

    panel = pd.DataFrame({"date": pd.date_range(start, end, freq="5min", inclusive="left")})
    for venue in base.VENUES:
        frames = [
            item["frame"].rename(
                columns={column: f"{venue}_{column}" for column in item["frame"] if column != "date"}
            )
            for item in results
            if item["venue"] == venue and item["available"]
        ]
        venue_panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"date": []})
        panel = panel.merge(venue_panel, on="date", how="left", validate="one_to_one")
    panel["source_complete"] = panel[["um_near_pressure", "cm_near_pressure"]].notna().all(axis=1)
    if panel["date"].duplicated().any() or not panel["date"].is_monotonic_increasing:
        raise ValueError("combined near-pressure timestamps are invalid")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"BTC_cross_collateral_near_pressure_5m_{cfg.start}_{cfg.end}.csv.gz"
    base._write_gzip_csv(panel, output)
    public = [base._public_record(item) for item in results]
    missing = {
        venue: [item["date"] for item in public if item["venue"] == venue and not item["available"]]
        for venue in base.VENUES
    }
    manifest = {
        "schema_version": 1,
        "protocol": {
            "name": "Binance BTC cross-collateral near-pressure future panel",
            "source": "official public Binance Vision daily bookDepth archives",
            "outcomes_opened": False,
            "price_or_return_loaded": False,
            "raw_archives_retained": False,
            "checksums_verified": True,
            "start_inclusive": cfg.start,
            "end_exclusive": cfg.end,
        },
        "config": asdict(cfg),
        "venues": base.VENUES,
        "formula": "(flow_net_bid1 + 0.5*flow_net_bid2) - (flow_net_ask1 + 0.5*flow_net_ask2)",
        "missing_archive_dates": missing,
        "archives": public,
        "file": {
            "path": str(output),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "rows": int(len(panel)),
            "columns": int(len(panel.columns)),
            "source_complete_rows": int(panel["source_complete"].sum()),
            "first_date": str(panel["date"].min()),
            "last_date": str(panel["date"].max()),
        },
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=Config.start)
    parser.add_argument("--end", default=Config.end)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=Config.workers)
    parser.add_argument("--retries", type=int, default=Config.retries)
    parser.add_argument("--timeout-seconds", type=int, default=Config.timeout_seconds)
    return parser.parse_args()


def main() -> None:
    manifest = build(Config(**vars(parse_args())))
    print(json.dumps({"file": manifest["file"], "missing": manifest["missing_archive_dates"]}, indent=2))


if __name__ == "__main__":
    main()
