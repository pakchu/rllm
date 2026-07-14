"""Build a verified 2023 cross-collateral book-credibility panel.

This outcome-blind builder replays the frozen Binance Vision bookDepth archives
and augments each accepted five-minute bar with scale-free within-bar depth
flicker, net change, and path activity at every +/-1..5 percent level.
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

import numpy as np
import pandas as pd

from training import build_binance_cross_collateral_book_depth_2023 as base


STATISTICS = ("log_mad", "log_net", "log_step")


@dataclass(frozen=True)
class Config:
    start: str = "2023-01-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_cross_collateral_book_credibility_btc_2023"
    manifest: str = (
        "results/binance_cross_collateral_book_credibility_btc_2023_manifest.json"
    )
    base_manifest: str = (
        "results/binance_cross_collateral_book_depth_btc_2023_manifest.json"
    )
    workers: int = 8
    retries: int = 5
    timeout_seconds: int = 60
    minimum_snapshots_per_bar: int = 8
    maximum_first_snapshot_offset_seconds: float = 60.0
    minimum_last_snapshot_offset_seconds: float = 240.0


def _stat_column(statistic: str, level: int) -> str:
    side = "m" if level < 0 else "p"
    return f"{statistic}_{side}{abs(level)}"


def aggregate_credibility(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    accepted = base.aggregate_five_minute(frame, cfg)
    if accepted.empty:
        for statistic in STATISTICS:
            for level in base.PERCENTAGES:
                accepted[_stat_column(statistic, level)] = pd.Series(dtype=float)
        return accepted

    work = frame.sort_values(["timestamp", "percentage"]).copy()
    work["date"] = work["timestamp"].dt.floor("5min")
    work["log_depth"] = np.log(work["depth"].astype(float))
    keys = ["date", "percentage"]
    grouped = work.groupby(keys, sort=True, observed=True)["log_depth"]
    center = grouped.transform("median")
    work["log_abs_deviation"] = (work["log_depth"] - center).abs()
    work["log_step"] = grouped.diff().abs()

    statistics = pd.DataFrame(
        {
            "log_mad": work.groupby(keys, sort=True, observed=True)[
                "log_abs_deviation"
            ].median(),
            "log_first": grouped.first(),
            "log_last": grouped.last(),
            "log_step": work.groupby(keys, sort=True, observed=True)[
                "log_step"
            ].mean(),
        }
    )
    statistics["log_net"] = statistics["log_last"] - statistics["log_first"]
    statistics = statistics.drop(columns=["log_first", "log_last"])

    wide_parts: list[pd.DataFrame] = []
    for statistic in STATISTICS:
        wide = statistics[statistic].unstack("percentage").reindex(
            columns=base.PERCENTAGES
        )
        wide.columns = [
            _stat_column(statistic, int(level)) for level in wide.columns
        ]
        wide_parts.append(wide)
    wide_stats = pd.concat(wide_parts, axis=1).reset_index()
    output = accepted.merge(wide_stats, on="date", validate="one_to_one")
    statistic_columns = [
        _stat_column(statistic, level)
        for statistic in STATISTICS
        for level in base.PERCENTAGES
    ]
    values = output[statistic_columns].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("accepted credibility bar contains non-finite statistics")
    if (output[[column for column in statistic_columns if "log_net" not in column]] < 0.0).any().any():
        raise ValueError("credibility dispersion/activity must be non-negative")
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
            fetcher(
                base.checksum_url(venue, symbol, day),
                retries=cfg.retries,
                timeout=cfg.timeout_seconds,
            )
        )
        payload = fetcher(
            base.archive_url(venue, symbol, day),
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
    except FileNotFoundError:
        return base._empty_day(venue, symbol, day)

    archive_hash = base.verify_sha256(payload, checksum)
    raw = base.read_archive(payload)
    day_start = pd.Timestamp(day)
    day_end = day_start + pd.Timedelta(days=1)
    if raw["timestamp"].lt(day_start).any() or raw["timestamp"].ge(day_end).any():
        raise ValueError(f"{venue} archive {day} contains another UTC date")
    bars = aggregate_credibility(raw, cfg)
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


def _verify_archive_replay(
    results: list[dict[str, Any]],
    base_manifest: dict[str, Any],
) -> None:
    frozen = {
        (item["venue"], item["date"]): item
        for item in base_manifest.get("archives", [])
    }
    if len(frozen) != len(results):
        raise ValueError("credibility archive count differs from frozen depth build")
    for result in results:
        previous = frozen.get((result["venue"], result["date"]))
        if previous is None:
            raise ValueError("credibility build contains an unknown archive day")
        for key in ("available", "archive_sha256", "raw_rows", "snapshot_count"):
            if result.get(key) != previous.get(key):
                raise ValueError(
                    f"credibility archive replay differs for {result['venue']} "
                    f"{result['date']}: {key}"
                )


def _verify_base_panel_replay(
    panel: pd.DataFrame,
    base_manifest: dict[str, Any],
) -> None:
    item = base_manifest.get("file", {})
    path = Path(item.get("path", ""))
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item.get(
        "sha256"
    ):
        raise ValueError("frozen base depth panel hash mismatch")
    frozen = pd.read_csv(path, compression="gzip", parse_dates=["date"])
    columns = frozen.columns.tolist()
    _assert_base_frame_equal(
        panel[columns].reset_index(drop=True),
        frozen.reset_index(drop=True),
    )


def _assert_base_frame_equal(
    replayed: pd.DataFrame,
    frozen: pd.DataFrame,
) -> None:
    """Allow only decimal CSV round-trip noise, never a material replay drift."""
    pd.testing.assert_frame_equal(
        replayed,
        frozen,
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-10,
    )


def build(cfg: Config) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start < date(2023, 1, 1) or end > date(2024, 1, 1):
        raise ValueError("book credibility build is physically bounded to 2023")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    if not 1 <= cfg.minimum_snapshots_per_bar <= 10:
        raise ValueError("minimum snapshots per bar must be in [1, 10]")

    base_path = Path(cfg.base_manifest)
    base_manifest = json.loads(base_path.read_text())
    if base_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("frozen base depth build opened outcomes")
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
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: (item["venue"], item["date"]))
    _verify_archive_replay(results, base_manifest)

    venue_panels: dict[str, pd.DataFrame] = {}
    for venue in base.VENUES:
        frames = [
            base._prefix_frame(item["frame"], venue)
            for item in results
            if item["venue"] == venue and item["available"]
        ]
        venue_panels[venue] = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
        )

    panel = pd.DataFrame(
        {"date": pd.date_range(start, end, freq="5min", inclusive="left")}
    )
    for venue in base.VENUES:
        panel = panel.merge(
            venue_panels[venue],
            on="date",
            how="left",
            validate="one_to_one",
        )
    required_depth = [
        f"{venue}_depth_{side}{distance}"
        for venue in base.VENUES
        for side in ("m", "p")
        for distance in range(1, 6)
    ]
    panel["source_complete"] = panel[required_depth].notna().all(axis=1)
    if panel["date"].duplicated().any() or not panel["date"].is_monotonic_increasing:
        raise ValueError("combined credibility panel timestamps are invalid")
    _verify_base_panel_replay(panel, base_manifest)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "BTC_cross_collateral_book_credibility_5m_2023.csv.gz"
    base._write_gzip_csv(panel, output)
    file_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    records = [base._public_record(item) for item in results]
    missing = {
        venue: [
            item["date"]
            for item in records
            if item["venue"] == venue and not item["available"]
        ]
        for venue in base.VENUES
    }
    manifest = {
        "protocol": {
            "name": "Binance BTC cross-collateral book-credibility 2023 panel",
            "outcomes_opened": False,
            "start_inclusive": str(pd.Timestamp(start)),
            "end_exclusive": str(pd.Timestamp(end)),
            "post_2023_rows_requested": False,
            "source": "official public Binance Vision daily archives",
            "base_depth_replayed_exactly": True,
        },
        "config": asdict(cfg),
        "venues": base.VENUES,
        "archive_root": base.BASE_URL,
        "statistics": {
            "log_mad": "median absolute deviation of log cumulative depth",
            "log_net": "last minus first log cumulative depth in the bar",
            "log_step": "mean absolute consecutive log-depth change",
        },
        "base_manifest_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "missing_archive_dates": missing,
        "archives": records,
        "file": {
            "path": str(output),
            "sha256": file_hash,
            "rows": int(len(panel)),
            "columns": int(len(panel.columns)),
            "source_complete_rows": int(panel["source_complete"].sum()),
            "first_date": str(panel["date"].min()),
            "last_date": str(panel["date"].max()),
        },
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=Config.workers)
    args = parser.parse_args()
    result = build(Config(workers=args.workers))
    print(
        json.dumps(
            {
                "outcomes_opened": result["protocol"]["outcomes_opened"],
                "base_depth_replayed_exactly": result["protocol"][
                    "base_depth_replayed_exactly"
                ],
                "missing_archive_dates": result["missing_archive_dates"],
                "file": result["file"],
                "manifest": Config.manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
