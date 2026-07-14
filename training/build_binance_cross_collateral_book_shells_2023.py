"""Build an outcome-blind 2023 non-overlapping book-shell panel.

Binance bookDepth levels are cumulative.  This builder first differences each
complete 30-second snapshot into radial 0-1, 1-2, ..., 4-5 percent shells,
then computes dimensionless five-minute shell path statistics.  It never
loads a market price or return outcome and retains no raw archive.
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

from training import build_binance_cross_collateral_book_credibility_2023 as cred
from training import build_binance_cross_collateral_book_depth_2023 as base


STATISTICS = (
    "share_median",
    "flow_net",
    "flow_add",
    "flow_withdraw",
    "flow_churn",
    "flow_efficiency",
)


@dataclass(frozen=True)
class Config:
    start: str = "2023-01-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_cross_collateral_book_shells_btc_2023"
    manifest: str = (
        "results/binance_cross_collateral_book_shells_btc_2023_manifest.json"
    )
    base_manifest: str = (
        "results/binance_cross_collateral_book_depth_btc_2023_manifest.json"
    )
    credibility_manifest: str = (
        "results/binance_cross_collateral_book_credibility_btc_2023_manifest.json"
    )
    workers: int = 8
    retries: int = 5
    timeout_seconds: int = 60
    minimum_snapshots_per_bar: int = 8
    maximum_first_snapshot_offset_seconds: float = 60.0
    minimum_last_snapshot_offset_seconds: float = 240.0


def _shell_column(statistic: str, side: str, shell: int) -> str:
    return f"shell_{statistic}_{side}{shell}"


def _snapshot_shells(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    if side not in ("m", "p"):
        raise ValueError("book shell side must be m or p")
    levels = [-1, -2, -3, -4, -5] if side == "m" else [1, 2, 3, 4, 5]
    pivot = frame.pivot(
        index="timestamp",
        columns="percentage",
        values="depth",
    ).loc[:, levels]
    cumulative = pivot.to_numpy(float)
    shells = np.column_stack(
        [cumulative[:, 0], np.diff(cumulative, axis=1)]
    )
    tolerance = np.maximum(1.0, cumulative[:, -1:]) * 1e-12
    if np.any(shells < -tolerance):
        raise ValueError("cumulative depth produced a negative radial shell")
    shells = np.maximum(shells, 0.0)
    total = cumulative[:, -1]
    if np.any(total <= 0.0) or not np.isfinite(total).all():
        raise ValueError("radial shell total depth must be positive and finite")
    shares = shells / total[:, None]
    if not np.allclose(shares.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("radial shell shares do not sum to one")
    output = pd.DataFrame(
        {
            "timestamp": pivot.index,
            "date": pivot.index.floor("5min"),
            "total": total,
        }
    )
    for shell in range(1, 6):
        output[f"mass_{shell}"] = shells[:, shell - 1]
        output[f"share_{shell}"] = shares[:, shell - 1]
    return output


def _aggregate_side_shells(
    snapshots: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    work = snapshots.sort_values("timestamp").copy()
    grouped = work.groupby("date", sort=True, observed=True)
    output: dict[str, pd.Series] = {}
    prior_total = grouped["total"].shift(1)
    total_scale = 0.5 * (work["total"] + prior_total)
    for shell in range(1, 6):
        output[_shell_column("share_median", side, shell)] = grouped[
            f"share_{shell}"
        ].median()
        flow = grouped[f"mass_{shell}"].diff() / total_scale
        flow_groups = flow.groupby(work["date"], sort=True, observed=True)
        net = flow_groups.sum(min_count=1)
        additions = flow.clip(lower=0.0).groupby(
            work["date"], sort=True, observed=True
        ).sum(min_count=1)
        withdrawals = (-flow).clip(lower=0.0).groupby(
            work["date"], sort=True, observed=True
        ).sum(min_count=1)
        churn = additions + withdrawals
        efficiency = (net.abs() / churn.replace(0.0, np.nan)).fillna(0.0)
        output[_shell_column("flow_net", side, shell)] = net
        output[_shell_column("flow_add", side, shell)] = additions
        output[_shell_column("flow_withdraw", side, shell)] = withdrawals
        output[_shell_column("flow_churn", side, shell)] = churn
        output[_shell_column("flow_efficiency", side, shell)] = efficiency
    return pd.DataFrame(output).rename_axis("date").reset_index()


def aggregate_shells(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    accepted = base.aggregate_five_minute(frame, cfg)
    shell_columns = [
        _shell_column(statistic, side, shell)
        for statistic in STATISTICS
        for side in ("m", "p")
        for shell in range(1, 6)
    ]
    if accepted.empty:
        for column in shell_columns:
            accepted[column] = pd.Series(dtype=float)
        return accepted

    side_panels = [
        _aggregate_side_shells(_snapshot_shells(frame, side), side)
        for side in ("m", "p")
    ]
    output = accepted.copy()
    for panel in side_panels:
        output = output.merge(panel, on="date", validate="one_to_one")
    values = output[shell_columns].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("accepted radial-shell bar contains non-finite values")
    bounded = [
        column
        for column in shell_columns
        if "share_median" in column or "flow_efficiency" in column
    ]
    if (
        (output[bounded] < -1e-12).any().any()
        or (output[bounded] > 1.0 + 1e-12).any().any()
    ):
        raise ValueError("radial-shell share statistic is outside [0, 1]")
    nonnegative = [
        column
        for column in shell_columns
        if any(
            statistic in column
            for statistic in ("flow_add", "flow_withdraw", "flow_churn")
        )
    ]
    if (output[nonnegative] < -1e-12).any().any():
        raise ValueError("radial-shell flow magnitude must be non-negative")
    for side in ("m", "p"):
        for shell in range(1, 6):
            net = output[_shell_column("flow_net", side, shell)]
            additions = output[_shell_column("flow_add", side, shell)]
            withdrawals = output[_shell_column("flow_withdraw", side, shell)]
            churn = output[_shell_column("flow_churn", side, shell)]
            if not np.allclose(net, additions - withdrawals, atol=1e-10, rtol=0.0):
                raise ValueError("radial-shell net flow identity failed")
            if not np.allclose(churn, additions + withdrawals, atol=1e-10, rtol=0.0):
                raise ValueError("radial-shell churn identity failed")
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
    bars = aggregate_shells(raw, cfg)
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


def build(cfg: Config) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start < date(2023, 1, 1) or end > date(2024, 1, 1):
        raise ValueError("book-shell build is physically bounded to 2023")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    if not 1 <= cfg.minimum_snapshots_per_bar <= 10:
        raise ValueError("minimum snapshots per bar must be in [1, 10]")
    if not 0.0 <= cfg.maximum_first_snapshot_offset_seconds < 300.0:
        raise ValueError("first snapshot offset bound is invalid")
    if not 0.0 <= cfg.minimum_last_snapshot_offset_seconds < 300.0:
        raise ValueError("last snapshot offset bound is invalid")

    base_path = Path(cfg.base_manifest)
    base_manifest = json.loads(base_path.read_text())
    credibility_path = Path(cfg.credibility_manifest)
    credibility_manifest = json.loads(credibility_path.read_text())
    for name, manifest in (
        ("base", base_manifest),
        ("credibility", credibility_manifest),
    ):
        protocol = manifest.get("protocol", {})
        if protocol.get("outcomes_opened") is not False:
            raise ValueError(f"frozen {name} build opened outcomes")
        if protocol.get("post_2023_rows_requested") is not False:
            raise ValueError(f"frozen {name} build requested post-2023 rows")

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
    cred._verify_archive_replay(results, credibility_manifest)

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
            venue_panels[venue], on="date", how="left", validate="one_to_one"
        )
    required_depth = [
        f"{venue}_depth_{side}{distance}"
        for venue in base.VENUES
        for side in ("m", "p")
        for distance in range(1, 6)
    ]
    panel["source_complete"] = panel[required_depth].notna().all(axis=1)
    if panel["date"].duplicated().any() or not panel["date"].is_monotonic_increasing:
        raise ValueError("combined radial-shell panel timestamps are invalid")
    cred._verify_base_panel_replay(panel, base_manifest)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "BTC_cross_collateral_book_shells_5m_2023.csv.gz"
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
            "name": "Binance BTC cross-collateral radial book-shell 2023 panel",
            "outcomes_opened": False,
            "start_inclusive": str(pd.Timestamp(start)),
            "end_exclusive": str(pd.Timestamp(end)),
            "post_2023_rows_requested": False,
            "source": "official public Binance Vision daily archives",
            "base_depth_replayed_exactly": True,
            "raw_archives_retained": False,
        },
        "config": asdict(cfg),
        "venues": base.VENUES,
        "archive_root": base.BASE_URL,
        "shell_definition": {
            "1": "0-1 percent",
            "2": "1-2 percent",
            "3": "2-3 percent",
            "4": "3-4 percent",
            "5": "4-5 percent",
        },
        "statistics": {
            "share_median": "median shell depth divided by cumulative 5% depth",
            "flow_net": "sum of normalized consecutive signed shell-depth changes",
            "flow_add": "sum of positive normalized consecutive shell-depth changes",
            "flow_withdraw": "sum of negative-change magnitudes",
            "flow_churn": "flow_add plus flow_withdraw",
            "flow_efficiency": "absolute flow_net divided by flow_churn, or zero",
        },
        "base_manifest_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "credibility_manifest_sha256": hashlib.sha256(
            credibility_path.read_bytes()
        ).hexdigest(),
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
