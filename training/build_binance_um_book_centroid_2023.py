"""Build a verified 2023 USD-M BTCUSDT book-depth centroid-skew panel.

This outcome-blind source builder reuses the official Binance Vision URL,
checksum, archive parsing, fetch, gzip, and five-minute timing acceptance
contracts from ``training.build_binance_cross_collateral_book_depth_2023``.
It transforms only USD-M BTCUSDT bookDepth cumulative depth/notional snapshots
into scale-free directional centroid-skew features and never reads prices or
returns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from training import build_binance_cross_collateral_book_depth_2023 as base

VENUE = "um"
SYMBOL = "BTCUSDT"
SKEW_DISTANCES = (2, 3, 4, 5)
SCHEMA_VERSION = 1
REFERENCE_MANIFEST_SHA256 = (
    "95ec6e133dfcc7ed3c058538f380d24d98552c0a921fc24a679d247159a4f080"
)


@dataclass(frozen=True)
class Config:
    start: str = "2023-01-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_um_book_centroid_btcusdt_2023"
    manifest: str = "results/binance_um_book_centroid_btcusdt_2023_manifest.json"
    reference_manifest: str = (
        "results/binance_cross_collateral_book_depth_btc_2023_manifest.json"
    )
    workers: int = 8
    retries: int = 5
    timeout_seconds: int = 60
    minimum_snapshots_per_bar: int = base.Config.minimum_snapshots_per_bar
    maximum_first_snapshot_offset_seconds: float = (
        base.Config.maximum_first_snapshot_offset_seconds
    )
    minimum_last_snapshot_offset_seconds: float = (
        base.Config.minimum_last_snapshot_offset_seconds
    )


def archive_url(day: date) -> str:
    """Official Binance Vision USD-M BTCUSDT bookDepth archive URL."""
    return base.archive_url(VENUE, SYMBOL, day)


def checksum_url(day: date) -> str:
    """Official Binance Vision USD-M BTCUSDT bookDepth checksum URL."""
    return base.checksum_url(VENUE, SYMBOL, day)


def _base_timing_config(cfg: Config) -> base.Config:
    return base.Config(
        start=cfg.start,
        end=cfg.end,
        workers=cfg.workers,
        retries=cfg.retries,
        timeout_seconds=cfg.timeout_seconds,
        minimum_snapshots_per_bar=cfg.minimum_snapshots_per_bar,
        maximum_first_snapshot_offset_seconds=cfg.maximum_first_snapshot_offset_seconds,
        minimum_last_snapshot_offset_seconds=cfg.minimum_last_snapshot_offset_seconds,
    )


def _load_reference_records(path: str | Path) -> dict[str, dict[str, Any]]:
    manifest_path = Path(path)
    observed_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if observed_hash != REFERENCE_MANIFEST_SHA256:
        raise ValueError("frozen cross-collateral reference manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("frozen cross-collateral reference opened outcomes")
    if protocol.get("post_2023_rows_requested") is not False:
        raise ValueError("frozen cross-collateral reference requested post-2023 rows")
    records: dict[str, dict[str, Any]] = {}
    for item in manifest["archives"]:
        if item.get("venue") == VENUE and item.get("symbol") == SYMBOL:
            if item["date"] in records:
                raise ValueError(f"duplicate frozen USD-M archive date: {item['date']}")
            records[item["date"]] = item
    return records


def _compare_reference(record: dict[str, Any], reference: dict[str, Any]) -> None:
    keys = ("available", "archive_sha256", "raw_rows", "snapshot_count")
    mismatches = {
        key: {"actual": record.get(key), "expected": reference.get(key)}
        for key in keys
        if record.get(key) != reference.get(key)
    }
    if mismatches:
        raise ValueError(
            "USD-M bookDepth archive replay mismatches frozen cross-collateral "
            f"manifest for {record['date']}: {mismatches}"
        )


def read_archive(payload: bytes) -> pd.DataFrame:
    """Parse and validate the official bookDepth archive via the base builder."""
    return base.read_archive(payload)


def snapshots_to_skew(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert complete snapshots to directional centroid-skew values.

    The cumulative average quote at each level is notional/depth. Bid averages
    must move non-increasingly outward, ask averages non-decreasingly outward,
    and every bid level must stay below its corresponding ask level.
    """
    avg = raw.copy()
    avg["avg_quote"] = avg["notional"] / avg["depth"]
    if not np.isfinite(avg["avg_quote"].to_numpy(float)).all():
        raise ValueError("book-depth average quote contains non-finite values")
    if (avg["avg_quote"] <= 0.0).any():
        raise ValueError("book-depth average quote contains non-positive values")

    pivot = avg.pivot(index="timestamp", columns="percentage", values="avg_quote")
    required = list(base.PERCENTAGES)
    if pivot.reindex(columns=required).isna().any().any():
        raise ValueError("book-depth snapshot does not contain all +/-1..5 levels")
    pivot = pivot.reindex(columns=required)

    bid = pivot.loc[:, [-1, -2, -3, -4, -5]].to_numpy(float)
    ask = pivot.loc[:, [1, 2, 3, 4, 5]].to_numpy(float)
    if (np.diff(bid, axis=1) > 0.0).any():
        raise ValueError("bid average quote is not non-increasing outward")
    if (np.diff(ask, axis=1) < 0.0).any():
        raise ValueError("ask average quote is not non-decreasing outward")
    if (bid >= ask).any():
        raise ValueError("book-depth average quotes are crossed")

    out = pd.DataFrame({"timestamp": pivot.index})
    for distance in SKEW_DISTANCES:
        out[f"skew_{distance}"] = (
            np.log((pivot[distance] / pivot[1]).to_numpy(float))
            - np.log((pivot[-1] / pivot[-distance]).to_numpy(float))
        )
    skew_columns = [f"skew_{distance}" for distance in SKEW_DISTANCES]
    if not np.isfinite(out[skew_columns].to_numpy(float)).all():
        raise ValueError("book-depth centroid skew contains non-finite values")
    return out.sort_values("timestamp").reset_index(drop=True)


def aggregate_five_minute(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Aggregate accepted five-minute bars using the base timing contract."""
    accepted_timing = base.aggregate_five_minute(raw, _base_timing_config(cfg))
    if accepted_timing.empty:
        return pd.DataFrame()
    accepted_timing = accepted_timing[
        ["date", "snapshot_count", "first_offset_seconds", "last_offset_seconds"]
    ].copy()

    skew = snapshots_to_skew(raw)
    work = skew.copy()
    work["date"] = work["timestamp"].dt.floor("5min")
    accepted_dates = set(accepted_timing["date"].tolist())

    rows: list[dict[str, Any]] = []
    skew_columns = [f"skew_{distance}" for distance in SKEW_DISTANCES]
    for bar_date, group in work.groupby("date", sort=True, observed=True):
        if bar_date not in accepted_dates:
            continue
        ordered = group.sort_values("timestamp")
        row: dict[str, Any] = {"date": bar_date}
        for column in skew_columns:
            values = ordered[column].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"accepted {column} values are non-finite")
            diffs = np.diff(values)
            net = float(values[-1] - values[0])
            path = float(np.abs(diffs).sum())
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_net"] = net
            row[f"{column}_path"] = path
            row[f"{column}_efficiency"] = 0.0 if path == 0.0 else abs(net) / path
        rows.append(row)

    output = pd.DataFrame(rows).merge(
        accepted_timing, on="date", how="inner", validate="one_to_one"
    )
    if output.empty:
        return output
    value_columns = [column for column in output.columns if column != "date"]
    if not np.isfinite(output[value_columns].to_numpy(float)).all():
        raise ValueError("accepted centroid bar contains non-finite values")
    return output.reset_index(drop=True)


def _empty_day(day: date) -> dict[str, Any]:
    return {
        "venue": VENUE,
        "symbol": SYMBOL,
        "date": day.isoformat(),
        "available": False,
        "reason": "official archive or checksum not published",
        "frame": pd.DataFrame(),
    }


def process_day(
    day: date,
    cfg: Config,
    reference: dict[str, Any] | None = None,
    *,
    fetcher: Callable[..., bytes] = base._fetch_bytes,
) -> dict[str, Any]:
    try:
        checksum = base.expected_sha256(
            fetcher(
                checksum_url(day),
                retries=cfg.retries,
                timeout=cfg.timeout_seconds,
            )
        )
        payload = fetcher(
            archive_url(day),
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
    except FileNotFoundError:
        result = _empty_day(day)
        if reference is not None:
            _compare_reference(result, reference)
        return result

    archive_hash = base.verify_sha256(payload, checksum)
    raw = read_archive(payload)
    day_start = pd.Timestamp(day)
    day_end = day_start + pd.Timedelta(days=1)
    if raw["timestamp"].lt(day_start).any() or raw["timestamp"].ge(day_end).any():
        raise ValueError(f"USD-M BTCUSDT archive {day} contains another UTC date")
    bars = aggregate_five_minute(raw, cfg)
    result = {
        "venue": VENUE,
        "symbol": SYMBOL,
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
    if reference is not None:
        _compare_reference(result, reference)
    return result


def _public_record(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "frame"}


def _validate_config(cfg: Config) -> tuple[date, date]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start != date(2023, 1, 1) or end != date(2024, 1, 1):
        raise ValueError(
            "USD-M book-centroid build is physically sealed to [2023-01-01, 2024-01-01)"
        )
    # Reuse the base builder's timing/config validation semantics.
    base_cfg = _base_timing_config(cfg)
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    if not 1 <= base_cfg.minimum_snapshots_per_bar <= 10:
        raise ValueError("minimum snapshots per bar must be in [1, 10]")
    if not 0.0 <= base_cfg.maximum_first_snapshot_offset_seconds < 300.0:
        raise ValueError("first snapshot offset bound is invalid")
    if not 0.0 <= base_cfg.minimum_last_snapshot_offset_seconds < 300.0:
        raise ValueError("last snapshot offset bound is invalid")
    return start, end


def _feature_distributions(panel: pd.DataFrame) -> dict[str, dict[str, float]]:
    distributions: dict[str, dict[str, float]] = {}
    feature_columns = [
        column
        for column in panel.columns
        if column.startswith("skew_") and panel[column].notna().any()
    ]
    for column in feature_columns:
        values = panel.loc[panel["source_complete"], column].dropna().to_numpy(float)
        if values.size == 0:
            continue
        distributions[column] = {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
    return distributions


def build(cfg: Config) -> dict[str, Any]:
    start, end = _validate_config(cfg)
    reference = _load_reference_records(cfg.reference_manifest)
    days = base._days(start, end)
    expected_dates = {day.isoformat() for day in days}
    missing_reference = sorted(expected_dates - set(reference))
    if missing_reference:
        raise ValueError(f"reference manifest is missing USD-M dates: {missing_reference}")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(process_day, day, cfg, reference.get(day.isoformat())): day
            for day in days
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["date"])

    extra_reference = sorted(set(reference) - {item["date"] for item in results})
    if extra_reference:
        raise ValueError(f"reference manifest contains unmatched USD-M dates: {extra_reference}")

    frames = [item["frame"] for item in results if item["available"]]
    features = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
    )
    full_grid = pd.DataFrame(
        {"date": pd.date_range(start, end, freq="5min", inclusive="left")}
    )
    panel = full_grid.merge(features, on="date", how="left", validate="one_to_one")
    feature_columns = [column for column in panel if column.startswith("skew_")]
    timing_columns = ["snapshot_count", "first_offset_seconds", "last_offset_seconds"]
    required_columns = feature_columns + timing_columns
    panel["source_complete"] = panel[required_columns].notna().all(axis=1)
    panel["source_available_at"] = panel["date"] + pd.Timedelta(minutes=5)
    if panel["date"].duplicated().any() or not panel["date"].is_monotonic_increasing:
        raise ValueError("centroid panel timestamps are invalid")
    if panel.loc[panel["source_complete"], required_columns].empty:
        raise ValueError("no accepted centroid bars were produced")
    if not np.isfinite(
        panel.loc[panel["source_complete"], required_columns].to_numpy(float)
    ).all():
        raise ValueError("accepted centroid panel contains non-finite values")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "BTCUSDT_um_book_centroid_skew_5m_2023.csv.gz"
    base._write_gzip_csv(panel, output)
    file_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    records = [_public_record(item) for item in results]
    missing = [item["date"] for item in records if not item["available"]]
    manifest = {
        "protocol": {
            "name": "Binance USD-M BTCUSDT book-depth notional-centroid 2023 panel",
            "schema_version": SCHEMA_VERSION,
            "outcomes_opened": False,
            "start_inclusive": str(pd.Timestamp(start)),
            "end_exclusive": str(pd.Timestamp(end)),
            "post_2023_rows_requested": False,
            "source": "official public Binance Vision USD-M BTCUSDT bookDepth daily archives",
            "source_semantics_caveat": (
                "Archive URL, checksum, and raw fields are official Binance Vision data; "
                "the notional/depth centroid-skew transformation is research inference."
            ),
            "source_availability": "bar features are available at bar open + 5 minutes",
            "price_or_return_inputs_opened": False,
        },
        "config": asdict(cfg),
        "venue": VENUE,
        "symbol": SYMBOL,
        "archive_root": base.BASE_URL,
        "official_urls": {
            "archive_template": archive_url(date(2023, 1, 1)).replace("2023-01-01", "{YYYY-MM-DD}"),
            "checksum_template": checksum_url(date(2023, 1, 1)).replace("2023-01-01", "{YYYY-MM-DD}"),
        },
        "feature_definition": {
            "avg_quote": "notional / cumulative depth at each complete +/-1..5% raw snapshot level",
            "skew_k": "log(ask_avg_k/ask_avg_1) - log(bid_avg_1/bid_avg_k), k=2..5",
            "direction": "positive means ask liquidity is radially farther from touch than bid liquidity",
            "bar_statistics": ["median", "net", "path", "efficiency"],
        },
        "missing_archive_dates": missing,
        "archives": records,
        "file": {
            "path": str(output),
            "sha256": file_hash,
            "rows": int(len(panel)),
            "source_complete_rows": int(panel["source_complete"].sum()),
            "first_date": str(panel["date"].min()),
            "last_date": str(panel["date"].max()),
            "columns": panel.columns.tolist(),
            "distributions": _feature_distributions(panel),
        },
        "reference_manifest": cfg.reference_manifest,
        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
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
