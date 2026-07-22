"""Build outcome-blind BTC same-millisecond cascade features."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from preprocessing.same_millisecond_cascade import BAR_COLUMNS, aggregate_same_millisecond_five_minute
from training import build_binance_aggtrade_microstructure as base


SCHEMA_VERSION = 1
ARCHIVE_MANIFEST_PATH = Path(
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json"
)
ARCHIVE_MANIFEST_SHA256 = (
    "6eec40460a6146c58994e52f1af9ace4eecc0c085887d97af5ef17c30b9f7e73"
)
SOURCE_AUDIT_PATH = Path("results/binance_aggtrade_microstructure_audit_2026-07-14.json")
SOURCE_AUDIT_SHA256 = (
    "5ac5a342d7f766ea0b6dcf9f97468ab70b9e1194775469ed0245d9208d0dc9c6"
)
SMCC_UNDERLYING_OVERLAP_COUNTS = {"2020-01-15": 1}


@dataclass(frozen=True)
class BuildConfig:
    symbol: str = "BTCUSDT"
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_um_same_millisecond_cascade_btc_2020_2023"
    workers: int = 4
    retries: int = 5
    timeout_seconds: int = 60
    overwrite: bool = False


@dataclass(frozen=True)
class SourceContract:
    archive_sha256_by_date: dict[str, str]
    archive_facts_by_date: dict[str, dict[str, int]]
    aggregate_gap_days: frozenset[str]
    underlying_overlap_counts: dict[str, int]
    verified_zero_volume_bins: frozenset[pd.Timestamp]

    @property
    def source_gap_days(self) -> frozenset[str]:
        return self.aggregate_gap_days | frozenset(self.underlying_overlap_counts)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_contract() -> SourceContract:
    if _sha256(ARCHIVE_MANIFEST_PATH) != ARCHIVE_MANIFEST_SHA256:
        raise ValueError("frozen aggTrade archive manifest hash mismatch")
    if _sha256(SOURCE_AUDIT_PATH) != SOURCE_AUDIT_SHA256:
        raise ValueError("frozen aggTrade source audit hash mismatch")
    manifest = json.loads(ARCHIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    audit = json.loads(SOURCE_AUDIT_PATH.read_text(encoding="utf-8"))
    if manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("frozen aggTrade archive manifest opened outcomes")
    if audit.get("passed") is not True:
        raise ValueError("frozen aggTrade source audit did not pass")
    archive_hashes: dict[str, str] = {}
    archive_facts: dict[str, dict[str, int]] = {}
    for month in manifest.get("months", []):
        for archive in month.get("archives", []):
            stamp = str(archive["date"])
            if stamp in archive_hashes:
                raise ValueError(f"duplicate frozen aggTrade archive date: {stamp}")
            archive_hashes[stamp] = str(archive["archive_sha256"])
            archive_facts[stamp] = {
                key: int(archive[key])
                for key in (
                    "agg_trade_rows",
                    "five_minute_rows",
                    "first_agg_trade_id",
                    "last_agg_trade_id",
                    "first_underlying_trade_id",
                    "last_underlying_trade_id",
                )
            }
    quarantine = audit.get("quarantine", {})
    gap_days = frozenset(str(value) for value in quarantine.get("source_gap_days", []))
    zero_bins = frozenset(
        cast(pd.Timestamp, pd.Timestamp(item["date"]))
        for item in quarantine.get("missing_bins", [])
        if float(item.get("volume", np.nan)) == 0.0
        and float(item.get("number_of_trades", np.nan)) == 0.0
        and item.get("source_gap_day") is False
        and item.get("documented") is True
    )
    if not archive_hashes or not gap_days or not zero_bins:
        raise ValueError("frozen aggTrade source contract is unexpectedly empty")
    if not set(SMCC_UNDERLYING_OVERLAP_COUNTS).issubset(archive_hashes):
        raise ValueError("SMCC underlying-overlap quarantine is outside the source interval")
    return SourceContract(
        archive_hashes,
        archive_facts,
        gap_days,
        dict(SMCC_UNDERLYING_OVERLAP_COUNTS),
        zero_bins,
    )


def _fetch_verified_day(
    day: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes],
    source_contract: SourceContract,
) -> tuple[bytes, str]:
    checksum_payload = fetcher(
        base.checksum_url(cfg.symbol, day), retries=cfg.retries, timeout=cfg.timeout_seconds
    )
    expected = base.expected_sha256(checksum_payload)
    payload = fetcher(
        base.archive_url(cfg.symbol, day), retries=cfg.retries, timeout=cfg.timeout_seconds
    )
    actual = base.verify_sha256(payload, expected)
    frozen = source_contract.archive_sha256_by_date.get(day.isoformat())
    if frozen is None:
        raise ValueError(f"day is absent from frozen aggTrade archive manifest: {day}")
    if actual != frozen:
        raise ValueError(f"aggTrade archive changed after source audit: {day}")
    return payload, actual


def _monthly_paths(cfg: BuildConfig, month: date) -> tuple[Path, Path]:
    monthly_dir = Path(cfg.output_dir) / "monthly"
    stem = f"{cfg.symbol}_same_millisecond_5m_{month:%Y-%m}"
    return monthly_dir / f"{stem}.csv.gz", monthly_dir / f"{stem}.json"


def _materialize_daily_grid(
    bars: pd.DataFrame,
    day: date,
    source_contract: SourceContract,
) -> pd.DataFrame:
    day_start = pd.Timestamp(day)
    grid = pd.date_range(day_start, periods=288, freq="5min")
    indexed = bars.set_index("date")
    if indexed.index.duplicated().any():
        raise ValueError(f"duplicate same-millisecond bars on {day}")
    observed = pd.Series(grid.isin(indexed.index), index=grid)
    output = indexed.reindex(grid)
    output.index.name = "date"
    gap_day = day.isoformat() in source_contract.source_gap_days
    verified_empty = pd.Series(
        [stamp in source_contract.verified_zero_volume_bins for stamp in grid],
        index=grid,
    ) & ~observed
    output["source_observed"] = observed.to_numpy(bool)
    output["source_gap_day"] = gap_day
    output["verified_zero_volume_empty"] = verified_empty.to_numpy(bool)
    output["post_gap_quarantine"] = False
    output["source_complete"] = (
        (observed | verified_empty).to_numpy(bool) & (not gap_day)
    )
    numeric_columns = [
        column
        for column in BAR_COLUMNS
        if column
        not in {
            "date",
            "source_observed",
            "source_complete",
            "source_gap_day",
            "verified_zero_volume_empty",
            "post_gap_quarantine",
        }
    ]
    output.loc[:, numeric_columns] = output.loc[:, numeric_columns].fillna(0.0)
    result = output.reset_index().loc[:, BAR_COLUMNS]
    if len(result) != 288 or result["date"].tolist() != list(grid):
        raise ValueError(f"failed to materialize full five-minute grid for {day}")
    return result


def _resume_metadata_is_current(
    metadata: dict[str, Any],
    *,
    cfg: BuildConfig,
    month: date,
    expected_days: list[date],
    output_path: Path,
    fetcher: Callable[..., bytes],
    source_contract: SourceContract,
) -> bool:
    expected_dates = [day.isoformat() for day in expected_days]
    archives = metadata.get("archives")
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("month") != f"{month:%Y-%m}"
        or metadata.get("symbol") != cfg.symbol
        or metadata.get("requested_dates") != expected_dates
        or metadata.get("output") != str(output_path)
        or metadata.get("columns") != list(BAR_COLUMNS)
        or metadata.get("rows") != 288 * len(expected_days)
        or not isinstance(archives, list)
        or [item.get("date") for item in archives] != expected_dates
    ):
        return False
    if hashlib.sha256(output_path.read_bytes()).hexdigest() != metadata.get("output_sha256"):
        raise ValueError(f"resume artifact hash mismatch: {output_path}")
    frame = pd.read_csv(output_path, compression="gzip", parse_dates=["date"])
    expected_grid = pd.date_range(
        pd.Timestamp(expected_days[0]),
        pd.Timestamp(expected_days[-1]) + pd.Timedelta("1d"),
        freq="5min",
        inclusive="left",
    )
    if (
        list(frame.columns) != list(BAR_COLUMNS)
        or len(frame) != len(expected_grid)
        or frame["date"].tolist() != list(expected_grid)
    ):
        return False
    for day, archive in zip(expected_days, archives, strict=True):
        current = base.expected_sha256(
            fetcher(
                base.checksum_url(cfg.symbol, day),
                retries=cfg.retries,
                timeout=cfg.timeout_seconds,
            )
        )
        frozen = source_contract.archive_sha256_by_date.get(day.isoformat())
        if current != frozen or current != archive.get("archive_sha256"):
            return False
        expected_facts = source_contract.archive_facts_by_date.get(day.isoformat())
        if expected_facts is None or any(
            archive.get(key) != value for key, value in expected_facts.items()
        ):
            return False
        day_start = pd.Timestamp(day)
        day_frame = frame.loc[
            (frame["date"] >= day_start)
            & (frame["date"] < day_start + pd.Timedelta("1d"))
        ]
        if int(day_frame["source_observed"].sum()) != expected_facts["five_minute_rows"]:
            return False
        expected_gap = day.isoformat() in source_contract.source_gap_days
        if bool(day_frame["source_gap_day"].all()) != expected_gap:
            return False
        expected_empty = sum(
            day_start <= stamp < day_start + pd.Timedelta("1d")
            for stamp in source_contract.verified_zero_volume_bins
        )
        if int(day_frame["verified_zero_volume_empty"].sum()) != expected_empty:
            return False
        expected_complete_flags = (
            day_frame["source_observed"].astype(bool)
            | day_frame["verified_zero_volume_empty"].astype(bool)
        ) & ~day_frame["source_gap_day"].astype(bool)
        if (
            not day_frame["source_complete"].astype(bool).equals(expected_complete_flags)
            or day_frame["post_gap_quarantine"].astype(bool).any()
        ):
            return False
        expected_complete = 0 if expected_gap else expected_facts["five_minute_rows"] + expected_empty
        if (
            archive.get("source_gap_day") != expected_gap
            or archive.get("source_observed_bars") != expected_facts["five_minute_rows"]
            or archive.get("source_complete_bars") != expected_complete
        ):
            return False
    return True


def _process_month(
    month: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = base._fetch_bytes,
    source_contract: SourceContract,
) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    expected_days = base._month_days(month, start, end)
    if not expected_days:
        raise ValueError(f"month {month:%Y-%m} has no requested days")

    output_dir = Path(cfg.output_dir)
    monthly_dir = output_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    output_path, metadata_path = _monthly_paths(cfg, month)
    if output_path.exists() and metadata_path.exists() and not cfg.overwrite:
        metadata = json.loads(metadata_path.read_text())
        if _resume_metadata_is_current(
            metadata,
            cfg=cfg,
            month=month,
            expected_days=expected_days,
            output_path=output_path,
            fetcher=fetcher,
            source_contract=source_contract,
        ):
            return metadata

    frames: list[pd.DataFrame] = []
    archives: list[dict[str, Any]] = []
    for day in expected_days:
        payload, archive_hash = _fetch_verified_day(
            day, cfg, fetcher=fetcher, source_contract=source_contract
        )
        raw = base.read_archive(payload)
        day_start = pd.Timestamp(day)
        day_end = day_start + pd.Timedelta("1d")
        raw_timestamps = pd.to_datetime(
            raw["transact_time"], unit="ms", utc=True, errors="raise"
        ).dt.tz_localize(None)
        if not ((raw_timestamps >= day_start) & (raw_timestamps < day_end)).all():
            raise ValueError(f"aggTrade archive contains timestamps outside {day}")
        aggregate_deltas = np.diff(raw["agg_trade_id"].to_numpy(np.int64))
        if np.any(aggregate_deltas <= 0):
            raise ValueError(f"aggregate-trade IDs overlap or regress on {day}")
        agg_id_gap_count = int(np.count_nonzero(aggregate_deltas > 1))
        underlying_deltas = (
            raw["first_trade_id"].to_numpy(np.int64)[1:]
            - raw["last_trade_id"].to_numpy(np.int64)[:-1]
            - 1
        )
        underlying_overlap_count = int(np.count_nonzero(underlying_deltas < 0))
        stamp = day.isoformat()
        expected_underlying_overlap_count = source_contract.underlying_overlap_counts.get(
            stamp, 0
        )
        if underlying_overlap_count != expected_underlying_overlap_count:
            raise ValueError(f"underlying-trade ID overlap contract changed on {day}")
        expected_aggregate_gap_day = stamp in source_contract.aggregate_gap_days
        if (agg_id_gap_count > 0) != expected_aggregate_gap_day:
            raise ValueError(f"aggTrade intra-day gap contract changed on {day}")
        expected_gap_day = stamp in source_contract.source_gap_days
        observed_bars = aggregate_same_millisecond_five_minute(raw)
        if not (
            (observed_bars["date"] >= day_start)
            & (observed_bars["date"] < day_end)
        ).all():
            raise ValueError(f"archive contains timestamps outside {day}")
        expected_facts = source_contract.archive_facts_by_date.get(day.isoformat())
        actual_facts = {
            "agg_trade_rows": int(len(raw)),
            "five_minute_rows": int(len(observed_bars)),
            "first_agg_trade_id": int(raw["agg_trade_id"].iloc[0]),
            "last_agg_trade_id": int(raw["agg_trade_id"].iloc[-1]),
            "first_underlying_trade_id": int(raw["first_trade_id"].iloc[0]),
            "last_underlying_trade_id": int(raw["last_trade_id"].iloc[-1]),
        }
        if actual_facts != expected_facts:
            raise ValueError(f"aggTrade archive facts changed after source audit: {day}")
        bars = _materialize_daily_grid(observed_bars, day, source_contract)
        frames.append(bars)
        archives.append(
            {
                "date": day.isoformat(),
                "archive_sha256": archive_hash,
                **actual_facts,
                "source_gap_day": expected_gap_day,
                "source_complete_bars": int(bars["source_complete"].sum()),
                "source_observed_bars": int(bars["source_observed"].sum()),
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
    source_contract: SourceContract | None = None,
) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")

    contract = source_contract if source_contract is not None else load_source_contract()
    metadata: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        future_map = {
            executor.submit(
                _process_month,
                month,
                cfg,
                fetcher=fetcher,
                source_contract=contract,
            ): month
            for month in base._month_starts(start, end)
        }
        for future in as_completed(future_map):
            month = future_map[future]
            result = future.result()
            metadata.append(result)
            print(f"completed {month:%Y-%m}: rows={result['rows']}", flush=True)
    metadata.sort(key=lambda item: item["month"])

    archives = [archive for item in metadata for archive in item["archives"]]
    expected_dates = pd.date_range(start, end - timedelta(days=1), freq="1D")
    if [item["date"] for item in archives] != [stamp.date().isoformat() for stamp in expected_dates]:
        raise ValueError("combined archive dates do not match the requested daily grid")
    for previous, current in zip(archives, archives[1:], strict=False):
        if current["first_agg_trade_id"] != previous["last_agg_trade_id"] + 1:
            raise ValueError(f"cross-day aggregate-trade ID discontinuity at {current['date']}")
        if current["first_underlying_trade_id"] != previous["last_underlying_trade_id"] + 1:
            raise ValueError(f"cross-day underlying-trade ID discontinuity at {current['date']}")

    monthly_frames: list[pd.DataFrame] = []
    for item in metadata:
        month = date.fromisoformat(f"{item['month']}-01")
        output_path, _ = _monthly_paths(cfg, month)
        if item["output"] != str(output_path):
            raise ValueError("monthly metadata output path changed after verification")
        if hashlib.sha256(output_path.read_bytes()).hexdigest() != item["output_sha256"]:
            raise ValueError("monthly output hash changed after verification")
        frame = pd.read_csv(output_path, compression="gzip", parse_dates=["date"])
        if list(frame.columns) != list(BAR_COLUMNS) or len(frame) != item["rows"]:
            raise ValueError("monthly output schema or row count changed")
        monthly_frames.append(frame)
    combined = (
        pd.concat(monthly_frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    if combined["date"].duplicated().any() or not combined["date"].is_monotonic_increasing:
        raise ValueError("combined same-millisecond output has duplicate or unordered timestamps")
    if tuple(combined.columns) != BAR_COLUMNS:
        raise ValueError("combined same-millisecond output schema changed")
    expected_grid = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="5min", inclusive="left")
    if len(combined) != len(expected_grid) or not combined["date"].reset_index(drop=True).equals(
        pd.Series(expected_grid)
    ):
        raise ValueError("combined same-millisecond output is not a complete five-minute grid")
    source_invalid = ~combined["source_complete"].astype(bool)
    post_gap = (
        source_invalid.shift(1, fill_value=False)
        .rolling(window=24, min_periods=1)
        .max()
        .astype(bool)
    )
    combined["post_gap_quarantine"] = post_gap
    combined["source_complete"] = combined["source_complete"].astype(bool) & ~post_gap

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_day = end - timedelta(days=1)
    combined_path = output_dir / f"{cfg.symbol}_same_millisecond_5m_{cfg.start}_{last_day}.csv.gz"
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
            "millisecond_group": "exact integer transaction timestamp equality within the completed 5m bar",
            "selected_group": "maximum quote notional; earliest millisecond wins exact ties",
            "pre_group_price": "last price of the strictly preceding millisecond group in the same 5m bar",
            "source_archive_manifest": str(ARCHIVE_MANIFEST_PATH),
            "source_archive_manifest_sha256": ARCHIVE_MANIFEST_SHA256,
            "source_audit": str(SOURCE_AUDIT_PATH),
            "source_audit_sha256": SOURCE_AUDIT_SHA256,
            "source_gap_days": sorted(contract.source_gap_days),
            "aggregate_gap_days": sorted(contract.aggregate_gap_days),
            "underlying_overlap_quarantine_counts": dict(
                sorted(contract.underlying_overlap_counts.items())
            ),
            "verified_zero_volume_empty_bins": len(contract.verified_zero_volume_bins),
            "full_five_minute_grid": True,
            "post_gap_quarantine_bars": 24,
            "combined_source_complete_is_post_quarantine": True,
            "monthly_resume_source_complete_is_pre_post_quarantine": True,
            "raw_archives_persisted": False,
            "outcomes_opened": False,
        },
        "combined_output": str(combined_path),
        "combined_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "source_complete_rows": int(combined["source_complete"].sum()),
        "post_gap_quarantine_rows": int(combined["post_gap_quarantine"].sum()),
        "first_date": str(combined["date"].min()),
        "last_date": str(combined["date"].max()),
        "columns": list(combined.columns),
        "months": metadata,
    }
    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
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
