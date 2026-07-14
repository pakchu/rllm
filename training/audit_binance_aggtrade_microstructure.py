"""Audit Binance aggTrade features before any alpha outcomes are opened."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.build_binance_aggtrade_microstructure import OUTPUT_COLUMNS


MARKET_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_quote",
)


@dataclass(frozen=True)
class AuditConfig:
    features: str = (
        "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
        "BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz"
    )
    manifest: str = (
        "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
        "build_manifest.json"
    )
    market: str = (
        "data/binance_um_kline_reference_btc_2020_2023/"
        "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
    )
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    output: str = "results/binance_aggtrade_microstructure_audit_2026-07-14.json"
    near_exact_relative_tolerance: float = 1e-9
    minimum_near_exact_fraction: float = 0.80
    value_p99_relative_tolerance: float = 2e-3
    value_p999_relative_tolerance: float = 1e-2
    daily_value_relative_tolerance: float = 1e-3
    monthly_value_relative_tolerance: float = 5e-5
    trade_count_p99_relative_tolerance: float = 2e-3
    trade_count_p999_relative_tolerance: float = 1e-2
    daily_trade_count_relative_tolerance: float = 6e-3
    maximum_missing_bin_fraction: float = 1e-3
    post_gap_quarantine_bars: int = 24


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_market_before(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=list(MARKET_COLUMNS), chunksize=200_000):
        timestamps = pd.to_datetime(chunk["date"], errors="raise")
        keep = timestamps.ge(start) & timestamps.lt(end)
        if keep.any():
            part = chunk.loc[keep].copy()
            part["date"] = timestamps.loc[keep]
            selected.append(part)
    if not selected:
        raise ValueError("market cache has no rows inside the audit interval")
    return pd.concat(selected, ignore_index=True).sort_values("date").reset_index(drop=True)


def _relative_error(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(actual - reference) / np.maximum(np.abs(reference), 1e-12)


def _error_summary(
    actual: pd.Series,
    reference: pd.Series,
    *,
    near_exact_tolerance: float = 1e-9,
) -> dict[str, float | int]:
    actual_values = actual.to_numpy(float)
    reference_values = reference.to_numpy(float)
    absolute = np.abs(actual_values - reference_values)
    relative = _relative_error(actual_values, reference_values)
    return {
        "rows": int(len(actual_values)),
        "exact_rows": int(np.count_nonzero(absolute == 0.0)),
        "exact_fraction": float(np.mean(absolute == 0.0)),
        "near_exact_relative_tolerance": float(near_exact_tolerance),
        "near_exact_rows": int(np.count_nonzero(relative <= near_exact_tolerance)),
        "near_exact_fraction": float(np.mean(relative <= near_exact_tolerance)),
        "mean_absolute_error": float(absolute.mean()),
        "max_absolute_error": float(absolute.max()),
        "p95_relative_error": float(np.quantile(relative, 0.95)),
        "p99_relative_error": float(np.quantile(relative, 0.99)),
        "p999_relative_error": float(np.quantile(relative, 0.999)),
        "max_relative_error": float(relative.max()),
        "total_actual": float(actual_values.sum()),
        "total_reference": float(reference_values.sum()),
        "total_relative_error": float(
            abs(actual_values.sum() - reference_values.sum())
            / max(abs(reference_values.sum()), 1e-12)
        ),
    }


def _feature_checks(frame: pd.DataFrame) -> tuple[dict[str, bool], dict[str, Any]]:
    numeric = frame.drop(columns="date").to_numpy(float)
    counts = frame[["agg_trade_count", "underlying_trade_count"]].to_numpy(float)
    first_time = pd.to_datetime(frame["first_transact_time_ms"], unit="ms", errors="raise")
    last_time = pd.to_datetime(frame["last_transact_time_ms"], unit="ms", errors="raise")
    end_time = frame["date"] + pd.Timedelta("5min")
    quote = frame["quote_notional"].to_numpy(float)
    buy = frame["buy_quote_notional"].to_numpy(float)
    sell = frame["sell_quote_notional"].to_numpy(float)
    signed = frame["signed_quote_notional"].to_numpy(float)

    checks = {
        "columns_exact": tuple(frame.columns) == OUTPUT_COLUMNS,
        "timestamps_unique": not frame["date"].duplicated().any(),
        "timestamps_monotonic": frame["date"].is_monotonic_increasing,
        "all_numeric_finite": bool(np.isfinite(numeric).all()),
        "counts_positive_integer": bool(
            (counts > 0.0).all() and np.allclose(counts, np.round(counts))
        ),
        "volumes_positive": bool(
            (frame[["base_volume", "quote_notional"]].to_numpy(float) > 0.0).all()
        ),
        "buy_sell_partition": bool(np.allclose(buy + sell, quote, rtol=1e-9, atol=1e-6)),
        "signed_partition": bool(np.allclose(buy - sell, signed, rtol=1e-8, atol=1e-2)),
        "flow_coherence_identity": bool(
            np.allclose(
                frame["flow_coherence"].to_numpy(float),
                np.abs(signed) / quote,
                rtol=1e-9,
                atol=1e-10,
            )
        ),
        "underlying_per_event_identity": bool(
            np.allclose(
                frame["underlying_trades_per_agg_event"].to_numpy(float),
                frame["underlying_trade_count"].to_numpy(float)
                / frame["agg_trade_count"].to_numpy(float),
                rtol=1e-9,
                atol=1e-10,
            )
        ),
        "micro_return_identity": bool(
            np.allclose(
                frame["micro_log_return"].to_numpy(float),
                np.log(
                    frame["last_price"].to_numpy(float)
                    / frame["first_price"].to_numpy(float)
                ),
                rtol=1e-9,
                atol=1e-11,
            )
        ),
        "signed_response_identity": bool(
            np.allclose(
                frame["signed_price_response"].to_numpy(float),
                np.sign(signed) * frame["micro_log_return"].to_numpy(float),
                rtol=1e-9,
                atol=1e-11,
            )
        ),
        "transaction_times_inside_bins": bool(
            (first_time >= frame["date"]).all()
            and (first_time <= last_time).all()
            and (last_time < end_time).all()
        ),
        "event_quantiles_ordered": bool(
            (
                frame["event_notional_p50"]
                <= frame["event_notional_p90"] + 1e-9
            ).all()
            and (
                frame["event_notional_p90"]
                <= frame["event_notional_p99"] + 1e-9
            ).all()
            and (
                frame["event_notional_p99"]
                <= frame["event_notional_max"] + 1e-9
            ).all()
        ),
        "bounded_features": bool(
            frame["flow_coherence"].between(0.0, 1.0 + 1e-10).all()
            and frame["event_notional_hhi"].between(1e-300, 1.0 + 1e-10).all()
            and frame["normalized_effective_event_count"].between(1e-300, 1.0 + 1e-10).all()
            and frame["signed_event_imbalance"].between(-1.0 - 1e-10, 1.0 + 1e-10).all()
            and frame["sign_flip_rate"].between(0.0, 1.0 + 1e-10).all()
            and frame["max_same_sign_run_share"].between(0.0, 1.0 + 1e-10).all()
            and frame["interarrival_burstiness"].between(-1.0 - 1e-10, 1.0 + 1e-10).all()
            and (frame["mean_same_sign_run_length"] >= 1.0 - 1e-10).all()
            and (
                frame["mean_same_sign_run_length"]
                <= frame["agg_trade_count"] + 1e-10
            ).all()
        ),
    }
    diagnostics = {
        "rows": int(len(frame)),
        "first_date": str(frame["date"].min()),
        "last_date": str(frame["date"].max()),
        "feature_minima": {
            column: float(frame[column].min())
            for column in (
                "agg_trade_count",
                "underlying_trade_count",
                "base_volume",
                "quote_notional",
            )
        },
        "feature_maxima": {
            column: float(frame[column].max())
            for column in (
                "agg_trade_count",
                "underlying_trade_count",
                "flow_coherence",
                "event_notional_hhi",
                "normalized_effective_event_count",
            )
        },
    }
    return checks, diagnostics


def _manifest_checks(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    features_path: Path,
    expected_days: pd.DatetimeIndex,
    frame: pd.DataFrame,
) -> tuple[dict[str, bool], dict[str, Any]]:
    archives = [archive for month in manifest.get("months", []) for archive in month.get("archives", [])]
    archive_dates = [archive.get("date") for archive in archives]
    expected_dates = [timestamp.date().isoformat() for timestamp in expected_days]
    aggregate_deltas = [
        int(current["first_agg_trade_id"]) - int(previous["last_agg_trade_id"]) - 1
        for previous, current in zip(archives, archives[1:])
    ]
    underlying_deltas = [
        int(current["first_underlying_trade_id"])
        - int(previous["last_underlying_trade_id"])
        - 1
        for previous, current in zip(archives, archives[1:])
    ]
    aggregate_internal_gaps = [
        int(archive["last_agg_trade_id"])
        - int(archive["first_agg_trade_id"])
        + 1
        - int(archive["agg_trade_rows"])
        for archive in archives
    ]
    source_gap_days = {
        archive["date"]
        for archive, gap in zip(archives, aggregate_internal_gaps, strict=True)
        if gap > 0
    }
    for position, delta in enumerate(aggregate_deltas, start=1):
        if delta > 0:
            source_gap_days.add(archives[position - 1]["date"])
            source_gap_days.add(archives[position]["date"])
    day_counts = (
        frame.assign(source_day=frame["date"].dt.date.astype(str))
        .groupby("source_day", sort=True, observed=True)["agg_trade_count"]
        .sum()
    )
    metadata_counts = pd.Series(
        {archive["date"]: int(archive["agg_trade_rows"]) for archive in archives},
        dtype="int64",
    ).sort_index()

    monthly_hashes_valid = True
    for month in manifest.get("months", []):
        path = Path(month.get("output", ""))
        monthly_hashes_valid &= path.is_file() and _sha256(path) == month.get("output_sha256")

    config = manifest.get("config", {})
    requested_dates_valid = all(
        month.get("schema_version") == 1
        and month.get("requested_dates")
        == [archive.get("date") for archive in month.get("archives", [])]
        for month in manifest.get("months", [])
    )
    archive_hashes_well_formed = all(
        isinstance(archive.get("archive_sha256"), str)
        and len(archive["archive_sha256"]) == 64
        and all(character in "0123456789abcdef" for character in archive["archive_sha256"].lower())
        for archive in archives
    )
    checks = {
        "manifest_config_interval_exact": (
            config.get("symbol") == "BTCUSDT"
            and config.get("start") == expected_dates[0]
            and config.get("end")
            == (expected_days[-1] + pd.Timedelta("1d")).date().isoformat()
        ),
        "manifest_outcomes_unopened": manifest.get("protocol", {}).get("outcomes_opened") is False,
        "manifest_combined_contract_exact": (
            manifest.get("combined_output") == str(features_path)
            and manifest.get("rows") == len(frame)
            and manifest.get("columns") == list(OUTPUT_COLUMNS)
        ),
        "combined_hash_valid": _sha256(features_path) == manifest.get("combined_sha256"),
        "monthly_hashes_valid": bool(monthly_hashes_valid),
        "monthly_requested_dates_exact": requested_dates_valid,
        "archive_hashes_well_formed": archive_hashes_well_formed,
        "archive_day_coverage_exact": archive_dates == expected_dates,
        "aggregate_ids_cross_day_nonoverlapping": all(delta >= 0 for delta in aggregate_deltas),
        "aggregate_ids_inside_days_nonoverlapping": all(gap >= 0 for gap in aggregate_internal_gaps),
        "underlying_ids_cross_day_nonoverlapping": all(delta >= 0 for delta in underlying_deltas),
        "daily_aggregate_row_counts_exact": day_counts.equals(metadata_counts),
        "manifest_is_sibling_of_features": manifest_path.parent == features_path.parent,
    }
    diagnostics = {
        "months": int(len(manifest.get("months", []))),
        "archives": int(len(archives)),
        "aggregate_id_gap_count": int(sum(delta > 0 for delta in aggregate_deltas)),
        "aggregate_id_overlap_count": int(sum(delta < 0 for delta in aggregate_deltas)),
        "aggregate_id_max_gap": int(max(aggregate_deltas, default=0)),
        "aggregate_id_internal_gap_days": int(sum(gap != 0 for gap in aggregate_internal_gaps)),
        "source_gap_days": sorted(source_gap_days),
        "underlying_id_gap_count": int(sum(delta > 0 for delta in underlying_deltas)),
        "underlying_id_overlap_count": int(sum(delta < 0 for delta in underlying_deltas)),
        "underlying_id_max_gap": int(max(underlying_deltas, default=0)),
    }
    return checks, diagnostics


def run_audit(cfg: AuditConfig) -> dict[str, Any]:
    features_path = Path(cfg.features)
    manifest_path = Path(cfg.manifest)
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")

    frame = pd.read_csv(features_path, compression="gzip", parse_dates=["date"])
    manifest = json.loads(manifest_path.read_text())
    market = _read_market_before(Path(cfg.market), start, end)
    expected_index = pd.date_range(start, end, inclusive="left", freq="5min")
    expected_days = pd.date_range(start, end, inclusive="left", freq="1D")

    feature_checks, feature_diagnostics = _feature_checks(frame)
    manifest_checks, manifest_diagnostics = _manifest_checks(
        manifest,
        manifest_path=manifest_path,
        features_path=features_path,
        expected_days=expected_days,
        frame=frame,
    )
    source_gap_days = set(manifest_diagnostics["source_gap_days"])
    missing_index = expected_index.difference(frame["date"])
    extra_index = pd.DatetimeIndex(frame["date"]).difference(expected_index)
    market_by_date = market.set_index("date")
    missing_rows = market_by_date.reindex(missing_index).rename_axis("date").reset_index()
    missing_rows["source_gap_day"] = missing_rows["date"].dt.strftime("%Y-%m-%d").isin(
        source_gap_days
    )
    missing_rows["documented"] = missing_rows["volume"].eq(0.0) | missing_rows[
        "source_gap_day"
    ]
    coverage_checks = {
        "feature_index_is_expected_subset": len(extra_index) == 0,
        "feature_missing_fraction_bounded": len(missing_index) / len(expected_index)
        <= cfg.maximum_missing_bin_fraction,
        "feature_missing_bins_documented": bool(missing_rows["documented"].all()),
        "market_rows_exact": len(market) == len(expected_index),
        "market_index_exact": market["date"].equals(pd.Series(expected_index, name="date")),
    }
    joined = frame.merge(market, on="date", how="inner", validate="one_to_one")
    joined["day"] = joined["date"].dt.floor("1D")
    clean_joined = joined[
        ~joined["date"].dt.strftime("%Y-%m-%d").isin(source_gap_days)
    ].copy()
    mappings = {
        "base_volume": "volume",
        "quote_notional": "quote_asset_volume",
        "underlying_trade_count": "number_of_trades",
        "buy_quote_notional": "taker_buy_quote",
        "first_price": "open",
        "last_price": "close",
    }
    five_minute = {
        source: _error_summary(
            clean_joined[source],
            clean_joined[target],
            near_exact_tolerance=cfg.near_exact_relative_tolerance,
        )
        for source, target in mappings.items()
    }

    daily_pairs: dict[str, tuple[pd.Series, pd.Series]] = {}
    clean_joined["day"] = clean_joined["date"].dt.floor("1D")
    for source, target in mappings.items():
        if source in {"first_price", "last_price"}:
            continue
        grouped = clean_joined.groupby("day", sort=True, observed=True)
        daily_pairs[source] = (grouped[source].sum(), grouped[target].sum())
    daily = {
        source: _error_summary(
            actual,
            reference,
            near_exact_tolerance=cfg.near_exact_relative_tolerance,
        )
        for source, (actual, reference) in daily_pairs.items()
    }
    clean_joined["month"] = clean_joined["date"].dt.to_period("M")
    monthly: dict[str, dict[str, float | int]] = {}
    for source, target in mappings.items():
        if source in {"first_price", "last_price"}:
            continue
        grouped = clean_joined.groupby("month", sort=True, observed=True)
        monthly[source] = _error_summary(
            grouped[source].sum(),
            grouped[target].sum(),
            near_exact_tolerance=cfg.near_exact_relative_tolerance,
        )

    value_sources = ("base_volume", "quote_notional", "buy_quote_notional")
    price_inside_envelope = (
        clean_joined["first_price"].ge(clean_joined["low"] - 1e-9)
        & clean_joined["first_price"].le(clean_joined["high"] + 1e-9)
        & clean_joined["last_price"].ge(clean_joined["low"] - 1e-9)
        & clean_joined["last_price"].le(clean_joined["high"] + 1e-9)
    )
    shifted_quote_errors: dict[str, float] = {}
    for hours in (-9, 9):
        shifted = market[["date", "quote_asset_volume"]].copy()
        shifted["date"] = shifted["date"] + pd.Timedelta(hours=hours)
        shifted_join = clean_joined[["date", "quote_notional"]].merge(
            shifted,
            on="date",
            how="inner",
        )
        shifted_quote_errors[f"{hours:+d}h"] = _error_summary(
            shifted_join["quote_notional"],
            shifted_join["quote_asset_volume"],
        )["p99_relative_error"]

    reconciliation_checks = {
        "all_feature_rows_joined": len(joined) == len(frame),
        "direct_join_beats_timezone_shifts": all(
            five_minute["quote_notional"]["p99_relative_error"] < error
            for error in shifted_quote_errors.values()
        ),
        "value_near_exact_fraction": all(
            five_minute[source]["near_exact_fraction"] >= cfg.minimum_near_exact_fraction
            for source in value_sources
        ),
        "value_p99_within_boundary_tolerance": all(
            five_minute[source]["p99_relative_error"] <= cfg.value_p99_relative_tolerance
            for source in value_sources
        ),
        "value_p999_within_boundary_tolerance": all(
            five_minute[source]["p999_relative_error"] <= cfg.value_p999_relative_tolerance
            for source in value_sources
        ),
        "daily_base_volume_conserved": daily["base_volume"]["max_relative_error"]
        <= cfg.daily_value_relative_tolerance,
        "daily_quote_notional_conserved": daily["quote_notional"]["max_relative_error"]
        <= cfg.daily_value_relative_tolerance,
        "daily_buy_quote_conserved": daily["buy_quote_notional"]["max_relative_error"]
        <= cfg.daily_value_relative_tolerance,
        "monthly_values_conserved": all(
            monthly[source]["max_relative_error"] <= cfg.monthly_value_relative_tolerance
            for source in value_sources
        ),
        "trade_count_near_exact_fraction": five_minute["underlying_trade_count"][
            "near_exact_fraction"
        ]
        >= cfg.minimum_near_exact_fraction,
        "trade_count_p99_within_tolerance": five_minute["underlying_trade_count"][
            "p99_relative_error"
        ]
        <= cfg.trade_count_p99_relative_tolerance,
        "trade_count_p999_within_tolerance": five_minute["underlying_trade_count"][
            "p999_relative_error"
        ]
        <= cfg.trade_count_p999_relative_tolerance,
        "daily_trade_count_close": daily["underlying_trade_count"]["max_relative_error"]
        <= cfg.daily_trade_count_relative_tolerance,
        "first_price_matches_open": five_minute["first_price"]["exact_fraction"] >= 0.93,
        "last_price_matches_close": five_minute["last_price"]["exact_fraction"] >= 0.9999,
        "trade_prices_inside_kline_envelope": bool(price_inside_envelope.all()),
    }

    all_checks = {
        **{f"feature.{key}": value for key, value in feature_checks.items()},
        **{f"coverage.{key}": value for key, value in coverage_checks.items()},
        **{f"manifest.{key}": value for key, value in manifest_checks.items()},
        **{f"reconciliation.{key}": value for key, value in reconciliation_checks.items()},
    }
    result = {
        "protocol": {
            "outcomes_opened": False,
            "interval": {"start": cfg.start, "end_exclusive": cfg.end},
            "five_minute_discrepancy_policy": (
                "diagnostic only: aggTrade events can straddle kline boundaries; "
                "daily value conservation is the hard integrity criterion"
            ),
            "daily_value_relative_tolerance": cfg.daily_value_relative_tolerance,
            "monthly_value_relative_tolerance": cfg.monthly_value_relative_tolerance,
            "daily_trade_count_relative_tolerance": cfg.daily_trade_count_relative_tolerance,
            "source_gap_policy": (
                "quarantine each UTC source-ID-gap day and the next configured bars; "
                "zero-volume missing bins reset sequential features"
            ),
            "post_gap_quarantine_bars": cfg.post_gap_quarantine_bars,
        },
        "config": asdict(cfg),
        "passed": all(all_checks.values()),
        "failed_checks": [key for key, value in all_checks.items() if not value],
        "checks": all_checks,
        "feature_diagnostics": feature_diagnostics,
        "manifest_diagnostics": manifest_diagnostics,
        "quarantine": {
            "source_gap_days": sorted(source_gap_days),
            "post_gap_bars": cfg.post_gap_quarantine_bars,
            "missing_bins": [
                {
                    "date": str(row.date),
                    "volume": float(row.volume),
                    "number_of_trades": float(row.number_of_trades),
                    "source_gap_day": bool(row.source_gap_day),
                    "documented": bool(row.documented),
                }
                for row in missing_rows.itertuples(index=False)
            ],
            "missing_bin_count": int(len(missing_rows)),
            "zero_volume_missing_bin_count": int(missing_rows["volume"].eq(0.0).sum()),
            "source_gap_missing_bin_count": int(missing_rows["source_gap_day"].sum()),
        },
        "reconciliation": {
            "five_minute": five_minute,
            "daily": daily,
            "monthly": monthly,
            "shifted_quote_p99_relative_error": shifted_quote_errors,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=AuditConfig.features)
    parser.add_argument("--manifest", default=AuditConfig.manifest)
    parser.add_argument("--market", default=AuditConfig.market)
    parser.add_argument("--start", default=AuditConfig.start)
    parser.add_argument("--end", default=AuditConfig.end)
    parser.add_argument("--output", default=AuditConfig.output)
    parser.add_argument(
        "--daily-value-relative-tolerance",
        type=float,
        default=AuditConfig.daily_value_relative_tolerance,
    )
    parser.add_argument(
        "--daily-trade-count-relative-tolerance",
        type=float,
        default=AuditConfig.daily_trade_count_relative_tolerance,
    )
    cfg = AuditConfig(**vars(parser.parse_args()))
    result = run_audit(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "failed_checks": result["failed_checks"],
                "rows": result["feature_diagnostics"]["rows"],
                "output": str(output),
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
