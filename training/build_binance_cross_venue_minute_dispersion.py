"""Build causal Spot/USD-M within-five-minute dispersion descriptors.

The source is the official Binance one-minute kline archive.  Descriptors are
computed across the five completed one-minute rows inside each five-minute
bucket.  They describe *minute-level* concentration and must not be interpreted
as trade-level ticket distributions.  Raw ZIP archives are checksum-verified
and discarded after each month is processed.
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

from training.build_binance_aggtrade_microstructure import (
    _fetch_bytes,
    _month_starts,
    _write_gzip_csv,
    expected_sha256,
    verify_sha256,
)
from training.build_binance_cross_venue_minute_leadership import (
    _checksum_url,
    _indexed_source,
    _normalize_expected_minutes,
    read_archive,
    spot_archive_url,
    um_archive_url,
)


SCHEMA_VERSION = 1
SEALED_END_EXCLUSIVE = date(2024, 1, 1)

AUDIT_COLUMNS = (
    "spot_rows",
    "um_rows",
    "spot_missing_minutes",
    "um_missing_minutes",
    "spot_invalid_source_minutes",
    "um_invalid_source_minutes",
)

VENUE_FEATURE_SUFFIXES = (
    "quote_time_hhi",
    "trade_time_hhi",
    "abs_flow_time_hhi",
    "quote_minus_trade_time_hhi",
    "mean_ticket_quote",
    "ticket_log_std",
    "ticket_log_range",
    "ticket_time_centroid",
    "net_flow_fraction",
    "flow_sign_persistence",
    "flow_sign_switch_rate",
    "signed_impact_bp",
    "impact_per_abs_flow_fraction_bp",
)

CROSS_FEATURE_COLUMNS = (
    "spot_minus_um_quote_time_hhi",
    "spot_minus_um_trade_time_hhi",
    "spot_minus_um_abs_flow_time_hhi",
    "spot_minus_um_quote_minus_trade_time_hhi",
    "log_spot_um_mean_ticket_ratio",
    "spot_minus_um_ticket_log_std",
    "spot_minus_um_ticket_time_centroid",
    "spot_minus_um_flow_sign_persistence",
    "spot_minus_um_flow_sign_switch_rate",
    "spot_minus_um_signed_impact_bp",
    "spot_minus_um_impact_per_abs_flow_fraction_bp",
    "net_flow_sign_agreement",
)

FEATURE_COLUMNS = tuple(
    f"{venue}_{suffix}"
    for venue in ("spot", "um")
    for suffix in VENUE_FEATURE_SUFFIXES
) + CROSS_FEATURE_COLUMNS

OUTPUT_COLUMNS = (
    "date",
    "feature_available_time_utc",
    "trade_earliest_time_utc",
    *AUDIT_COLUMNS,
    *FEATURE_COLUMNS,
    "source_complete",
    "minute_dispersion_feature_valid",
    "feature_invalid_reason",
)


@dataclass(frozen=True)
class BuildConfig:
    symbol: str = "BTCUSDT"
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_cross_venue_minute_dispersion_btc"
    workers: int = 4
    retries: int = 5
    timeout_seconds: int = 60
    overwrite: bool = False
    open_oos: bool = False


def _group_sum(values: pd.Series, groups: pd.Series) -> pd.Series:
    return values.groupby(groups, sort=True, observed=True).sum(min_count=1)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).divide(denominator.astype(float).replace(0.0, np.nan))


def _hhi(values: pd.Series, groups: pd.Series) -> pd.Series:
    total = _group_sum(values, groups)
    return _ratio(_group_sum(values.pow(2), groups), total.pow(2))


def _weighted_centroid(
    values: pd.Series,
    positions: pd.Series,
    groups: pd.Series,
) -> pd.Series:
    return _ratio(_group_sum(values * positions, groups), _group_sum(values, groups)).divide(4.0)


def _append_invalid_reason(
    reasons: pd.Series,
    mask: pd.Series,
    label: str,
) -> pd.Series:
    prefix = reasons.where(reasons.eq("ok"), reasons + "|").where(reasons.ne("ok"), "")
    return reasons.where(~mask, prefix + label)


def aggregate_cross_venue_minute_dispersion(
    spot: pd.DataFrame,
    um: pd.DataFrame,
    *,
    expected_minutes: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Aggregate five completed one-minute observations into causal descriptors."""
    expected = _normalize_expected_minutes(spot, um, expected_minutes)
    work = pd.concat(
        [
            _indexed_source(spot, expected, "spot"),
            _indexed_source(um, expected, "um"),
        ],
        axis=1,
    )
    work.index.name = "minute_open_time"
    work["date"] = work.index.floor("5min")
    work["minute_position"] = (
        (work.index.to_series(index=work.index) - work["date"])
        .dt.total_seconds()
        .astype(int)
        .floordiv(60)
        .astype(float)
    )
    groups = work["date"]
    grouped = work.groupby("date", sort=True, observed=True)
    output = pd.DataFrame(index=pd.DatetimeIndex(sorted(groups.unique()), name="date"))

    for venue in ("spot", "um"):
        present = work[f"{venue}_present"].eq(True)
        valid = work[f"{venue}_source_row_valid"].eq(True)
        quote = work[f"{venue}_quote_notional"].astype(float)
        trades = work[f"{venue}_trade_count"].astype(float)
        signed_quote = 2.0 * work[f"{venue}_taker_buy_quote"].astype(float) - quote
        abs_flow = signed_quote.abs()
        ticket = _ratio(quote, trades)
        log_ticket = np.log(ticket.where(ticket.gt(0.0)))

        output[f"{venue}_rows"] = present.groupby(groups, observed=True).sum().astype(int)
        output[f"{venue}_missing_minutes"] = 5 - output[f"{venue}_rows"]
        output[f"{venue}_invalid_source_minutes"] = (
            (present & ~valid).groupby(groups, observed=True).sum().astype(int)
        )
        output[f"{venue}_quote_time_hhi"] = _hhi(quote, groups)
        output[f"{venue}_trade_time_hhi"] = _hhi(trades, groups)
        output[f"{venue}_abs_flow_time_hhi"] = _hhi(abs_flow, groups)
        output[f"{venue}_quote_minus_trade_time_hhi"] = (
            output[f"{venue}_quote_time_hhi"] - output[f"{venue}_trade_time_hhi"]
        )
        output[f"{venue}_mean_ticket_quote"] = _ratio(
            _group_sum(quote, groups), _group_sum(trades, groups)
        )
        output[f"{venue}_ticket_log_std"] = log_ticket.groupby(
            groups, sort=True, observed=True
        ).std(ddof=0)
        output[f"{venue}_ticket_log_range"] = (
            log_ticket.groupby(groups, sort=True, observed=True).max()
            - log_ticket.groupby(groups, sort=True, observed=True).min()
        )
        output[f"{venue}_ticket_time_centroid"] = _weighted_centroid(
            ticket, work["minute_position"], groups
        )
        output[f"{venue}_net_flow_fraction"] = _ratio(
            _group_sum(signed_quote, groups), _group_sum(quote, groups)
        )
        output[f"{venue}_flow_sign_persistence"] = _ratio(
            _group_sum(signed_quote, groups).abs(), _group_sum(abs_flow, groups)
        )

        sign = np.sign(signed_quote)
        previous_sign = sign.groupby(groups, sort=False, observed=True).shift()
        transition = work["minute_position"].gt(0.0) & sign.ne(0.0) & previous_sign.ne(0.0)
        flip = transition & sign.ne(previous_sign)
        output[f"{venue}_flow_sign_switch_rate"] = _ratio(
            flip.groupby(groups, observed=True).sum(),
            transition.groupby(groups, observed=True).sum(),
        )

        first_open = grouped[f"{venue}_open"].first()
        last_close = grouped[f"{venue}_close"].last()
        signed_impact = (
            np.sign(_group_sum(signed_quote, groups))
            * np.log(last_close / first_open)
            * 10_000.0
        )
        output[f"{venue}_signed_impact_bp"] = signed_impact
        output[f"{venue}_impact_per_abs_flow_fraction_bp"] = signed_impact.divide(
            output[f"{venue}_net_flow_fraction"].abs().clip(lower=0.01)
        )

    output["spot_minus_um_quote_time_hhi"] = (
        output["spot_quote_time_hhi"] - output["um_quote_time_hhi"]
    )
    output["spot_minus_um_trade_time_hhi"] = (
        output["spot_trade_time_hhi"] - output["um_trade_time_hhi"]
    )
    output["spot_minus_um_abs_flow_time_hhi"] = (
        output["spot_abs_flow_time_hhi"] - output["um_abs_flow_time_hhi"]
    )
    output["spot_minus_um_quote_minus_trade_time_hhi"] = (
        output["spot_quote_minus_trade_time_hhi"]
        - output["um_quote_minus_trade_time_hhi"]
    )
    output["log_spot_um_mean_ticket_ratio"] = np.log(
        _ratio(output["spot_mean_ticket_quote"], output["um_mean_ticket_quote"])
    )
    output["spot_minus_um_ticket_log_std"] = (
        output["spot_ticket_log_std"] - output["um_ticket_log_std"]
    )
    output["spot_minus_um_ticket_time_centroid"] = (
        output["spot_ticket_time_centroid"] - output["um_ticket_time_centroid"]
    )
    output["spot_minus_um_flow_sign_persistence"] = (
        output["spot_flow_sign_persistence"] - output["um_flow_sign_persistence"]
    )
    output["spot_minus_um_flow_sign_switch_rate"] = (
        output["spot_flow_sign_switch_rate"] - output["um_flow_sign_switch_rate"]
    )
    output["spot_minus_um_signed_impact_bp"] = (
        output["spot_signed_impact_bp"] - output["um_signed_impact_bp"]
    )
    output["spot_minus_um_impact_per_abs_flow_fraction_bp"] = (
        output["spot_impact_per_abs_flow_fraction_bp"]
        - output["um_impact_per_abs_flow_fraction_bp"]
    )
    output["net_flow_sign_agreement"] = (
        np.sign(output["spot_net_flow_fraction"])
        * np.sign(output["um_net_flow_fraction"])
    )

    output["source_complete"] = (
        output["spot_rows"].eq(5)
        & output["um_rows"].eq(5)
        & output["spot_missing_minutes"].eq(0)
        & output["um_missing_minutes"].eq(0)
        & output["spot_invalid_source_minutes"].eq(0)
        & output["um_invalid_source_minutes"].eq(0)
    )
    finite_features = pd.Series(
        np.isfinite(output.loc[:, FEATURE_COLUMNS].to_numpy(float)).all(axis=1),
        index=output.index,
    )
    output["minute_dispersion_feature_valid"] = output["source_complete"] & finite_features
    reasons = pd.Series("ok", index=output.index, dtype="object")
    reasons = _append_invalid_reason(reasons, ~output["source_complete"], "source_incomplete")
    reasons = _append_invalid_reason(reasons, ~finite_features, "nonfinite_descriptor")
    output["feature_invalid_reason"] = reasons
    invalid = ~output["minute_dispersion_feature_valid"]
    output.loc[invalid, FEATURE_COLUMNS] = np.nan
    output["feature_available_time_utc"] = output.index + pd.Timedelta("5min")
    output["trade_earliest_time_utc"] = output["feature_available_time_utc"]
    output = output.reset_index().loc[:, OUTPUT_COLUMNS]
    if output["date"].duplicated().any() or not output["date"].is_monotonic_increasing:
        raise ValueError("minute-dispersion timestamps are duplicate or unordered")
    return output


def _resume_metadata_is_current(
    metadata: dict[str, Any],
    *,
    cfg: BuildConfig,
    month: date,
    output_path: Path,
    fetcher: Callable[..., bytes],
) -> bool:
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("month") != f"{month:%Y-%m}"
        or metadata.get("symbol") != cfg.symbol
    ):
        return False
    if hashlib.sha256(output_path.read_bytes()).hexdigest() != metadata.get("output_sha256"):
        raise ValueError(f"resume artifact hash mismatch: {output_path}")
    hashes = {}
    for venue, archive in (
        ("spot", spot_archive_url(cfg.symbol, month)),
        ("um", um_archive_url(cfg.symbol, month)),
    ):
        hashes[venue] = expected_sha256(
            fetcher(
                _checksum_url(archive),
                retries=cfg.retries,
                timeout=cfg.timeout_seconds,
            )
        )
    return (
        hashes["spot"] == metadata.get("spot_archive_sha256")
        and hashes["um"] == metadata.get("um_archive_sha256")
    )


def _process_month(
    month: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    monthly_dir = Path(cfg.output_dir) / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.symbol}_cross_venue_minute_dispersion_5m_{month:%Y-%m}"
    output_path = monthly_dir / f"{stem}.csv.gz"
    metadata_path = monthly_dir / f"{stem}.json"
    if output_path.exists() and metadata_path.exists() and not cfg.overwrite:
        metadata = json.loads(metadata_path.read_text())
        if _resume_metadata_is_current(
            metadata,
            cfg=cfg,
            month=month,
            output_path=output_path,
            fetcher=fetcher,
        ):
            return metadata

    payloads: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for venue, archive in (
        ("spot", spot_archive_url(cfg.symbol, month)),
        ("um", um_archive_url(cfg.symbol, month)),
    ):
        checksum = expected_sha256(
            fetcher(
                _checksum_url(archive),
                retries=cfg.retries,
                timeout=cfg.timeout_seconds,
            )
        )
        payload = fetcher(archive, retries=cfg.retries, timeout=cfg.timeout_seconds)
        hashes[venue] = verify_sha256(payload, checksum)
        payloads[venue] = payload

    spot = read_archive(payloads.pop("spot"), venue="spot")
    um = read_archive(payloads.pop("um"), venue="um")
    month_start = pd.Timestamp(month)
    next_month = month_start + pd.offsets.MonthBegin(1)
    expected_minutes = pd.date_range(month_start, next_month, inclusive="left", freq="1min")
    output = aggregate_cross_venue_minute_dispersion(
        spot, um, expected_minutes=expected_minutes
    )
    expected_bars = pd.date_range(month_start, next_month, inclusive="left", freq="5min")
    if not output["date"].equals(pd.Series(expected_bars, name="date")):
        raise ValueError(f"minute-dispersion month {month:%Y-%m} has an invalid grid")

    _write_gzip_csv(output, output_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "month": f"{month:%Y-%m}",
        "symbol": cfg.symbol,
        "spot_archive_sha256": hashes["spot"],
        "um_archive_sha256": hashes["um"],
        "spot_raw_rows": int(len(spot)),
        "um_raw_rows": int(len(um)),
        "rows": int(len(output)),
        "source_complete_rows": int(output["source_complete"].sum()),
        "feature_valid_rows": int(output["minute_dispersion_feature_valid"].sum()),
        "first_date": str(output["date"].min()),
        "last_date": str(output["date"].max()),
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def build(cfg: BuildConfig) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start.day != 1 or end.day != 1:
        raise ValueError("build boundaries must be month starts")
    if end > SEALED_END_EXCLUSIVE and not cfg.open_oos:
        raise ValueError("2024+ source is sealed; pass --open-oos only after candidate freeze")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")

    months = _month_starts(start, end)
    metadata: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {executor.submit(_process_month, month, cfg): month for month in months}
        for future in as_completed(futures):
            month = futures[future]
            item = future.result()
            metadata.append(item)
            print(
                f"completed {month:%Y-%m}: rows={item['rows']} valid={item['feature_valid_rows']}",
                flush=True,
            )
    metadata.sort(key=lambda item: item["month"])
    frames = [
        pd.read_csv(
            item["output"],
            compression="gzip",
            parse_dates=["date", "feature_available_time_utc", "trade_earliest_time_utc"],
        )
        for item in metadata
    ]
    combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    expected = pd.date_range(start, end, inclusive="left", freq="5min")
    if not combined["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("combined minute-dispersion output has an invalid grid")
    if not combined["feature_available_time_utc"].equals(
        combined["date"] + pd.Timedelta("5min")
    ):
        raise ValueError("feature availability timestamp is inconsistent")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_month = date(end.year - (end.month == 1), 12 if end.month == 1 else end.month - 1, 1)
    combined_path = output_dir / (
        f"{cfg.symbol}_cross_venue_minute_dispersion_5m_"
        f"{start:%Y-%m}_{last_month:%Y-%m}.csv.gz"
    )
    _write_gzip_csv(combined, combined_path)
    valid = combined["minute_dispersion_feature_valid"].astype(bool)
    manifest = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "sources": [
                "official Binance Spot monthly one-minute kline archives",
                "official Binance USD-M monthly one-minute kline archives",
            ],
            "archive_checksums_verified": True,
            "feature_granularity": "dispersion across five one-minute aggregates, not trades",
            "end_is_exclusive": True,
            "sealed_end_exclusive": SEALED_END_EXCLUSIVE.isoformat(),
            "post2023_opened": bool(cfg.open_oos and end > SEALED_END_EXCLUSIVE),
            "join_key": "exact UTC one-minute open_time",
            "feature_available_time": "five-minute bar open time plus five minutes",
            "raw_archives_persisted": False,
            "outcomes_opened": False,
        },
        "combined_output": str(combined_path),
        "combined_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "feature_valid_rows": int(valid.sum()),
        "quarantined_rows": int((~valid).sum()),
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
    parser.add_argument("--open-oos", action="store_true")
    manifest = build(BuildConfig(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "combined_output",
                    "rows",
                    "feature_valid_rows",
                    "first_date",
                    "last_date",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
