"""Attach Binance 5m open interest from PostgreSQL to an existing market cache.

The join is backward-as-of and therefore causal: each market bar can only see the
latest OI row whose timestamp is <= the bar timestamp.  When a Binance UM
metrics CSV is provided, public positioning ratios are attached with exactly one
completed 5m source-bar delay and no ratio interpolation or forward fill.
Credentials are loaded from PG_* environment variables or an env file, but never
printed.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import pandas as pd

from preprocessing.live_db_features import load_env_file, postgres_url_from_env


@dataclass(frozen=True)
class OiEnrichConfig:
    input_csv: str
    output_csv: str
    env_file: str = ".env"
    symbol: str = "BTCUSDT"
    period: str = "5m"
    tolerance: str = "10min"
    metrics_csv: str = ""


def _coerce_naive(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise").dt.tz_convert(None)


POSITIONING_RATIO_COLUMNS = [
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_five_minute_timestamps(series: pd.Series, *, label: str) -> None:
    if series.duplicated().any():
        duplicated = (
            series.loc[series.duplicated(keep=False)].head(5).astype(str).tolist()
        )
        raise ValueError(f"{label} contains duplicate timestamps: {duplicated}")
    epoch_ns = series.astype("int64")
    five_min_ns = pd.Timedelta("5min").value
    if (epoch_ns % five_min_ns != 0).any():
        bad = series.loc[epoch_ns % five_min_ns != 0].head(5).astype(str).tolist()
        raise ValueError(f"{label} timestamps must be aligned to the 5-minute grid: {bad}")


def _load_positioning_metrics(cfg: OiEnrichConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(cfg.metrics_csv, compression="infer")
    required = {"create_time", "symbol", *POSITIONING_RATIO_COLUMNS}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"metrics_csv missing required columns: {missing}")

    metrics = raw[["create_time", "symbol", *POSITIONING_RATIO_COLUMNS]].copy()
    metrics["create_time"] = _coerce_naive(metrics["create_time"])
    symbols = set(metrics["symbol"].astype(str).unique())
    if symbols != {cfg.symbol}:
        raise ValueError(
            f"metrics_csv symbol set {sorted(symbols)} "
            f"does not match requested symbol {cfg.symbol!r}"
        )
    metrics = (
        metrics.drop(columns=["symbol"])
        .sort_values("create_time")
        .reset_index(drop=True)
    )
    five_min_ns = pd.Timedelta("5min").value
    aligned = metrics["create_time"].astype("int64") % five_min_ns == 0
    off_grid_examples = (
        metrics.loc[~aligned, "create_time"].head(5).astype(str).tolist()
    )
    off_grid_rows = int((~aligned).sum())
    metrics = metrics.loc[aligned].reset_index(drop=True)
    if metrics.empty:
        raise ValueError("metrics_csv has no valid 5-minute-grid rows")
    _validate_five_minute_timestamps(
        metrics["create_time"], label="metrics_csv create_time"
    )

    zero_as_missing: dict[str, int] = {}
    for column in POSITIONING_RATIO_COLUMNS:
        raw_values = metrics[column]
        values = pd.to_numeric(raw_values, errors="coerce")
        malformed = raw_values.notna() & values.isna()
        if bool(malformed.any()):
            bad = (
                metrics.loc[malformed, ["create_time", column]]
                .head(5)
                .to_dict("records")
            )
            raise ValueError(
                f"metrics_csv column {column} contains malformed numeric values; "
                f"bad={bad}"
            )
        numeric = values.astype(float)
        invalid_present = values.notna() & (~np.isfinite(numeric) | (numeric < 0.0))
        if bool(invalid_present.any()):
            bad = (
                metrics.loc[invalid_present, ["create_time", column]]
                .head(5)
                .to_dict("records")
            )
            raise ValueError(
                f"metrics_csv column {column} must contain only nonnegative finite "
                f"non-null values; bad={bad}"
            )
        zero = values.eq(0.0)
        zero_as_missing[column] = int(zero.sum())
        values = values.mask(zero)
        metrics[column] = values

    report = {
        "metrics_csv": cfg.metrics_csv,
        "metrics_sha256": _sha256_file(cfg.metrics_csv),
        "metrics_schema": list(raw.columns),
        "metrics_ratio_columns": list(POSITIONING_RATIO_COLUMNS),
        "metrics_zero_as_missing": zero_as_missing,
        "metrics_rows_loaded": int(len(metrics)),
        "metrics_off_grid_rows_dropped": off_grid_rows,
        "metrics_off_grid_examples": off_grid_examples,
        "metrics_range": {
            "start": str(metrics["create_time"].min()) if len(metrics) else "NaT",
            "end": str(metrics["create_time"].max()) if len(metrics) else "NaT",
        },
    }
    return metrics, report


def _attach_delayed_positioning_metrics(
    market: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    base = market.copy()
    source = metrics.rename(columns={"create_time": "positioning_source_time"}).copy()
    base["_positioning_expected_source_time"] = base["date"] - pd.Timedelta("5min")
    merged = base.merge(
        source,
        how="left",
        left_on="_positioning_expected_source_time",
        right_on="positioning_source_time",
        validate="many_to_one",
    )
    all_ratios_present = merged[POSITIONING_RATIO_COLUMNS].notna().all(axis=1)
    exact_age = merged["positioning_source_time"].notna() & (
        (merged["date"] - merged["positioning_source_time"]) == pd.Timedelta("5min")
    )
    merged["positioning_available"] = (all_ratios_present & exact_age).astype(float)
    merged["positioning_age_minutes"] = (
        (merged["date"] - merged["positioning_source_time"]).dt.total_seconds() / 60.0
    )
    merged["positioning_gap"] = merged["positioning_source_time"].isna().astype(float)
    merged["positioning_missing_required"] = (~all_ratios_present).astype(float)
    merged["positioning_exact_one_bar_delay"] = exact_age.astype(float)
    return merged.drop(columns=["_positioning_expected_source_time"])


def _load_oi(cfg: OiEnrichConfig, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    load_env_file(cfg.env_file)
    engine = create_engine(
        postgres_url_from_env(cfg.env_file), connect_args={"connect_timeout": 10}
    )
    query = text(
        """
        SELECT
            ts AS date,
            sum_open_interest AS open_interest,
            sum_open_interest_value AS open_interest_value,
            cmc_circulating_supply
        FROM open_interest_binance
        WHERE symbol = :symbol
          AND period = :period
          AND ts >= :start_ts
          AND ts <= :end_ts
        ORDER BY ts
        """
    )
    with engine.connect() as conn:
        oi = pd.read_sql_query(
            query,
            conn,
            params={
                "symbol": cfg.symbol,
                "period": cfg.period,
                "start_ts": (
                    start.tz_localize("UTC").to_pydatetime()
                    if start.tzinfo is None
                    else start.to_pydatetime()
                ),
                "end_ts": (
                    end.tz_localize("UTC").to_pydatetime()
                    if end.tzinfo is None
                    else end.to_pydatetime()
                ),
            },
        )
    if oi.empty:
        raise ValueError(
            f"no OI rows found for {cfg.symbol} {cfg.period} between {start} and {end}"
        )
    oi["date"] = _coerce_naive(oi["date"])
    for col in ["open_interest", "open_interest_value", "cmc_circulating_supply"]:
        if col in oi.columns:
            oi[col] = pd.to_numeric(oi[col], errors="coerce")
    return (
        oi.dropna(subset=["date", "open_interest"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def run(cfg: OiEnrichConfig) -> dict[str, Any]:
    market = pd.read_csv(cfg.input_csv, parse_dates=["date"], compression="infer")
    market["date"] = _coerce_naive(market["date"])
    market = (
        market.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if cfg.metrics_csv:
        _validate_five_minute_timestamps(market["date"], label="market date")
    start = pd.Timestamp(market["date"].min()) - pd.Timedelta(cfg.tolerance)
    end = pd.Timestamp(market["date"].max()) + pd.Timedelta(cfg.tolerance)
    oi = _load_oi(cfg, start, end)
    metrics_report: dict[str, Any] | None = None
    base = market.copy()
    base["_row"] = range(len(base))
    merged = pd.merge_asof(
        base.sort_values("date"),
        oi.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(cfg.tolerance),
    ).sort_values("_row").drop(columns=["_row"]).reset_index(drop=True)
    merged["open_interest_available"] = merged["open_interest"].notna().astype(float)
    for col in ["open_interest", "open_interest_value", "cmc_circulating_supply"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").ffill()
    if cfg.metrics_csv:
        metrics, metrics_report = _load_positioning_metrics(cfg)
        merged = _attach_delayed_positioning_metrics(merged, metrics)
    Path(cfg.output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cfg.output_csv, index=False, compression="infer")
    available = int(merged["open_interest_available"].sum())
    report = {
        "input_csv": cfg.input_csv,
        "output_csv": cfg.output_csv,
        "market_rows": int(len(market)),
        "oi_rows_loaded": int(len(oi)),
        "oi_range": {"start": str(oi["date"].min()), "end": str(oi["date"].max())},
        "available_rows": available,
        "available_frac": available / len(merged) if len(merged) else 0.0,
        "columns_added": [
            c
            for c in [
                "open_interest",
                "open_interest_value",
                "cmc_circulating_supply",
                "open_interest_available",
                *POSITIONING_RATIO_COLUMNS,
                "positioning_source_time",
                "positioning_available",
                "positioning_age_minutes",
                "positioning_gap",
                "positioning_missing_required",
                "positioning_exact_one_bar_delay",
            ]
            if c in merged.columns
        ],
        "leakage_guard": {
            "backward_asof_join": True,
            "tolerance": cfg.tolerance,
            "no_forward_fill_before_join": True,
        },
    }
    if metrics_report is not None:
        report["positioning_metrics"] = metrics_report
        report["positioning_available_rows"] = int(merged["positioning_available"].sum())
        report["positioning_available_frac"] = (
            float(merged["positioning_available"].mean()) if len(merged) else 0.0
        )
        report["positioning_leakage_guard"] = {
            "exact_one_completed_source_bar_delay": True,
            "source_bar_delay": "5min",
            "ratio_forward_fill": False,
            "ratio_interpolation": False,
            "oi_forward_fill_semantics_separate": True,
        }
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--env-file", default=OiEnrichConfig.env_file)
    p.add_argument("--symbol", default=OiEnrichConfig.symbol)
    p.add_argument("--period", default=OiEnrichConfig.period)
    p.add_argument("--tolerance", default=OiEnrichConfig.tolerance)
    p.add_argument("--metrics-csv", default=OiEnrichConfig.metrics_csv)
    return p.parse_args()


def main() -> None:
    import json
    report = run(OiEnrichConfig(**vars(parse_args())))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
