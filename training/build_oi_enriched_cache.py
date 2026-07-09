"""Attach Binance 5m open interest from PostgreSQL to an existing market cache.

The join is backward-as-of and therefore causal: each market bar can only see the
latest OI row whose timestamp is <= the bar timestamp.  Credentials are loaded
from PG_* environment variables or an env file, but never printed.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from preprocessing.live_db_features import load_env_file, postgres_url_from_env


@dataclass(frozen=True)
class OiEnrichConfig:
    input_csv: str
    output_csv: str
    env_file: str = ".env"
    symbol: str = "BTCUSDT"
    period: str = "5m"
    tolerance: str = "10min"


def _coerce_naive(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise").dt.tz_convert(None)


def _load_oi(cfg: OiEnrichConfig, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    load_env_file(cfg.env_file)
    engine = create_engine(postgres_url_from_env(cfg.env_file), connect_args={"connect_timeout": 10})
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
                "start_ts": start.tz_localize("UTC").to_pydatetime() if start.tzinfo is None else start.to_pydatetime(),
                "end_ts": end.tz_localize("UTC").to_pydatetime() if end.tzinfo is None else end.to_pydatetime(),
            },
        )
    if oi.empty:
        raise ValueError(f"no OI rows found for {cfg.symbol} {cfg.period} between {start} and {end}")
    oi["date"] = _coerce_naive(oi["date"])
    for col in ["open_interest", "open_interest_value", "cmc_circulating_supply"]:
        if col in oi.columns:
            oi[col] = pd.to_numeric(oi[col], errors="coerce")
    return oi.dropna(subset=["date", "open_interest"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def run(cfg: OiEnrichConfig) -> dict[str, Any]:
    market = pd.read_csv(cfg.input_csv, parse_dates=["date"], compression="infer")
    market["date"] = _coerce_naive(market["date"])
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    start = pd.Timestamp(market["date"].min()) - pd.Timedelta(cfg.tolerance)
    end = pd.Timestamp(market["date"].max()) + pd.Timedelta(cfg.tolerance)
    oi = _load_oi(cfg, start, end)
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
        "columns_added": [c for c in ["open_interest", "open_interest_value", "cmc_circulating_supply", "open_interest_available"] if c in merged.columns],
        "leakage_guard": {"backward_asof_join": True, "tolerance": cfg.tolerance, "no_forward_fill_before_join": True},
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
    return p.parse_args()


def main() -> None:
    import json
    report = run(OiEnrichConfig(**vars(parse_args())))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
