"""Build live REX/RLLM feature snapshots from database-sourced frames.

The functions here keep the live path independent from historical CSV caches:
BTCUSDT, KRW-BTC, USDKRW, synthetic-DXY FX components, premium-index klines,
and funding rows can all be supplied from PostgreSQL query results.  Joins are
backward-as-of only and premium values are timestamped by close_time.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.binance_aux_features import attach_binance_um_aux_frames
from preprocessing.external_features import DXY_WEIGHTS, attach_external_features, build_external_feature_frame
from preprocessing.market_features import build_market_feature_frame


@dataclass(frozen=True)
class LiveDbFeatureConfig:
    symbol: str = "BTCUSDT"
    base_interval: str = "1min"
    decision_interval: str = "5min"
    lookback_minutes: int = 45_000
    external_tolerance: str = "10min"
    premium_tolerance: str = "10min"
    funding_tolerance: str = "12h"
    zscore_window: int = 96
    feature_window_size: int = 288
    volume_window: int = 96
    include_forex_components: bool = True


def load_env_file(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE env lines without printing or returning secret values."""

    env_path = Path(path).expanduser()
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def postgres_url_from_env(env_path: str | Path = ".env") -> str:
    """Return PostgreSQL URL from PG_* env vars, without logging credentials."""

    load_env_file(env_path)
    required = ["PG_USER", "PG_PASSWORD", "PG_HOST", "PG_PORT", "PG_DB_NAME"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise ValueError(f"missing PostgreSQL env keys: {missing}")
    return (
        f"postgresql://{os.environ['PG_USER']}:{os.environ['PG_PASSWORD']}"
        f"@{os.environ['PG_HOST']}:{os.environ['PG_PORT']}/{os.environ['PG_DB_NAME']}"
    )


def sqlalchemy_engine_from_env(env_path: str | Path = ".env") -> Any:
    """Create a SQLAlchemy engine when SQLAlchemy is installed in the runtime."""

    try:
        from sqlalchemy import create_engine
    except Exception as exc:  # pragma: no cover - depends on runtime packages
        raise RuntimeError("SQLAlchemy is required for direct DB querying; use wave_trading venv or pass frames directly") from exc
    return create_engine(postgres_url_from_env(env_path), connect_args={"connect_timeout": 5})


def _normalise_bar_frame(df: pd.DataFrame, *, tic: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if "ts" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"ts": "date"})
    if "date" not in out.columns:
        raise ValueError("bar frame must contain date or ts")
    out["date"] = pd.to_datetime(out["date"], errors="raise", utc=True).dt.tz_convert(None)
    if tic is not None and "tic" not in out.columns:
        out["tic"] = tic
    for col in ["open", "high", "low", "close", "volume", "number_of_trades", "taker_buy_base"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("date").drop_duplicates(["date", "tic"] if "tic" in out.columns else ["date"], keep="last").reset_index(drop=True)


def resample_market_bars(market_1m: pd.DataFrame, interval: str = "5min") -> pd.DataFrame:
    """Resample open-time 1m OHLCV rows to open-time decision bars."""

    market = _normalise_bar_frame(market_1m, tic="BTCUSDT")
    agg: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for optional in ["number_of_trades", "taker_buy_base", "taker_buy_quote", "quote_asset_volume"]:
        if optional in market.columns:
            agg[optional] = "sum"
    resampled = market.set_index("date").resample(interval, label="left", closed="left").agg(agg).dropna(subset=["open", "high", "low", "close"])
    row_counts = market.set_index("date").resample(interval, label="left", closed="left")["close"].count()
    expected_rows = max(1, int(round(pd.Timedelta(interval).total_seconds() / 60.0)))
    resampled = resampled.loc[row_counts[row_counts >= expected_rows].index]
    out = resampled.reset_index()
    out["tic"] = "BTCUSDT"
    out["day"] = out["date"].dt.dayofweek
    return out.reset_index(drop=True)


def _attach_1m_decision_close_and_rows(
    market: pd.DataFrame,
    source_1m: pd.DataFrame,
    *,
    close_col: str,
    rows_col: str,
    interval: str,
) -> pd.DataFrame:
    """Attach exact per-decision 1m close/count fields by open-time bucket.

    The count column is intentionally the observed source-row count, not a
    boolean availability flag.  Downstream live parity gates can therefore
    fail closed by requiring the exact expected count for the decision bar.
    """

    out = market.copy()
    source = _normalise_bar_frame(source_1m)
    if source.empty:
        out[close_col] = np.nan
        out[rows_col] = 0
        return out
    grouped = source.set_index("date").resample(interval, label="left", closed="left")
    summary = grouped.agg({"close": "last"}).rename(columns={"close": close_col})
    summary[rows_col] = grouped["close"].count().astype(int)
    summary = summary.reset_index()
    return out.merge(summary[["date", close_col, rows_col]], on="date", how="left", validate="one_to_one").assign(
        **{rows_col: lambda frame: pd.to_numeric(frame[rows_col], errors="coerce").fillna(0).astype(int)}
    )


def build_live_feature_frame_from_frames(
    *,
    btcusdt_1m: pd.DataFrame,
    btckrw_1m: pd.DataFrame,
    usdkrw_1m: pd.DataFrame,
    forex_1m: pd.DataFrame,
    premium_1m: pd.DataFrame,
    funding: pd.DataFrame,
    spot_1m: pd.DataFrame | None = None,
    cfg: LiveDbFeatureConfig = LiveDbFeatureConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(enriched_market, feature_frame)`` from DB-sourced frames."""

    market = resample_market_bars(btcusdt_1m, cfg.decision_interval)
    btckrw = _normalise_bar_frame(btckrw_1m, tic="KRW-BTC")
    usdkrw = _normalise_bar_frame(usdkrw_1m, tic="USDKRW")
    forex = _normalise_bar_frame(forex_1m)
    external = build_external_feature_frame(
        market,
        forex_bars=forex,
        btckrw_bars=btckrw,
        usdkrw_bars=usdkrw,
        interval=cfg.decision_interval,
        include_forex_components=cfg.include_forex_components,
    )
    enriched = attach_external_features(
        market,
        external,
        tolerance=cfg.external_tolerance,
        zscore_window=cfg.zscore_window,
        momentum_period=cfg.zscore_window,
    )
    enriched = attach_binance_um_aux_frames(
        enriched,
        funding_frame=funding,
        premium_frame=premium_1m,
        funding_tolerance=cfg.funding_tolerance,
        premium_tolerance=cfg.premium_tolerance,
        zscore_window=cfg.zscore_window,
    )
    enriched = _attach_1m_decision_close_and_rows(
        enriched,
        premium_1m,
        close_col="premium_index_1m_close",
        rows_col="premium_rows",
        interval=cfg.decision_interval,
    )
    if spot_1m is not None:
        enriched = _attach_1m_decision_close_and_rows(
            enriched,
            spot_1m,
            close_col="spot_close",
            rows_col="spot_rows",
            interval=cfg.decision_interval,
        )
    features = build_market_feature_frame(
        enriched,
        window_size=cfg.feature_window_size,
        zscore_window=cfg.zscore_window,
        volume_window=cfg.volume_window,
    )
    return enriched, features


def latest_live_feature_snapshot(enriched: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    """Serialize the latest row as a replayable live feature snapshot."""

    if enriched.empty or features.empty:
        raise ValueError("cannot build snapshot from empty frames")
    idx = len(enriched) - 1
    date = pd.to_datetime(enriched.iloc[idx]["date"], utc=True)
    feature_row = features.iloc[idx]
    snapshot = {
        "date": date.isoformat(),
        "feature_snapshot": {k: float(v) for k, v in feature_row.items() if isinstance(v, (int, float, np.number)) and np.isfinite(float(v))},
        "data_quality": {
            "dxy_available": float(enriched.iloc[idx].get("dxy_available", 0.0)),
            "kimchi_available": float(enriched.iloc[idx].get("kimchi_available", 0.0)),
            "usdkrw_available": float(enriched.iloc[idx].get("usdkrw_available", 0.0)),
            "premium_available": float(enriched.iloc[idx].get("premium_available", 0.0)),
            "funding_available": float(enriched.iloc[idx].get("funding_available", 0.0)),
            "external_any_available": float(enriched.iloc[idx].get("external_any_available", 0.0)),
            "binance_aux_any_available": float(enriched.iloc[idx].get("binance_aux_any_available", 0.0)),
        },
    }
    return snapshot


def live_source_sql(cfg: LiveDbFeatureConfig, *, asof_param: str = "asof", start_param: str = "start") -> dict[str, str]:
    """Return read-only SQL templates for the live DB source frames."""

    return {
        "btcusdt_1m": f"""
            SELECT
                ts AS date,
                open, high, low, close, volume,
                quote_asset_volume, number_of_trades, taker_buy_base, taker_buy_quote,
                symbol AS tic
            FROM bars_binance
            WHERE symbol = '{cfg.symbol}' AND interval = '1m' AND ts >= :{start_param} AND ts <= :{asof_param}
            ORDER BY ts
        """,
        "btckrw_1m": f"""
            SELECT ts AS date, open, high, low, close, volume, symbol AS tic
            FROM bars_upbit
            WHERE symbol = 'KRW-BTC' AND interval = '1m' AND ts >= :{start_param} AND ts <= :{asof_param}
            ORDER BY ts
        """,
        "usdkrw_1m": f"""
            SELECT ts AS date, open, high, low, close, volume, symbol AS tic
            FROM bars_polygon
            WHERE symbol = 'USDKRW' AND interval = '1m' AND ts >= :{start_param} AND ts <= :{asof_param}
            ORDER BY ts
        """,
        "forex_1m": f"""
            SELECT ts AS date, open, high, low, close, volume, symbol AS tic
            FROM bars_polygon
            WHERE symbol IN ({', '.join(repr(s) for s in DXY_WEIGHTS)}) AND interval = '1m' AND ts >= :{start_param} AND ts <= :{asof_param}
            ORDER BY ts, symbol
        """,
        "premium_1m": f"""
            SELECT ts AS date, close_time, close
            FROM bars_binance_premium
            WHERE symbol = '{cfg.symbol}' AND interval = '1m' AND ts >= :{start_param} AND ts <= :{asof_param}
            ORDER BY ts
        """,
        "spot_1m": f"""
            SELECT
                ts AS date,
                open, high, low, close, volume,
                quote_asset_volume, number_of_trades, taker_buy_base, taker_buy_quote,
                symbol AS tic
            FROM bars_binance_spot
            WHERE symbol = '{cfg.symbol}' AND interval = '1m' AND ts >= :{start_param} AND ts <= :{asof_param}
            ORDER BY ts
        """,
        "funding": f"""
            SELECT funding_time, funding_rate, mark_price
            FROM funding_rates_binance
            WHERE symbol = '{cfg.symbol}' AND funding_time <= :{asof_param}
            ORDER BY funding_time DESC
            LIMIT 2000
        """,
    }


def query_live_source_frames(
    engine_or_conn: Any,
    *,
    asof: str | pd.Timestamp,
    cfg: LiveDbFeatureConfig = LiveDbFeatureConfig(),
    start_by_key: dict[str, pd.Timestamp] | None = None,
) -> dict[str, pd.DataFrame]:
    """Query all DB source frames with SQLAlchemy-style named parameters."""

    try:
        from sqlalchemy import text
    except Exception as exc:  # pragma: no cover - depends on runtime packages
        raise RuntimeError("SQLAlchemy is required for direct DB querying") from exc
    asof_ts = pd.Timestamp(asof, tz="UTC") if pd.Timestamp(asof).tzinfo is None else pd.Timestamp(asof).tz_convert("UTC")
    start_ts = asof_ts - pd.Timedelta(minutes=int(cfg.lookback_minutes))
    frames: dict[str, pd.DataFrame] = {}
    sql_map = live_source_sql(cfg)
    with engine_or_conn.connect() if hasattr(engine_or_conn, "connect") else engine_or_conn as conn:
        for key, sql in sql_map.items():
            key_start = pd.Timestamp((start_by_key or {}).get(key, start_ts))
            key_start = key_start.tz_localize("UTC") if key_start.tzinfo is None else key_start.tz_convert("UTC")
            frames[key] = pd.read_sql_query(
                text(sql),
                conn,
                params={"asof": asof_ts.to_pydatetime(), "start": key_start.to_pydatetime()},
            )
    return frames


def build_latest_snapshot_from_db(engine_or_conn: Any, *, asof: str | pd.Timestamp, cfg: LiveDbFeatureConfig = LiveDbFeatureConfig()) -> dict[str, Any]:
    frames = query_live_source_frames(engine_or_conn, asof=asof, cfg=cfg)
    enriched, features = build_live_feature_frame_from_frames(cfg=cfg, **frames)
    snapshot = latest_live_feature_snapshot(enriched, features)
    snapshot["config"] = asdict(cfg)
    return snapshot
