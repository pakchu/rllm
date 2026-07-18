"""Backtest the deterministic pre-fix REX policy as ``wrong_rex``.

This intentionally freezes the old causal formula and gates.  It does *not*
attempt to turn the former positional tail-cache drift into an alpha: that bug
depended on process start time and cache length and therefore has no stable
historical strategy definition.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, create_engine, text

from preprocessing.external_features import (
    DXY_WEIGHTS,
    attach_external_features,
    build_external_feature_frame,
)
from preprocessing.live_db_features import load_env_file, postgres_url_from_env
from preprocessing.market_features import build_market_feature_frame
from training.event_candidate_pool_probe import _feature_candidates
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class WrongRexConfig:
    env_file: str = ".env"
    output: str = "results/wrong_rex_backtest_2026-07-14.json"
    cache: str = "/tmp/rllm_wrong_rex_db_5m_v2.pkl"
    start: str = "2020-06-01"
    end: str = "2026-07-15"
    quantile: float = 0.75
    min_positive_strengths: int = 50
    hold_bars: int = 144
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002
    leverage: float = 1.0
    rebuild_cache: bool = False


SPLITS = {
    "train": ("2020-09-01", "2024-01-01"),
    "test2024": ("2024-01-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    "ytd2026": ("2026-01-01", "2026-07-14 23:59:59"),
}


def _query_aggregate(
    conn: Any,
    table: str,
    *,
    symbols: tuple[str, ...],
    start: str,
    end: str,
    market: bool,
) -> pd.DataFrame:
    extras = (
        """
        (array_agg(open ORDER BY ts))[1] AS open,
        max(high) AS high,
        min(low) AS low,
        (array_agg(close ORDER BY ts DESC))[1] AS close,
        sum(volume) AS volume,
        sum(number_of_trades) AS number_of_trades,
        sum(taker_buy_base) AS taker_buy_base,
        count(*) AS source_rows
    """
        if market
        else """
        (array_agg(close ORDER BY ts DESC))[1] AS close,
        count(*) AS source_rows
    """
    )
    stmt = text(f"""
        SELECT date_bin('5 minutes', ts, TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,
               symbol, {extras}
        FROM {table}
        WHERE interval = '1m' AND symbol IN :symbols AND ts >= :start AND ts < :end
        GROUP BY 1, symbol
        ORDER BY 1, symbol
    """).bindparams(bindparam("symbols", expanding=True))
    return pd.read_sql_query(
        stmt, conn, params={"symbols": symbols, "start": start, "end": end}
    )


def _load_db_frame(cfg: WrongRexConfig) -> pd.DataFrame:
    cache = Path(cfg.cache).expanduser()
    if cache.exists() and not cfg.rebuild_cache:
        return pd.read_pickle(cache)
    load_env_file(cfg.env_file)
    engine = create_engine(
        postgres_url_from_env(cfg.env_file), connect_args={"connect_timeout": 10}
    )
    with engine.connect() as conn:
        market = _query_aggregate(
            conn,
            "bars_binance",
            symbols=("BTCUSDT",),
            start=cfg.start,
            end=cfg.end,
            market=True,
        )
        upbit = _query_aggregate(
            conn,
            "bars_upbit",
            symbols=("KRW-BTC",),
            start=cfg.start,
            end=cfg.end,
            market=False,
        )
        fx_symbols = tuple(DXY_WEIGHTS) + ("USDKRW",)
        polygon = _query_aggregate(
            conn,
            "bars_polygon",
            symbols=fx_symbols,
            start=cfg.start,
            end=cfg.end,
            market=False,
        )
        oi = pd.read_sql_query(
            text("""
                WITH historical AS (
                    SELECT ts AS date, sum_open_interest AS open_interest, 1 AS priority
                    FROM open_interest_binance
                    WHERE symbol='BTCUSDT' AND period='5m' AND ts >= :start AND ts < :end
                ), live_ranked AS (
                    SELECT date_bin('5 minutes', fetched_at, TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,
                           open_interest,
                           ROW_NUMBER() OVER (
                               PARTITION BY date_bin('5 minutes', fetched_at, TIMESTAMPTZ '1970-01-01 00:00:00+00')
                               ORDER BY fetched_at DESC
                           ) AS snapshot_rank
                    FROM open_interest_binance_live
                    WHERE symbol='BTCUSDT' AND fetched_at >= :start AND fetched_at < :end
                ), combined AS (
                    SELECT date, open_interest, priority FROM historical
                    UNION ALL
                    SELECT date, open_interest, 0 AS priority FROM live_ranked WHERE snapshot_rank=1
                )
                SELECT DISTINCT ON (date) date, open_interest
                FROM combined
                ORDER BY date, priority
            """),
            conn,
            params={"start": cfg.start, "end": cfg.end},
        )

    for frame in (market, upbit, polygon, oi):
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert(None)
    market = (
        market.loc[market["source_rows"].eq(5)]
        .drop(columns=["source_rows", "symbol"])
        .reset_index(drop=True)
    )
    market["tic"] = "BTCUSDT"
    market = pd.merge_asof(
        market.sort_values("date"),
        oi.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("10min"),
    )
    market["open_interest_available"] = market["open_interest"].notna().astype(float)
    upbit = upbit.rename(columns={"symbol": "tic"}).drop(columns=["source_rows"])
    polygon = polygon.rename(columns={"symbol": "tic"}).drop(columns=["source_rows"])
    external = build_external_feature_frame(
        market,
        forex_bars=polygon[polygon["tic"].isin(DXY_WEIGHTS)],
        btckrw_bars=upbit,
        usdkrw_bars=polygon[polygon["tic"].eq("USDKRW")],
        interval="5min",
        include_forex_components=False,
    )
    enriched = attach_external_features(
        market, external, tolerance="10min", zscore_window=96, momentum_period=96
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_pickle(cache)
    return enriched


def _expanding_positive_quantile(
    strength: np.ndarray, quantile: float, min_count: int
) -> np.ndarray:
    positive = pd.Series(
        np.where(np.isfinite(strength) & (strength > 0.0), strength, np.nan)
    )
    return (
        positive.expanding(min_periods=int(min_count))
        .quantile(float(quantile))
        .to_numpy(float)
    )


def _weekend_fx_closed(dates: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(dates, utc=True)
    return np.asarray(
        (dt.dt.dayofweek.eq(5))
        | (dt.dt.dayofweek.eq(6) & dt.dt.hour.lt(22))
        | (dt.dt.dayofweek.eq(4) & dt.dt.hour.ge(22))
    )


def _candidate_mask(
    enriched: pd.DataFrame, features: pd.DataFrame, cfg: WrongRexConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strength, direction = _feature_candidates(features)["rex_htf_pullback_reclaim"]
    threshold = _expanding_positive_quantile(
        strength, cfg.quantile, cfg.min_positive_strengths
    )
    candidate = (
        np.isfinite(strength)
        & np.isfinite(threshold)
        & (strength > threshold)
        & (direction != 0.0)
    )
    primary = (features["range_vol"].to_numpy(float) >= 0.023959233645008706) & (
        features["kimchi_premium_change"].to_numpy(float) <= 0.0
    )
    alternate = (
        features["rex_8640_range_width_pct"].to_numpy(float) >= 0.2836633876944003
    ) & (features["usdkrw_zscore"].to_numpy(float) <= 0.2603593471820541)
    quality = _weekend_fx_closed(enriched["date"])
    weekday_quality = np.ones(len(enriched), dtype=bool)
    for col in ("dxy_available", "kimchi_available", "usdkrw_available"):
        weekday_quality &= (
            enriched.get(col, pd.Series(0.0, index=enriched.index)).to_numpy(float)
            >= 1.0
        )
    quality = quality | weekday_quality
    veto7 = (features["htf_1w_return_4"].to_numpy(float) >= -0.26588062806734514) & (
        features["oi_zscore"].to_numpy(float) <= 1.5910475818293068
    )
    return candidate & (primary | alternate) & quality & veto7, direction, threshold


def _simulate(
    market: pd.DataFrame,
    positions: np.ndarray,
    direction: np.ndarray,
    cfg: WrongRexConfig,
    start: str,
    end: str,
) -> dict[str, Any]:
    op = market["open"].to_numpy(float)
    hi = market["high"].to_numpy(float)
    lo = market["low"].to_numpy(float)
    eq = peak = 1.0
    max_dd = 0.0
    next_allowed = 0
    returns: list[float] = []
    executed: list[dict[str, Any]] = []
    side_counts = {"LONG": 0, "SHORT": 0}
    side_cost = (cfg.fee_rate + cfg.slippage_rate) * cfg.leverage
    for pos in positions:
        if pos < next_allowed:
            continue
        ep = int(pos) + cfg.entry_delay_bars
        xp = ep + cfg.hold_bars
        if xp >= len(market) or pd.Timestamp(market.iloc[xp]["date"]) >= pd.Timestamp(
            end
        ):
            continue
        side = 1 if direction[pos] > 0 else -1
        entry_eq = eq
        eq *= max(0.0, 1.0 - side_cost)
        max_dd = max(max_dd, 1.0 - eq / peak)
        for j in range(ep, xp):
            adverse = ((lo[j] - op[j]) if side > 0 else (op[j] - hi[j])) / op[j]
            max_dd = max(
                max_dd, 1.0 - max(0.0, eq * (1.0 + cfg.leverage * adverse)) / peak
            )
            bar_ret = ((op[j + 1] - op[j]) if side > 0 else (op[j] - op[j + 1])) / op[j]
            eq *= max(0.0, 1.0 + cfg.leverage * bar_ret)
            peak = max(peak, eq)
        eq *= max(0.0, 1.0 - side_cost)
        max_dd = max(max_dd, 1.0 - eq / peak)
        peak = max(peak, eq)
        trade_ret = eq / entry_eq - 1.0
        returns.append(trade_ret)
        label = "LONG" if side > 0 else "SHORT"
        side_counts[label] += 1
        executed.append(
            {
                "signal_date": str(market.iloc[pos]["date"]),
                "entry_date": str(market.iloc[ep]["date"]),
                "exit_date": str(market.iloc[xp]["date"]),
                "side": label,
                "ret_pct": trade_ret * 100.0,
            }
        )
        next_allowed = xp
    years = max(
        1 / 365.25,
        (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 86400),
    )
    ret = eq - 1.0
    cagr = (eq ** (1 / years) - 1.0) if eq > 0 else -1.0
    return {
        "sim": {
            "return_pct": ret * 100,
            "cagr_pct": cagr * 100,
            "strict_mdd_pct": max_dd * 100,
            "cagr_to_strict_mdd": cagr / max_dd if max_dd > 0 else None,
            "trade_entries": len(returns),
            "side_counts": side_counts,
            "win_rate_pct": 100 * sum(x > 0 for x in returns) / len(returns)
            if returns
            else 0.0,
        },
        "trade_stats": _trade_stats(returns),
        "executed": executed,
    }


def run(cfg: WrongRexConfig) -> dict[str, Any]:
    market = _load_db_frame(cfg)
    features = build_market_feature_frame(
        market, window_size=288, zscore_window=96, volume_window=96
    )
    active, direction, threshold = _candidate_mask(market, features, cfg)
    split_results: dict[str, Any] = {}
    dates = pd.to_datetime(market["date"])
    for name, (start, end) in SPLITS.items():
        positions = np.flatnonzero(
            active
            & np.asarray((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)))
        )
        variants = {
            "all": _simulate(market, positions, direction, cfg, start, end),
            "long_only": _simulate(
                market,
                positions[direction[positions] > 0.0],
                direction,
                cfg,
                start,
                end,
            ),
            "short_only": _simulate(
                market,
                positions[direction[positions] < 0.0],
                direction,
                cfg,
                start,
                end,
            ),
        }
        split_results[name] = {
            **variants["all"],
            "raw_signal_count": int(len(positions)),
            "variants": variants,
        }
    result = {
        "name": "wrong_rex",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "definition": "deterministic pre-fix dynamic-q75 REX formula + old dual rule gate + veto7; excludes positional cache drift",
        "config": asdict(cfg),
        "data": {
            "rows": len(market),
            "start": str(market["date"].min()),
            "end": str(market["date"].max()),
            "latest_threshold": float(threshold[np.isfinite(threshold)][-1]),
        },
        "splits": split_results,
    }
    path = Path(cfg.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in asdict(WrongRexConfig()).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            p.add_argument(flag, action="store_true")
        else:
            p.add_argument(flag, type=type(default), default=default)
    return p.parse_args()


if __name__ == "__main__":
    out = run(WrongRexConfig(**vars(parse_args())))
    print(
        json.dumps(
            {k: v["sim"] for k, v in out["splits"].items()},
            indent=2,
            ensure_ascii=False,
        )
    )
