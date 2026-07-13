"""Search causal BTC alpha in unexplained Deribit implied-volatility shocks."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_deribit_dvol_alpha import attach_dvol
from training.search_nested_barrier_witness_alpha import (
    admission,
    print_stats,
    rank_key,
    simulate,
)
from training.search_positioning_disagreement_alpha import _future_extreme
from training.search_positioning_hgb_path_alpha import _read_before

MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
SPOT = "data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz"
DVOL = "data/deribit_btc_dvol_1h_2020-09-01_2026-06-02.csv.gz"
CUTOFF = "2024-01-01"
FIT_START = "2021-04-15"
FIT_END = "2023-01-01"


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pre2024_bundle() -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(MARKET, "date", CUTOFF)
    spot = _read_before(SPOT, "date", CUTOFF)
    dvol = _read_before(DVOL, "close_time", CUTOFF)
    for frame in (market, spot):
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last")
    spot = spot.sort_values("date").drop_duplicates("date", keep="last")
    market = market.merge(
        spot[
            [
                "date",
                "spot_close",
                "spot_rows",
                "premium_index_1m_close",
                "premium_rows",
            ]
        ],
        on="date",
        how="left",
        validate="one_to_one",
    ).reset_index(drop=True)
    market["premium_available"] = (
        pd.to_numeric(market["premium_rows"], errors="coerce").eq(5)
        & pd.to_numeric(market["premium_index_1m_close"], errors="coerce").notna()
    )
    market["spot_available"] = (
        pd.to_numeric(market["spot_rows"], errors="coerce").eq(5)
        & pd.to_numeric(market["spot_close"], errors="coerce").gt(0.0)
        & market["premium_available"]
    )
    market = attach_dvol(market, dvol, tolerance="65min")
    dates = pd.to_datetime(market["date"])
    if dates.max() >= pd.Timestamp(CUTOFF):
        raise RuntimeError("future rows opened")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("orphan-convexity search requires a complete futures 5-minute grid")
    return market, dates


def causal_rolling_residual(
    target: np.ndarray,
    design: np.ndarray,
    *,
    window: int,
    min_observations: int,
) -> np.ndarray:
    """Predict each row from a rolling OLS fit containing only earlier rows."""
    target = np.asarray(target, dtype=float)
    design = np.asarray(design, dtype=float)
    output = np.full(len(target), np.nan)
    observations: deque[tuple[np.ndarray, float]] = deque()
    xtx = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    xty = np.zeros(design.shape[1], dtype=float)
    for position, (features, value) in enumerate(zip(design, target)):
        if not (np.isfinite(features).all() and np.isfinite(value)):
            continue
        if len(observations) >= min_observations:
            coefficient = np.linalg.lstsq(xtx, xty, rcond=1e-10)[0]
            output[position] = value - features @ coefficient
        observations.append((features.copy(), float(value)))
        xtx += np.outer(features, features)
        xty += features * value
        if len(observations) > window:
            old_features, old_value = observations.popleft()
            xtx -= np.outer(old_features, old_features)
            xty -= old_features * old_value
    return output


def _hourly_dvol_change(market: pd.DataFrame, update: np.ndarray) -> np.ndarray:
    output = np.full(len(market), np.nan)
    indices = np.flatnonzero(update)
    if len(indices) < 2:
        return output
    close_time = pd.to_datetime(market.loc[indices, "close_time"]).reset_index(drop=True)
    close = np.log(pd.to_numeric(market.loc[indices, "dvol_close"], errors="coerce").to_numpy(float))
    consecutive = close_time.diff().eq(pd.Timedelta("1h")).to_numpy(bool)
    changes = np.r_[np.nan, np.diff(close)]
    valid_indices = indices[consecutive]
    output[valid_indices] = changes[consecutive]
    return output


def build_orphan_features(market: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    close = pd.to_numeric(market["close"], errors="coerce")
    high = pd.to_numeric(market["high"], errors="coerce")
    low = pd.to_numeric(market["low"], errors="coerce")
    log_return = np.log(close.where(close > 0.0)).diff()
    realized_variance = log_return.pow(2).rolling(12, min_periods=12).sum()
    realized_range = np.log(
        high.rolling(12, min_periods=12).max()
        / low.rolling(12, min_periods=12).min().replace(0.0, np.nan)
    )
    price_jump = np.log(close / close.shift(12)).abs()
    premium = pd.to_numeric(market["premium_index_1m_close"], errors="coerce").where(
        market["premium_available"]
    )
    premium_change = premium.diff(12)

    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    taker_flow = ((2.0 * taker_buy - quote) / quote.replace(0.0, np.nan)).rolling(
        12, min_periods=12
    ).mean()
    spot = pd.to_numeric(market["spot_close"], errors="coerce")
    basis_residual = (np.log(close / spot) - premium).where(market["spot_available"])
    basis_impulse = basis_residual.diff(12)

    close_time = pd.to_datetime(market["close_time"])
    update = (close_time.notna() & close_time.ne(close_time.shift(1))).to_numpy(bool)
    dvol_change = _hourly_dvol_change(market, update)
    fit_mask = (
        (dates >= pd.Timestamp(FIT_START))
        & (dates < pd.Timestamp(FIT_END))
        & pd.Series(update, index=dates.index)
    ).to_numpy(bool)

    raw_design = np.column_stack(
        [realized_variance, realized_range, price_jump, premium_change]
    ).astype(float)
    # Keep one complete cross-venue observation contract for both residual
    # estimation and either direction proxy. Partial spot aggregates must not
    # silently contribute to the rolling model used by a taker-flow policy.
    raw_design[~market["spot_available"].to_numpy(bool)] = np.nan
    fit_design = raw_design[fit_mask]
    center = np.nanmean(fit_design, axis=0)
    scale = np.nanstd(fit_design, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    design = np.column_stack([np.ones(len(market)), (raw_design - center) / scale])

    output = {
        "dvol_update": update.astype(float),
        "dvol_change": dvol_change,
        "taker_flow_1h": taker_flow.to_numpy(float),
        "spot_perp_basis_impulse_1h": basis_impulse.to_numpy(float),
        "realized_variance_1h": realized_variance.to_numpy(float),
        "fit_reference": fit_mask.astype(float),
    }
    for days in (30, 90):
        output[f"orphan_residual_{days}d"] = causal_rolling_residual(
            dvol_change,
            design,
            window=days * 24,
            min_observations=days * 12,
        )
    return pd.DataFrame(output, index=market.index).replace([np.inf, -np.inf], np.nan)


def orphan_signals(
    features: pd.DataFrame,
    *,
    residual_days: int,
    tail: float,
    direction_proxy: str,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    residual = features[f"orphan_residual_{residual_days}d"].to_numpy(float)
    proxy = features[direction_proxy].to_numpy(float)
    fit = features["fit_reference"].to_numpy(bool)
    reference = residual[fit & np.isfinite(residual)]
    threshold = float(np.quantile(reference, tail))
    event = (
        features["dvol_update"].to_numpy(bool)
        & np.isfinite(residual)
        & (residual >= threshold)
        & np.isfinite(proxy)
    )
    long_active = event & (proxy > 0.0)
    short_active = event & (proxy < 0.0)
    if flip:
        long_active, short_active = short_active, long_active
    return long_active, short_active, threshold


def _raw_tail_signals(
    features: pd.DataFrame,
    *,
    feature: str,
    tail: float,
    direction_proxy: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = features[feature].to_numpy(float)
    proxy = features[direction_proxy].to_numpy(float)
    fit = features["fit_reference"].to_numpy(bool)
    threshold = float(np.quantile(values[fit & np.isfinite(values)], tail))
    event = (
        features["dvol_update"].to_numpy(bool)
        & np.isfinite(values)
        & (values >= threshold)
        & np.isfinite(proxy)
    )
    return event & (proxy > 0.0), event & (proxy < 0.0)


def main() -> None:
    market, dates = load_pre2024_bundle()
    features = build_orphan_features(market, dates)
    holds = (144, 288)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    proxies = ("taker_flow_1h", "spot_perp_basis_impulse_1h")
    rows: list[dict[str, Any]] = []
    signal_bank: dict[tuple[int, float, str], tuple[np.ndarray, np.ndarray]] = {}
    for residual_days, tail, direction_proxy in itertools.product(
        (30, 90),
        (0.90, 0.95),
        proxies,
    ):
        long_active, short_active, threshold = orphan_signals(
            features,
            residual_days=residual_days,
            tail=tail,
            direction_proxy=direction_proxy,
        )
        signal_bank[(residual_days, tail, direction_proxy)] = (long_active, short_active)
        for hold in holds:
            stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
            rows.append(
                {
                    "residual_days": residual_days,
                    "tail": tail,
                    "direction_proxy": direction_proxy,
                    "hold": hold,
                    "threshold": threshold,
                    "signals": int((long_active | short_active).sum()),
                    "rank": rank_key(stats),
                    "prelim_admitted": admission(stats),
                    "stats": stats,
                }
            )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "admitted", sum(row["prelim_admitted"] for row in rows))
    for index, row in enumerate(rows, 1):
        print_stats(
            f"RANK {index} d{row['residual_days']} q{row['tail']} "
            f"{row['direction_proxy']} h{row['hold']} sig={row['signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    key = (top["residual_days"], top["tail"], top["direction_proxy"])
    long_active, short_active = signal_bank[key]
    hold = top["hold"]
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(
            market, dates, short_active, long_active, hold, extremes[hold]
        )
    }
    lag = 144
    controls["signal_lag_12h"] = simulate(
        market,
        dates,
        np.r_[np.zeros(lag, dtype=bool), long_active[:-lag]],
        np.r_[np.zeros(lag, dtype=bool), short_active[:-lag]],
        hold,
        extremes[hold],
    )
    for name, feature in (
        ("raw_dvol_change_tail", "dvol_change"),
        ("raw_realized_variance_tail", "realized_variance_1h"),
    ):
        control_long, control_short = _raw_tail_signals(
            features,
            feature=feature,
            tail=top["tail"],
            direction_proxy=top["direction_proxy"],
        )
        controls[name] = simulate(
            market,
            dates,
            control_long,
            control_short,
            hold,
            extremes[hold],
        )
    other_proxy = next(proxy for proxy in proxies if proxy != top["direction_proxy"])
    proxy_long, proxy_short, _ = orphan_signals(
        features,
        residual_days=top["residual_days"],
        tail=top["tail"],
        direction_proxy=other_proxy,
    )
    controls["direction_proxy_swap"] = simulate(
        market, dates, proxy_long, proxy_short, hold, extremes[hold]
    )
    for name, stats in controls.items():
        print_stats("CONTROL " + name, stats)

    cost_stress = {
        str(side_bp): simulate(
            market,
            dates,
            long_active,
            short_active,
            hold,
            extremes[hold],
            side_cost=side_bp / 10_000,
        )
        for side_bp in (0, 1, 3, 6)
    }
    for side_bp, stats in cost_stress.items():
        print_stats(f"COST {side_bp}BP_SIDE", stats)

    result = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "fit_reference": [FIT_START, FIT_END],
            "grid_size": 16,
            "rolling_models": "30d/90d OLS fitted only through the preceding hourly updates",
            "dvol_availability": "hourly candle close_time backward-asof only",
            "premium_availability": "five completed 1m rows in the completed 5m decision bar",
            "entry": "next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "source_file_hashes": {path: _file_hash(path) for path in (MARKET, SPOT, DVOL)},
            "oos_opened": False,
        },
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(
            top["prelim_admitted"] and not any(admission(stats) for stats in controls.values())
        ),
    }
    Path("results/dvol_orphan_convexity_alpha_scan_2026-07-13.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
