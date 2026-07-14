"""Search a hidden FX safe-haven stress alpha for BTC.

The aggregate Dollar Index can look quiet while JPY and CHF strengthen against
the dollar relative to the other DXY currencies.  This experiment treats that
cross-sectional cancellation as latent risk stress.  It trades only when BTC's
completed six-hour response has not absorbed the same signed stress.

All FX bars are aggregated from completed one-minute rows.  An hour stamped
``h`` becomes usable only at ``h + 1h``; BTC enters at the following 5-minute
open.  The returned analysis frame is hard-filtered before 2024.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _read_before


MARKET_PATH = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
FX_PATH = Path(
    "/home/pakchu/workspace/wave_trading/data/"
    "2020-01-01_2025-12-15_b5f4987f5d3bd807042f43a9c44fa871.csv.gz"
)
RESULT_PATH = Path("results/hidden_safe_haven_cancellation_alpha_scan_2026-07-14.json")
CUTOFF = "2024-01-01"

TICKERS = ("EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY", "USDSEK")
SAFE_HAVENS = ("USDJPY", "USDCHF")
CYCLICALS = ("EURUSD", "GBPUSD", "USDCAD", "USDSEK")
USD_ORIENTATION = {
    "EURUSD": -1.0,
    "GBPUSD": -1.0,
    "USDCAD": 1.0,
    "USDCHF": 1.0,
    "USDJPY": 1.0,
    "USDSEK": 1.0,
}

RETURN_HOURS = 6
NORMALIZATION_HOURS = 30 * 24
NORMALIZATION_MIN_OBSERVATIONS = 10 * 24
FIT_TAIL = 0.90
HOLD_BARS = 12 * 12
SIDE_COST = 0.0006

WINDOWS = {
    "fit": ("2020-06-01", "2023-01-01"),
    "fit_2020_h2": ("2020-06-01", "2021-01-01"),
    "fit_2021_h1": ("2021-01-01", "2021-07-01"),
    "fit_2021_h2": ("2021-07-01", "2022-01-01"),
    "fit_2022_h1": ("2022-01-01", "2022-07-01"),
    "fit_2022_h2": ("2022-07-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
}
SEGMENTS = (
    "fit_2020_h2",
    "fit_2021_h1",
    "fit_2021_h2",
    "fit_2022_h1",
    "fit_2022_h2",
    "select_2023_h1",
    "select_2023_h2",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_market_before(path: str | Path = MARKET_PATH, cutoff: str = CUTOFF) -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(str(path), "date", cutoff)
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cutoff):
        raise RuntimeError("market rows at or after the cutoff entered the analysis frame")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("safe-haven search requires a complete 5-minute BTC grid")
    return market, dates


def read_completed_fx_hours_before(
    path: str | Path = FX_PATH,
    cutoff: str = CUTOFF,
    *,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Stream a sorted one-minute FX cache into completed hourly observations."""
    cutoff_ts = pd.Timestamp(cutoff)
    pieces: list[pd.DataFrame] = []
    last_seen: pd.Timestamp | None = None
    usecols = ["date", "tic", "interval", "close"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, compression="infer"):
        dates = pd.to_datetime(chunk["date"], utc=True, errors="raise").dt.tz_convert(None)
        if not dates.is_monotonic_increasing:
            raise ValueError("FX cache must be globally sorted by source timestamp")
        if last_seen is not None and len(dates) and dates.iloc[0] < last_seen:
            raise ValueError("FX cache chunks are not globally time ordered")
        if len(dates):
            last_seen = dates.iloc[-1]
        chunk = chunk.assign(date=dates)
        before = chunk["date"] < cutoff_ts
        selected = chunk.loc[
            before
            & chunk["tic"].astype(str).isin(TICKERS)
            & chunk["interval"].astype(str).eq("1m")
        ].copy()
        if not selected.empty:
            selected["close"] = pd.to_numeric(selected["close"], errors="coerce")
            selected = selected.dropna(subset=["close"])
            selected["source_hour"] = selected["date"].dt.floor("h")
            selected = selected.sort_values(["source_hour", "tic", "date"])
            grouped = selected.groupby(["source_hour", "tic"], sort=True)
            pieces.append(
                grouped.agg(
                    close=("close", "last"),
                    source_rows=("date", "size"),
                    source_time=("date", "max"),
                ).reset_index()
            )
        if len(chunk) and chunk["date"].iloc[-1] >= cutoff_ts:
            break
    if not pieces:
        raise ValueError("no pre-cutoff FX rows found")

    partial = pd.concat(pieces, ignore_index=True).sort_values(
        ["source_hour", "tic", "source_time"]
    )
    combined = partial.groupby(["source_hour", "tic"], sort=True).agg(
        close=("close", "last"),
        source_rows=("source_rows", "sum"),
        source_time=("source_time", "max"),
    )
    close = combined["close"].unstack("tic").reindex(columns=TICKERS)
    rows = combined["source_rows"].unstack("tic").reindex(columns=TICKERS)
    source_time = combined["source_time"].unstack("tic").reindex(columns=TICKERS)
    full_index = pd.date_range(close.index.min(), close.index.max(), freq="h")
    close = close.reindex(full_index)
    rows = rows.reindex(full_index)
    source_time = source_time.reindex(full_index)

    hour_end = pd.Series(full_index + pd.Timedelta("1h"), index=full_index)
    latest_required = pd.DataFrame(
        {ticker: full_index + pd.Timedelta("59min") for ticker in TICKERS},
        index=full_index,
    )
    before_hour_end = source_time.lt(
        pd.DataFrame({ticker: hour_end for ticker in TICKERS}, index=full_index)
    )
    valid = (
        close.notna().all(axis=1)
        & rows.ge(55).all(axis=1)
        & source_time.ge(latest_required).all(axis=1)
        & before_hour_end.all(axis=1)
    )
    output = close.add_prefix("close_")
    output["valid_hour"] = valid
    output["source_rows_min"] = rows.min(axis=1)
    output["source_time"] = source_time.max(axis=1)
    output["effective_time"] = full_index + pd.Timedelta("1h")
    output = output.loc[output["effective_time"] < cutoff_ts].reset_index(drop=True)
    if output["effective_time"].max() >= cutoff_ts:
        raise RuntimeError("post-cutoff FX effective time entered the analysis frame")
    observed = output["source_time"].notna()
    if (output.loc[observed, "source_time"] >= output.loc[observed, "effective_time"]).any():
        raise RuntimeError("FX source timestamp is not strictly earlier than effective time")
    return output


def continuous_log_return(
    values: pd.Series,
    valid: pd.Series,
    *,
    horizon: int = RETURN_HOURS,
) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    valid = pd.Series(valid, index=values.index, dtype=bool)
    continuous = valid.rolling(horizon + 1, min_periods=horizon + 1).sum().eq(horizon + 1)
    result = np.log(values.where(values > 0.0) / values.shift(horizon).where(lambda x: x > 0.0))
    return result.where(continuous)


def prior_zscore(
    values: pd.Series,
    *,
    window: int = NORMALIZATION_HOURS,
    min_observations: int = NORMALIZATION_MIN_OBSERVATIONS,
) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    prior = values.shift(1)
    mean = prior.rolling(window, min_periods=min_observations).mean()
    std = prior.rolling(window, min_periods=min_observations).std(ddof=0).replace(0.0, np.nan)
    return (values - mean) / std


def prior_scaled_horizon_return(
    values: pd.Series,
    valid: pd.Series,
    *,
    horizon: int = RETURN_HOURS,
    window: int = NORMALIZATION_HOURS,
    min_observations: int = NORMALIZATION_MIN_OBSERVATIONS,
) -> tuple[pd.Series, pd.Series]:
    """Scale a continuous horizon return by prior one-hour volatility."""
    one_hour = continuous_log_return(values, valid, horizon=1)
    horizon_return = continuous_log_return(values, valid, horizon=horizon)
    prior = one_hour.shift(1)
    prior_mean = prior.rolling(window, min_periods=min_observations).mean()
    prior_std = prior.rolling(window, min_periods=min_observations).std(ddof=0).replace(0.0, np.nan)
    scaled = (horizon_return - horizon * prior_mean) / (np.sqrt(float(horizon)) * prior_std)
    return horizon_return, scaled


def safe_haven_risk_stress(oriented_usd_returns: pd.DataFrame) -> pd.Series:
    missing = set(TICKERS).difference(oriented_usd_returns.columns)
    if missing:
        raise ValueError(f"missing oriented USD-return columns: {sorted(missing)}")
    safe_mean = oriented_usd_returns.loc[:, SAFE_HAVENS].mean(axis=1)
    cyclical_mean = oriented_usd_returns.loc[:, CYCLICALS].mean(axis=1)
    return -(safe_mean - cyclical_mean)


def build_fx_features(hourly: pd.DataFrame) -> pd.DataFrame:
    valid = hourly["valid_hour"].astype(bool)
    z_returns: dict[str, pd.Series] = {}
    raw_returns: dict[str, pd.Series] = {}
    for ticker in TICKERS:
        raw, scaled = prior_scaled_horizon_return(hourly[f"close_{ticker}"], valid)
        raw_returns[ticker] = raw
        z_returns[ticker] = scaled * USD_ORIENTATION[ticker]
    oriented = pd.DataFrame(z_returns)
    oriented_raw = pd.DataFrame(
        {ticker: raw_returns[ticker] * USD_ORIENTATION[ticker] for ticker in TICKERS}
    )
    out = pd.DataFrame(
        {
            "effective_time": pd.to_datetime(hourly["effective_time"]),
            "fx_source_time": pd.to_datetime(hourly["source_time"]),
            "fx_valid": valid,
            "risk_stress": safe_haven_risk_stress(oriented),
            "raw_risk_stress": safe_haven_risk_stress(oriented_raw),
            "broad_usd_strength": oriented.mean(axis=1),
        }
    )
    complete = oriented.notna().all(axis=1) & valid
    out.loc[~complete, ["risk_stress", "raw_risk_stress", "broad_usd_strength"]] = np.nan
    return out


def build_state(market: pd.DataFrame, dates: pd.Series, fx_hourly: pd.DataFrame) -> pd.DataFrame:
    # At minute-00 the preceding BTC minute-55 bar and FX minute-59 bar have
    # both completed.  Put the signal on that minute-00 bar so the shared
    # simulator enters only at minute-05, strictly after the information edge.
    decision_positions = np.flatnonzero(dates.dt.minute.eq(0).to_numpy(bool))
    decision_positions = decision_positions[decision_positions > 0]
    decision_time = dates.iloc[decision_positions].reset_index(drop=True)
    btc_close = (
        pd.to_numeric(market["close"], errors="coerce")
        .iloc[decision_positions - 1]
        .reset_index(drop=True)
    )
    btc_valid = pd.Series(True, index=btc_close.index)
    _, btc_return_z = prior_scaled_horizon_return(btc_close, btc_valid)
    decisions = pd.DataFrame(
        {
            "position": decision_positions,
            "effective_time": decision_time,
            "btc_return_z": btc_return_z,
        }
    )
    merged = decisions.merge(build_fx_features(fx_hourly), on="effective_time", how="left", validate="one_to_one")
    source_valid = merged["fx_source_time"].notna()
    if (
        merged.loc[source_valid, "fx_source_time"]
        > merged.loc[source_valid, "effective_time"] - pd.Timedelta("1min")
    ).any():
        raise RuntimeError("FX source is not complete before the BTC decision time")
    merged["unpriced_stress"] = merged["risk_stress"] + merged["btc_return_z"]
    same_sign = np.sign(merged["risk_stress"]) == np.sign(merged["unpriced_stress"])
    finite = np.isfinite(merged["risk_stress"]) & np.isfinite(merged["unpriced_stress"])
    merged["eligible"] = finite & same_sign & merged["risk_stress"].ne(0.0)
    # The residual is already in common volatility units.  Using its absolute
    # value avoids double-loading risk_stress through a multiplicative score.
    merged["cancellation_score"] = merged["unpriced_stress"].abs().where(merged["eligible"])

    state = pd.DataFrame(
        {
            "decision": np.zeros(len(market), dtype=bool),
            "fx_source_time": pd.Series(pd.NaT, index=np.arange(len(market)), dtype="datetime64[ns]"),
            "btc_return_z": np.full(len(market), np.nan),
            "risk_stress": np.full(len(market), np.nan),
            "raw_risk_stress": np.full(len(market), np.nan),
            "broad_usd_strength": np.full(len(market), np.nan),
            "unpriced_stress": np.full(len(market), np.nan),
            "eligible": np.zeros(len(market), dtype=bool),
            "cancellation_score": np.full(len(market), np.nan),
        }
    )
    positions = merged["position"].to_numpy(np.int64)
    state.loc[positions, "decision"] = True
    for column in (
        "fx_source_time",
        "btc_return_z",
        "risk_stress",
        "raw_risk_stress",
        "broad_usd_strength",
        "unpriced_stress",
        "eligible",
        "cancellation_score",
    ):
        state.loc[positions, column] = merged[column].to_numpy()
    return state


def fit_threshold(state: pd.DataFrame, dates: pd.Series, column: str, quantile: float = FIT_TAIL) -> float:
    start, end = WINDOWS["fit"]
    fit = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    values = pd.to_numeric(state[column], errors="coerce").to_numpy(float)
    reference = values[fit & np.isfinite(values)]
    if len(reference) < 1_000:
        raise ValueError(f"insufficient fit observations for {column}: {len(reference)}")
    return float(np.quantile(np.abs(reference), quantile))


def signed_tail_masks(values: np.ndarray, decision: np.ndarray, threshold: float, *, fade: bool) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    active = np.asarray(decision, dtype=bool) & np.isfinite(values) & (np.abs(values) >= threshold)
    side = -np.sign(values) if fade else np.sign(values)
    return active & (side > 0.0), active & (side < 0.0)


def policy_masks(state: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    active = state["eligible"].to_numpy(bool) & (
        state["cancellation_score"].to_numpy(float) >= threshold
    )
    unpriced = state["unpriced_stress"].to_numpy(float)
    return active & (unpriced < 0.0), active & (unpriced > 0.0)


def lag_mask(values: np.ndarray, bars: int) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    if bars <= 0:
        raise ValueError("lag must be positive")
    return np.r_[np.zeros(bars, dtype=bool), values[:-bars]]


def support_counts(
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    window: str,
) -> dict[str, int]:
    start, end = WINDOWS[window]
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    long_active = np.asarray(long_active, dtype=bool)
    short_active = np.asarray(short_active, dtype=bool)
    active = long_active | short_active
    raw_positions = np.flatnonzero(period & active)
    executable: list[int] = []
    next_position = 0
    for position in raw_positions:
        entry = int(position) + 1
        exit_position = entry + HOLD_BARS
        if position < next_position or exit_position >= len(dates) or not period[exit_position]:
            continue
        if long_active[position] == short_active[position]:
            continue
        executable.append(int(position))
        next_position = exit_position + 1
    executable_array = np.asarray(executable, dtype=np.int64)
    return {
        "raw": int(len(raw_positions)),
        "raw_long": int((period & long_active & ~short_active).sum()),
        "raw_short": int((period & short_active & ~long_active).sum()),
        "strict_executable": int(len(executable)),
        "strict_executable_long": int(long_active[executable_array].sum()) if len(executable) else 0,
        "strict_executable_short": int(short_active[executable_array].sum()) if len(executable) else 0,
    }


def support_passes(support: dict[str, dict[str, int]]) -> bool:
    fit = support["fit"]
    select = support["select_2023"]
    h1 = support["select_2023_h1"]
    h2 = support["select_2023_h2"]
    return bool(
        fit["strict_executable"] >= 80
        and select["strict_executable"] >= 24
        and min(h1["strict_executable"], h2["strict_executable"]) >= 8
        and min(fit["strict_executable_long"], fit["strict_executable_short"]) >= 15
        and min(select["strict_executable_long"], select["strict_executable_short"]) >= 4
    )


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    extremes: tuple[np.ndarray, np.ndarray],
    *,
    side_cost: float = SIDE_COST,
) -> dict[str, dict[str, Any]]:
    return {
        window: _simulate_no_stop(
            market,
            dates,
            long_active,
            short_active,
            window=window,
            hold_bars=HOLD_BARS,
            stride_bars=1,
            leverage=0.5,
            fee_rate=side_cost,
            slippage_rate=0.0,
            extremes=extremes,
            windows=WINDOWS,
        )
        for window in WINDOWS
    }


def admission(stats: dict[str, dict[str, Any]]) -> bool:
    return bool(
        stats["fit"]["trades"] >= 80
        and stats["select_2023"]["trades"] >= 24
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 8
        and min(stats["fit"]["longs"], stats["fit"]["shorts"]) >= 15
        and min(stats["select_2023"]["longs"], stats["select_2023"]["shorts"]) >= 4
        and stats["fit"]["return_pct"] > 0.0
        and stats["fit"]["ratio"] >= 3.0
        and stats["select_2023"]["return_pct"] > 0.0
        and stats["select_2023"]["ratio"] >= 3.0
        and stats["select_2023_h1"]["return_pct"] > 0.0
        and stats["select_2023_h2"]["return_pct"] > 0.0
    )


def event_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = int((left | right).sum())
    return float((left & right).sum() / union) if union else 0.0


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    return float(pd.Series(left[valid]).corr(pd.Series(right[valid]), method="spearman")) if valid.sum() >= 100 else float("nan")


def _control_masks(state: pd.DataFrame, dates: pd.Series) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    decision = state["decision"].to_numpy(bool)
    controls: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, column, fade in (
        ("fx_safe_haven_only", "risk_stress", True),
        ("btc_reversal_only", "btc_return_z", True),
        ("broad_usd_only", "broad_usd_strength", True),
        ("raw_unstandardized_safe_contrast", "raw_risk_stress", True),
    ):
        threshold = fit_threshold(state, dates, column)
        controls[name] = signed_tail_masks(
            state[column].to_numpy(float), decision, threshold, fade=fade
        )
    return controls


def _print_stats(title: str, stats: dict[str, dict[str, Any]]) -> None:
    print("\n" + title)
    for window in ("fit", "select_2023", *SEGMENTS):
        value = stats[window]
        print(
            window,
            f"ret={value['return_pct']:.2f}",
            f"cagr={value['cagr_pct']:.2f}",
            f"mdd={value['strict_mdd_pct']:.2f}",
            f"ratio={value['ratio']:.2f}",
            f"n={value['trades']}",
            f"L/S={value['longs']}/{value['shorts']}",
        )


def run(
    *,
    market_path: str | Path = MARKET_PATH,
    fx_path: str | Path = FX_PATH,
    support_only: bool = False,
) -> dict[str, Any]:
    market, dates = load_market_before(market_path)
    fx_hourly = read_completed_fx_hours_before(fx_path)
    state = build_state(market, dates, fx_hourly)
    threshold = fit_threshold(state, dates, "cancellation_score")
    primary_long, primary_short = policy_masks(state, threshold)
    support = {
        window: support_counts(dates, primary_long, primary_short, window=window)
        for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")
    }
    preflight = {
        "support_only": True,
        "threshold": threshold,
        "support": support,
        "support_passed": support_passes(support),
        "valid_fx_hours": int(fx_hourly["valid_hour"].sum()),
        "eligible_events": int(state["eligible"].sum()),
        "source_latest": str(pd.to_datetime(fx_hourly["source_time"]).max()),
    }
    if support_only:
        return preflight

    extremes = (
        _future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"),
        _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"),
    )
    primary_stats = simulate(market, dates, primary_long, primary_short, extremes)
    controls_masks = _control_masks(state, dates)
    controls_masks["exact_direction_flip"] = (primary_short.copy(), primary_long.copy())
    for name, bars in (("signal_delay_1h", 12), ("signal_delay_24h", 288), ("signal_delay_7d", 2016)):
        controls_masks[name] = (
            lag_mask(primary_long, bars),
            lag_mask(primary_short, bars),
        )
    controls = {
        name: simulate(market, dates, long_active, short_active, extremes)
        for name, (long_active, short_active) in controls_masks.items()
    }
    cost_stress = {
        str(bp): simulate(
            market,
            dates,
            primary_long,
            primary_short,
            extremes,
            side_cost=bp / 10_000.0,
        )
        for bp in (0, 1, 3, 6, 10, 15)
    }
    primary_events = primary_long | primary_short
    overlap = {
        name: event_jaccard(primary_events, long_active | short_active)
        for name, (long_active, short_active) in controls_masks.items()
        if name in {
            "fx_safe_haven_only",
            "btc_reversal_only",
            "broad_usd_only",
            "raw_unstandardized_safe_contrast",
        }
    }
    feature_spearman = {
        name: finite_spearman(
            state["cancellation_score"].to_numpy(float), state[column].to_numpy(float)
        )
        for name, column in {
            "abs_fx_safe_haven": "risk_stress",
            "abs_btc_response": "btc_return_z",
            "broad_usd": "broad_usd_strength",
            "raw_safe_contrast": "raw_risk_stress",
        }.items()
    }
    # Absolute components are the relevant score controls for the first two.
    feature_spearman["abs_fx_safe_haven"] = finite_spearman(
        state["cancellation_score"].to_numpy(float),
        np.abs(state["risk_stress"].to_numpy(float)),
    )
    feature_spearman["abs_btc_response"] = finite_spearman(
        state["cancellation_score"].to_numpy(float),
        np.abs(state["btc_return_z"].to_numpy(float)),
    )
    novelty_pass = bool(max(overlap.values(), default=1.0) < 0.50)
    control_admissions = {name: admission(stats) for name, stats in controls.items()}
    output = {
        "protocol": {
            "source_cutoff": "returned market and FX analysis frames hard-filtered strictly before 2024-01-01",
            "source_io_disclosure": "chunk parsers may read and immediately discard later rows in a cutoff-crossing chunk; no discarded row enters returned frames or computation",
            "mechanism": "cross-sectional JPY/CHF safe-haven strength orthogonal to four other USD pairs, multiplied by same-signed BTC non-acceptance",
            "fx_hour_completion": "one-minute source hour requires all six pairs, >=55 rows each and a minute-59 observation; effective only at next hour",
            "return_horizon_hours": RETURN_HOURS,
            "normalization": "6h log return minus 6x prior 1h mean, divided by sqrt(6) times prior 30d 1h std; >=240 valid prior observations",
            "fit_tail": FIT_TAIL,
            "entry": "FX minute-59 and BTC minute-55 complete at minute-00; signal sits on minute-00 and enters minute-05 open",
            "hold_bars": HOLD_BARS,
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "support_only_preflight": {"performed_before_returns": True, **preflight},
            "oos_opened": False,
            "contamination_note": "pre-2024 exploratory mechanism; 2023 is inspected internal selection and 2024+ remains excluded",
        },
        "source": {
            "market_path": str(market_path),
            "market_sha256": _sha256(market_path),
            "fx_path": str(fx_path),
            "fx_sha256": _sha256(fx_path),
        },
        "state_summary": {
            "valid_fx_hours": int(fx_hourly["valid_hour"].sum()),
            "valid_cancellation_scores": int(np.isfinite(state["cancellation_score"]).sum()),
            "eligible_events": int(state["eligible"].sum()),
            "primary_raw_events": int(primary_events.sum()),
            "primary_raw_long_short": [int(primary_long.sum()), int(primary_short.sum())],
        },
        "primary": {
            "threshold": threshold,
            "stats": primary_stats,
            "prelim_admitted": admission(primary_stats),
        },
        "controls": controls,
        "control_admissions": control_admissions,
        "cost_stress": cost_stress,
        "novelty_overlap_audit": {
            "event_jaccard": overlap,
            "feature_spearman": feature_spearman,
            "max_event_jaccard": max(overlap.values(), default=1.0),
            "novelty_pass": novelty_pass,
            "gate": "all fixed component-control event Jaccards below 0.50",
        },
        "prelim_admitted": admission(primary_stats),
        "final_admitted": bool(
            admission(primary_stats)
            and novelty_pass
            and not any(control_admissions.values())
        ),
        "oos_opened": False,
    }
    _print_stats("PRIMARY hidden safe-haven cancellation", primary_stats)
    for name, stats in controls.items():
        _print_stats("CONTROL " + name, stats)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-path", default=str(MARKET_PATH))
    parser.add_argument("--fx-path", default=str(FX_PATH))
    parser.add_argument("--support-only", action="store_true")
    args = parser.parse_args()
    output = run(
        market_path=args.market_path,
        fx_path=args.fx_path,
        support_only=args.support_only,
    )
    if args.support_only:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
