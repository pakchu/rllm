"""Search a causal distributed-impact-debt alpha before opening 2024+ OOS.

The experiment removes the predictable component of signed taker flow with a
fit-only AR model, estimates a fit-only finite impulse response from flow
innovations to completed-bar returns, and computes the still-unrealized tail of
that response at every completed bar.  A signal is therefore a claim about
remaining impact, not a generic flow threshold.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import (
    _future_extreme,
    _simulate_no_stop,
)
from training.search_positioning_hgb_path_alpha import _read_before

MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
CUTOFF = "2024-01-01"
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
AR_ORDER = 12
KERNEL_HORIZONS = (12, 36)
DEBT_Z_THRESHOLDS = (1.5, 2.0, 2.5, 3.0, 3.5)
EVENT_MODES = ("state", "onset")
RIDGE_PENALTY = 1e-2
NORMALIZER_WINDOW = 8640


def lag_matrix(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Return columns ``values[t-lag]`` for lag 0..max_lag."""
    values = np.asarray(values, dtype=float)
    out = np.full((len(values), int(max_lag) + 1), np.nan, dtype=np.float64)
    for lag in range(int(max_lag) + 1):
        out[lag:, lag] = values[: len(values) - lag]
    return out


def fit_ridge(
    predictors: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray,
    penalty: float = RIDGE_PENALTY,
) -> tuple[np.ndarray, int]:
    """Fit a scaled ridge with an unpenalized intercept on explicit fit rows."""
    predictors = np.asarray(predictors, dtype=float)
    target = np.asarray(target, dtype=float)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    finite = fit_mask & np.isfinite(target) & np.isfinite(predictors).all(axis=1)
    if int(finite.sum()) < 20_000:
        raise ValueError(f"insufficient fit observations: {int(finite.sum())}")
    design = np.column_stack([np.ones(int(finite.sum())), predictors[finite]])
    scales = design[:, 1:].std(axis=0)
    scales[scales < 1e-12] = 1.0
    scaled = design.copy()
    scaled[:, 1:] /= scales
    regularizer = np.eye(scaled.shape[1], dtype=float) * float(penalty)
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        scaled.T @ scaled + regularizer,
        scaled.T @ target[finite],
    )
    coefficients[1:] /= scales
    return coefficients, int(finite.sum())


def prior_z(values: np.ndarray, window: int = NORMALIZER_WINDOW) -> np.ndarray:
    """Standardize the current completed value using history through t-1."""
    series = pd.Series(np.asarray(values, dtype=float))
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=window // 2).mean()
    std = prior.rolling(window, min_periods=window // 2).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).to_numpy(float)


def impact_debt(
    innovations: np.ndarray,
    response_kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute future kernel mass still owed by current and prior innovations."""
    kernel = np.asarray(response_kernel, dtype=float)
    if len(kernel) < 2:
        raise ValueError("response kernel must include current and future lags")
    horizon = len(kernel) - 1
    innovation_lags = lag_matrix(innovations, horizon - 1)
    tail_weights = np.array(
        [kernel[lag + 1 :].sum() for lag in range(horizon)],
        dtype=float,
    )
    debt = np.sum(innovation_lags * tail_weights, axis=1)
    debt[~np.isfinite(innovation_lags).all(axis=1)] = np.nan
    return debt, tail_weights


def load_pre2024() -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(MARKET, "date", CUTOFF)
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(CUTOFF):
        raise RuntimeError("future market rows opened")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("propagator search requires a complete 5-minute grid")
    return market, dates


def build_impact_state(
    market: pd.DataFrame,
    dates: pd.Series,
    kernel_horizon: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    close = pd.to_numeric(market["close"], errors="coerce").to_numpy(float)
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    imbalance = ((2.0 * taker_buy - quote) / quote.replace(0.0, np.nan)).clip(-1.0, 1.0)
    prior_activity = quote.shift(1).rolling(
        NORMALIZER_WINDOW,
        min_periods=NORMALIZER_WINDOW // 2,
    ).median()
    activity_scale = (quote / prior_activity.replace(0.0, np.nan)).clip(0.0, 16.0).pow(0.5)
    signed_flow = (imbalance * activity_scale).clip(-4.0, 4.0).to_numpy(float)
    returns = np.full(len(market), np.nan, dtype=np.float64)
    valid_close = np.isfinite(close) & (close > 0.0)
    valid_pair = valid_close[1:] & valid_close[:-1]
    returns[1:][valid_pair] = np.log(close[1:][valid_pair] / close[:-1][valid_pair])
    fit_mask = (
        (dates >= pd.Timestamp(WINDOWS["fit"][0]))
        & (dates < pd.Timestamp(WINDOWS["fit"][1]))
    ).to_numpy(bool)

    flow_lags = lag_matrix(signed_flow, AR_ORDER)
    ar_coefficients, ar_observations = fit_ridge(
        flow_lags[:, 1:],
        signed_flow,
        fit_mask,
    )
    innovations = signed_flow - (
        ar_coefficients[0]
        + np.sum(flow_lags[:, 1:] * ar_coefficients[1:], axis=1)
    )
    innovations[~np.isfinite(flow_lags[:, 1:]).all(axis=1)] = np.nan

    innovation_lags = lag_matrix(innovations, kernel_horizon)
    response_coefficients, response_observations = fit_ridge(
        innovation_lags,
        returns,
        fit_mask,
    )
    response_kernel = response_coefficients[1:]
    debt, tail_weights = impact_debt(innovations, response_kernel)
    frame = pd.DataFrame(
        {
            "signed_flow": signed_flow,
            "flow_innovation": innovations,
            "impact_debt": debt,
            "impact_debt_z": prior_z(debt),
        },
        index=market.index,
    ).replace([np.inf, -np.inf], np.nan)
    diagnostics = {
        "kernel_horizon": int(kernel_horizon),
        "ar_order": AR_ORDER,
        "ridge_penalty": RIDGE_PENALTY,
        "ar_observations": ar_observations,
        "response_observations": response_observations,
        "ar_coefficients": ar_coefficients.tolist(),
        "response_intercept": float(response_coefficients[0]),
        "response_kernel": response_kernel.tolist(),
        "response_kernel_sum": float(response_kernel.sum()),
        "instantaneous_response": float(response_kernel[0]),
        "remaining_tail_after_current": float(response_kernel[1:].sum()),
        "tail_weights": tail_weights.tolist(),
    }
    return frame, diagnostics


def build_signals(
    debt_z: np.ndarray,
    threshold: float,
    event_mode: str,
    *,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(debt_z, dtype=float)
    state = np.isfinite(values) & (np.abs(values) >= float(threshold))
    if event_mode == "state":
        active = state
    elif event_mode == "onset":
        active = state & ~np.r_[False, state[:-1]]
    else:
        raise KeyError(event_mode)
    side = np.sign(values)
    if flip:
        side = -side
    return active & (side > 0.0), active & (side < 0.0)


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    hold: int,
    extremes: tuple[np.ndarray, np.ndarray],
    side_cost: float = 0.0006,
) -> dict[str, dict[str, Any]]:
    return {
        window: _simulate_no_stop(
            market,
            dates,
            long_active,
            short_active,
            window=window,
            hold_bars=hold,
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
    enough = (
        stats["fit"]["trades"] >= 60
        and stats["select_2023"]["trades"] >= 20
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"])
        >= 6
    )
    return bool(
        enough
        and stats["fit"]["return_pct"] > 0.0
        and stats["fit"]["ratio"] >= 3.0
        and stats["select_2023"]["return_pct"] > 0.0
        and stats["select_2023"]["ratio"] >= 3.0
        and stats["select_2023_h1"]["return_pct"] >= 0.0
        and stats["select_2023_h2"]["return_pct"] >= 0.0
    )


def rank_key(stats: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    enough = (
        stats["fit"]["trades"] >= 60
        and stats["select_2023"]["trades"] >= 20
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"])
        >= 6
    )
    core = [
        stats["fit"]["ratio"],
        stats["select_2023"]["ratio"],
        stats["select_2023_h1"]["ratio"],
        stats["select_2023_h2"]["ratio"],
    ]
    return (
        admission(stats),
        enough,
        min(core) > 0.0,
        sum(stats[window]["return_pct"] > 0.0 for window in SEGMENTS),
        min(core),
        float(np.median(core)),
        stats["select_2023"]["trades"],
    )


def print_stats(title: str, stats: dict[str, dict[str, Any]]) -> None:
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


def main() -> None:
    market, dates = load_pre2024()
    holds = tuple(sorted({horizon for horizon in KERNEL_HORIZONS} | {3 * horizon for horizon in KERNEL_HORIZONS}))
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    rows: list[dict[str, Any]] = []
    feature_bank: dict[int, pd.DataFrame] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for kernel_horizon in KERNEL_HORIZONS:
        features, kernel_diagnostics = build_impact_state(market, dates, kernel_horizon)
        feature_bank[kernel_horizon] = features
        diagnostics[str(kernel_horizon)] = kernel_diagnostics
        for threshold, event_mode, hold in itertools.product(
            DEBT_Z_THRESHOLDS,
            EVENT_MODES,
            (kernel_horizon, 3 * kernel_horizon),
        ):
            long_active, short_active = build_signals(
                features["impact_debt_z"].to_numpy(float),
                threshold,
                event_mode,
            )
            stats = simulate(
                market,
                dates,
                long_active,
                short_active,
                hold,
                extremes[hold],
            )
            rows.append(
                {
                    "kernel_horizon": kernel_horizon,
                    "threshold": threshold,
                    "event_mode": event_mode,
                    "hold": hold,
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
            f"RANK {index} k{row['kernel_horizon']} z{row['threshold']} "
            f"{row['event_mode']} h{row['hold']} sig={row['signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    features = feature_bank[top["kernel_horizon"]]
    debt_z = features["impact_debt_z"].to_numpy(float)
    long_active, short_active = build_signals(
        debt_z,
        top["threshold"],
        top["event_mode"],
    )
    hold = top["hold"]
    flip_long, flip_short = build_signals(
        debt_z,
        top["threshold"],
        top["event_mode"],
        flip=True,
    )
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(
            market, dates, flip_long, flip_short, hold, extremes[hold]
        )
    }
    active = long_active | short_active
    innovation = features["flow_innovation"].to_numpy(float)
    controls["innovation_direction_only"] = simulate(
        market,
        dates,
        active & (innovation > 0.0),
        active & (innovation < 0.0),
        hold,
        extremes[hold],
    )
    lag = 12
    controls["signal_lag_1h"] = simulate(
        market,
        dates,
        np.r_[np.zeros(lag, dtype=bool), long_active[:-lag]],
        np.r_[np.zeros(lag, dtype=bool), short_active[:-lag]],
        hold,
        extremes[hold],
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
            "fit_model": WINDOWS["fit"],
            "selection": WINDOWS["select_2023"],
            "grid_size": len(rows),
            "mechanism": "AR(12) flow innovation -> fit-only FIR response -> causal unrealized kernel-tail debt",
            "normalization": "current debt standardized by prior 30d history",
            "entry": "next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
            "contamination_note": "2023 is internal selection and has been inspected; 2024+ remained sealed",
        },
        "kernel_diagnostics": diagnostics,
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(top["prelim_admitted"]),
    }
    Path("results/propagator_impact_debt_alpha_scan_2026-07-13.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
