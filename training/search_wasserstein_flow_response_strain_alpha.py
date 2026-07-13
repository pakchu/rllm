"""Search a causal Wasserstein flow-response strain alpha.

The state compares the empirical price-response distributions conditional on
aggressive buy and sell flow.  Exact one-dimensional monotone transport
quantiles measure whether one side currently moves price more efficiently than
the other.  No historical analogue labels or learned model are used.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _read_before

MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
CUTOFF = "2024-01-01"
WINDOWS = {
    "fit": ("2020-10-15", "2023-01-01"),
    "fit_2020_h2": ("2020-10-15", "2021-01-01"),
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
LOOKBACKS = (288, 2016)
SCORE_QUANTILES = (0.80, 0.90)
HOLDS = (72, 144)
TRANSPORT_QUANTILES = np.arange(0.1, 1.0, 0.1)
FLOW_TAIL_QUANTILE = 0.70
VOL_WINDOW = 2016
VOL_MIN_PERIODS = 1008
DECISION_MINUTE = 55
DECISION_STRIDE = 12
MIN_SIDE_OBSERVATIONS = 40


def load_pre2024() -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(MARKET, "date", CUTOFF)
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(CUTOFF):
        raise RuntimeError("future rows opened")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("WFRS requires a complete 5-minute grid")
    return market, dates


def window_mask(dates: pd.Series, name: str) -> np.ndarray:
    start, end = WINDOWS[name]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def build_response_inputs(
    market: pd.DataFrame,
    dates: pd.Series,
) -> tuple[pd.DataFrame, float]:
    open_price = pd.to_numeric(market["open"], errors="coerce").where(lambda value: value > 0.0)
    close_price = pd.to_numeric(market["close"], errors="coerce").where(lambda value: value > 0.0)
    bar_return = np.log(close_price / open_price)
    prior_volatility = (
        bar_return.shift(1)
        .rolling(VOL_WINDOW, min_periods=VOL_MIN_PERIODS)
        .std(ddof=0)
        .replace(0.0, np.nan)
    )
    response = (bar_return / prior_volatility).clip(-5.0, 5.0)
    quote_volume = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy_quote = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    flow = (2.0 * taker_buy_quote / quote_volume.replace(0.0, np.nan) - 1.0).clip(-1.0, 1.0)
    fit = window_mask(dates, "fit")
    fit_flow = np.abs(flow.to_numpy(float)[fit])
    fit_flow = fit_flow[np.isfinite(fit_flow)]
    if len(fit_flow) == 0:
        raise ValueError("no finite fit flow for tail threshold")
    flow_tail = float(np.quantile(fit_flow, FLOW_TAIL_QUANTILE))
    return pd.DataFrame(
        {
            "bar_return": bar_return,
            "prior_volatility": prior_volatility,
            "response": response,
            "flow": flow,
            "decision": dates.dt.minute.eq(DECISION_MINUTE),
        }
    ), flow_tail


def transport_components(plus: np.ndarray, minus: np.ndarray) -> dict[str, float]:
    """Exact empirical 1D monotone-transport summaries on a fixed quantile grid."""
    buy_response = np.asarray(plus, dtype=float)
    sell_response = np.asarray(minus, dtype=float)
    if min(len(buy_response), len(sell_response)) < MIN_SIDE_OBSERVATIONS:
        raise ValueError("insufficient side observations")
    if not np.isfinite(buy_response).all() or not np.isfinite(sell_response).all():
        raise ValueError("non-finite response sample")
    displacement = np.quantile(buy_response, TRANSPORT_QUANTILES) - np.quantile(
        sell_response, TRANSPORT_QUANTILES
    )
    location = float(displacement[1:8].mean())
    shape = float(displacement[6:9].mean() - displacement[:3].mean())
    score = location + 0.5 * shape
    mean_gap = float(buy_response.mean() - sell_response.mean())
    std_gap = float(buy_response.std(ddof=0) - sell_response.std(ddof=0))
    iqr_gap = float(
        (np.quantile(buy_response, 0.75) - np.quantile(buy_response, 0.25))
        - (np.quantile(sell_response, 0.75) - np.quantile(sell_response, 0.25))
    )
    moment_magnitude = float(np.sqrt(mean_gap**2 + std_gap**2 + iqr_gap**2))
    return {
        "score": score,
        "w1": float(np.abs(displacement).mean()),
        "location": location,
        "shape": shape,
        "mean_only": mean_gap,
        "moment_state": float(np.sign(mean_gap) * moment_magnitude),
    }


def build_transport_state(
    inputs: pd.DataFrame,
    *,
    lookback: int,
    flow_tail: float,
) -> pd.DataFrame:
    response = inputs["response"].to_numpy(float)
    flow = inputs["flow"].to_numpy(float)
    decision = inputs["decision"].to_numpy(bool)
    fields = ("score", "w1", "location", "shape", "mean_only", "moment_state")
    output = {field: np.full(len(inputs), np.nan, dtype=float) for field in fields}
    buy_count = np.zeros(len(inputs), dtype=np.int32)
    sell_count = np.zeros(len(inputs), dtype=np.int32)
    for position in np.flatnonzero(decision):
        start = position - int(lookback) + 1
        if start < 0:
            continue
        window_response = response[start : position + 1]
        window_flow = flow[start : position + 1]
        finite = np.isfinite(window_response) & np.isfinite(window_flow)
        buy_response = window_response[finite & (window_flow >= flow_tail)]
        sell_response = -window_response[finite & (window_flow <= -flow_tail)]
        buy_count[position] = len(buy_response)
        sell_count[position] = len(sell_response)
        if min(len(buy_response), len(sell_response)) < MIN_SIDE_OBSERVATIONS:
            continue
        values = transport_components(buy_response, sell_response)
        for field in fields:
            output[field][position] = values[field]
    output["buy_count"] = buy_count
    output["sell_count"] = sell_count
    output["decision"] = decision
    return pd.DataFrame(output)


def fit_score_threshold(
    score: np.ndarray,
    dates: pd.Series,
    quantile: float,
) -> float:
    fit = window_mask(dates, "fit")
    values = np.abs(np.asarray(score, dtype=float)[fit])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("no finite fit scores")
    return float(np.quantile(values, quantile))


def policy_masks(
    score: np.ndarray,
    decision: np.ndarray,
    threshold: float,
    *,
    flip: bool = False,
    onset_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(score, dtype=float)
    active = np.asarray(decision, dtype=bool) & np.isfinite(values) & (np.abs(values) >= threshold)
    side = np.nan_to_num(np.sign(values)).astype(np.int8)
    if onset_only:
        previous = np.r_[np.zeros(DECISION_STRIDE), side[:-DECISION_STRIDE]]
        previous_active = np.r_[np.zeros(DECISION_STRIDE, dtype=bool), active[:-DECISION_STRIDE]]
        active &= (~previous_active) | (previous != side)
    if flip:
        side = -side
    return active & (side > 0), active & (side < 0)


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    hold: int,
    extremes: tuple[np.ndarray, np.ndarray],
    *,
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
        stats["fit"]["trades"] >= 80
        and stats["select_2023"]["trades"] >= 48
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 12
        and min(stats["select_2023"]["longs"], stats["select_2023"]["shorts"]) >= 10
    )
    return bool(
        enough
        and stats["fit"]["return_pct"] > 0.0
        and stats["fit"]["ratio"] >= 3.0
        and stats["select_2023"]["return_pct"] > 0.0
        and stats["select_2023"]["ratio"] >= 3.0
        and stats["select_2023_h1"]["return_pct"] > 0.0
        and stats["select_2023_h2"]["return_pct"] > 0.0
    )


def rank_key(stats: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    enough = (
        stats["fit"]["trades"] >= 80
        and stats["select_2023"]["trades"] >= 48
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 12
    )
    core = [stats[name]["ratio"] for name in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")]
    return (
        admission(stats),
        enough,
        min(core) > 0.0,
        sum(stats[name]["return_pct"] > 0.0 for name in SEGMENTS),
        min(core),
        float(np.median(core)),
        stats["select_2023"]["trades"],
    )


def lag_mask(values: np.ndarray, bars: int) -> np.ndarray:
    return np.r_[np.zeros(bars, dtype=bool), np.asarray(values, dtype=bool)[:-bars]]


def compact_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return stats


def print_stats(title: str, stats: dict[str, dict[str, Any]]) -> None:
    print("\n" + title)
    for name in ("fit", "select_2023", *SEGMENTS):
        value = stats[name]
        print(
            name,
            f"ret={value['return_pct']:.2f}",
            f"cagr={value['cagr_pct']:.2f}",
            f"mdd={value['strict_mdd_pct']:.2f}",
            f"ratio={value['ratio']:.2f}",
            f"n={value['trades']}",
            f"L/S={value['longs']}/{value['shorts']}",
        )


def main() -> None:
    market, dates = load_pre2024()
    inputs, flow_tail = build_response_inputs(market, dates)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in HOLDS
    }
    states = {
        lookback: build_transport_state(inputs, lookback=lookback, flow_tail=flow_tail)
        for lookback in LOOKBACKS
    }
    rows: list[dict[str, Any]] = []
    for lookback, quantile, hold in itertools.product(LOOKBACKS, SCORE_QUANTILES, HOLDS):
        state = states[lookback]
        threshold = fit_score_threshold(state["score"].to_numpy(float), dates, quantile)
        long_active, short_active = policy_masks(
            state["score"].to_numpy(float), state["decision"].to_numpy(bool), threshold
        )
        stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
        rows.append(
            {
                "lookback": lookback,
                "score_quantile": quantile,
                "hold": hold,
                "flow_tail": flow_tail,
                "score_threshold": threshold,
                "raw_signals": int((long_active | short_active).sum()),
                "rank": rank_key(stats),
                "prelim_admitted": admission(stats),
                "stats": compact_stats(stats),
            }
        )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "admitted", sum(row["prelim_admitted"] for row in rows))
    for index, row in enumerate(rows, 1):
        print_stats(
            f"RANK {index} L{row['lookback']} q{row['score_quantile']} h{row['hold']} "
            f"raw={row['raw_signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    state = states[top["lookback"]]
    decision = state["decision"].to_numpy(bool)
    score = state["score"].to_numpy(float)
    threshold = top["score_threshold"]
    long_active, short_active = policy_masks(score, decision, threshold)
    flip_long, flip_short = policy_masks(score, decision, threshold, flip=True)
    onset_long, onset_short = policy_masks(score, decision, threshold, onset_only=True)
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(market, dates, flip_long, flip_short, top["hold"], extremes[top["hold"]]),
        "onset_only": simulate(market, dates, onset_long, onset_short, top["hold"], extremes[top["hold"]]),
    }
    for field in ("mean_only", "moment_state"):
        control_score = state[field].to_numpy(float)
        control_threshold = fit_score_threshold(control_score, dates, top["score_quantile"])
        control_long, control_short = policy_masks(control_score, decision, control_threshold)
        controls[field] = simulate(
            market, dates, control_long, control_short, top["hold"], extremes[top["hold"]]
        )
    signed_w1 = np.sign(score) * state["w1"].to_numpy(float)
    w1_threshold = fit_score_threshold(signed_w1, dates, top["score_quantile"])
    w1_long, w1_short = policy_masks(signed_w1, decision, w1_threshold)
    controls["signed_w1_no_transport_shape"] = simulate(
        market, dates, w1_long, w1_short, top["hold"], extremes[top["hold"]]
    )
    for hours in (1, 6, 24 * 7):
        bars = hours * DECISION_STRIDE
        controls[f"signal_lag_{hours}h"] = simulate(
            market,
            dates,
            lag_mask(long_active, bars),
            lag_mask(short_active, bars),
            top["hold"],
            extremes[top["hold"]],
        )
    for name, stats in controls.items():
        print_stats("CONTROL " + name, stats)

    cost_stress = {
        str(side_bp): simulate(
            market,
            dates,
            long_active,
            short_active,
            top["hold"],
            extremes[top["hold"]],
            side_cost=side_bp / 10_000,
        )
        for side_bp in (0, 1, 3, 6, 10, 15)
    }
    for side_bp, stats in cost_stress.items():
        print_stats("COST " + side_bp + "BP_SIDE", stats)

    feature_hash = hashlib.sha256(
        np.nan_to_num(
            np.column_stack([states[lookback]["score"].to_numpy(float) for lookback in LOOKBACKS]),
            nan=3.4028235e38,
        ).astype("<f4").tobytes(order="C")
    ).hexdigest()
    output = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "mechanism": "exact 1D monotone transport between side-aligned buy-flow and sell-flow return-response distributions",
            "final_grid_size": len(rows),
            "final_grid": "2 lookbacks x 2 fit-only score tails x 2 holds",
            "design_history": "two earlier pre-2024 researcher probes used within-window flow terciles with W1 coherence and a trailing-median innovation variant; both were weak. Their 16 policies plus this final 8-policy WFRS family are contaminated and frozen.",
            "flow_tail": "q70 absolute taker imbalance fit only",
            "response_scale": "current completed bar log(close/open) divided by prior-only 7d bar-return volatility and clipped to +/-5",
            "decision": "open-time timestamp minute 55 completed at next hour, then enter following minute-00 open",
            "entry": "next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
            "contamination_note": "2023 is inspected internal selection; 2024+ remained sealed",
        },
        "flow_tail": flow_tail,
        "feature_hash": feature_hash,
        "state_summary": {
            str(lookback): {
                "valid_decisions": int(np.isfinite(state["score"]).sum()),
                "median_buy_observations": float(state.loc[np.isfinite(state["score"]), "buy_count"].median()),
                "median_sell_observations": float(state.loc[np.isfinite(state["score"]), "sell_count"].median()),
            }
            for lookback, state in states.items()
        },
        "rows": rows,
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(top["prelim_admitted"]),
    }
    path = Path("results/wasserstein_flow_response_strain_alpha_scan_2026-07-14.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
