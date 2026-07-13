"""Search a causal aggressor execution-centroid inversion alpha.

Completed Binance klines report taker-buy volume in both base and quote units.
Their ratio is the average execution price of buyer-initiated trades; subtracting
those quantities from total volume reconstructs the seller-initiated execution
centroid.  The primary event is a threshold-free ordering anomaly:

* long: buyer centroid < seller centroid < completed-hour close;
* short: completed-hour close < buyer centroid < seller centroid.

In both cases the ultimately winning aggressor side transacted at a better
average price than the losing side and the close settled beyond both centroids.
The rule enters only at the next 5-minute open and holds for a fixed 12 hours.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_conformal_sr_pressure_alpha import (
    WINDOWS,
    admission,
    event_jaccard,
    finite_spearman,
    lag_boolean,
    load_pre2024,
    print_stats,
    rank_key,
)
from training.search_positioning_disagreement_alpha import (
    _future_extreme,
    _simulate_no_stop,
)

HOUR_BARS = 12
DECISION_MINUTE = 55
HOLD_BARS = 12 * 12
SIDE_COST = 0.0006
RESULT_PATH = Path("results/aggressor_centroid_inversion_alpha_scan_2026-07-14.json")


def _rolling_sum(market: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(market[column], errors="coerce").rolling(
        HOUR_BARS, min_periods=HOUR_BARS
    ).sum()


def build_centroid_state(market: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Reconstruct completed-hour buyer and seller execution centroids."""
    required = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
    }
    missing = sorted(required.difference(market.columns))
    if missing:
        raise KeyError(f"missing centroid inputs: {missing}")
    if len(market) != len(dates):
        raise ValueError("market and dates must align")

    decision_positions = np.flatnonzero(
        dates.dt.minute.eq(DECISION_MINUTE).to_numpy(bool)
    )
    total_base = _rolling_sum(market, "volume")
    total_quote = _rolling_sum(market, "quote_asset_volume")
    buy_base = _rolling_sum(market, "taker_buy_base")
    buy_quote = _rolling_sum(market, "taker_buy_quote")
    sell_base = (total_base - buy_base).clip(lower=0.0)
    sell_quote = (total_quote - buy_quote).clip(lower=0.0)

    buy_centroid = buy_quote / buy_base.where(buy_base > 0.0)
    sell_centroid = sell_quote / sell_base.where(sell_base > 0.0)
    market_vwap = total_quote / total_base.where(total_base > 0.0)
    hourly_open = pd.to_numeric(market["open"], errors="coerce").shift(HOUR_BARS - 1)
    hourly_close = pd.to_numeric(market["close"], errors="coerce")
    hourly_low = pd.to_numeric(market["low"], errors="coerce").rolling(
        HOUR_BARS, min_periods=HOUR_BARS
    ).min()
    hourly_high = pd.to_numeric(market["high"], errors="coerce").rolling(
        HOUR_BARS, min_periods=HOUR_BARS
    ).max()
    taker_imbalance = (2.0 * buy_quote - total_quote) / total_quote.where(
        total_quote > 0.0
    )

    state = pd.DataFrame(
        {
            "decision": np.zeros(len(market), dtype=bool),
            "total_base": np.full(len(market), np.nan),
            "total_quote": np.full(len(market), np.nan),
            "buy_base": np.full(len(market), np.nan),
            "buy_quote": np.full(len(market), np.nan),
            "sell_base": np.full(len(market), np.nan),
            "sell_quote": np.full(len(market), np.nan),
            "buy_centroid": np.full(len(market), np.nan),
            "sell_centroid": np.full(len(market), np.nan),
            "market_vwap": np.full(len(market), np.nan),
            "hourly_open": np.full(len(market), np.nan),
            "hourly_close": np.full(len(market), np.nan),
            "hourly_low": np.full(len(market), np.nan),
            "hourly_high": np.full(len(market), np.nan),
            "taker_imbalance": np.full(len(market), np.nan),
        }
    )
    state.loc[decision_positions, "decision"] = True
    sources = {
        "total_base": total_base,
        "total_quote": total_quote,
        "buy_base": buy_base,
        "buy_quote": buy_quote,
        "sell_base": sell_base,
        "sell_quote": sell_quote,
        "buy_centroid": buy_centroid,
        "sell_centroid": sell_centroid,
        "market_vwap": market_vwap,
        "hourly_open": hourly_open,
        "hourly_close": hourly_close,
        "hourly_low": hourly_low,
        "hourly_high": hourly_high,
        "taker_imbalance": taker_imbalance,
    }
    for name, values in sources.items():
        state.loc[decision_positions, name] = values.iloc[
            decision_positions
        ].to_numpy(float)

    buy = state["buy_centroid"].to_numpy(float)
    sell = state["sell_centroid"].to_numpy(float)
    close = state["hourly_close"].to_numpy(float)
    midpoint = np.sqrt(buy * sell)
    inversion = np.log(sell / buy)
    settlement = np.log(close / midpoint)
    primary_long = (buy < sell) & (sell < close)
    primary_short = (close < buy) & (buy < sell)
    state["centroid_log_wedge"] = inversion
    state["close_midpoint_log_gap"] = settlement
    state["signed_inversion_settlement"] = np.where(
        primary_long,
        np.minimum(inversion, np.log(close / sell)),
        np.where(primary_short, -np.minimum(inversion, np.log(buy / close)), 0.0),
    )
    return state.replace([np.inf, -np.inf], np.nan)


def validate_centroid_accounting(state: pd.DataFrame) -> dict[str, Any]:
    """Check exact volume accounting and physically possible centroid bounds."""
    decision = state["decision"].to_numpy(bool)
    total_base = state["total_base"].to_numpy(float)
    total_quote = state["total_quote"].to_numpy(float)
    buy_base = state["buy_base"].to_numpy(float)
    sell_base = state["sell_base"].to_numpy(float)
    buy = state["buy_centroid"].to_numpy(float)
    sell = state["sell_centroid"].to_numpy(float)
    low = state["hourly_low"].to_numpy(float)
    high = state["hourly_high"].to_numpy(float)
    valid = (
        decision
        & np.isfinite(total_base)
        & (total_base > 0.0)
        & np.isfinite(total_quote)
        & np.isfinite(buy)
        & np.isfinite(sell)
    )
    reconstructed_quote = buy_base * buy + sell_base * sell
    relative_error = np.abs(reconstructed_quote - total_quote) / np.maximum(
        np.abs(total_quote), 1.0
    )
    tolerance = 1e-10
    bound_tolerance = 1e-8
    bound_violation = valid & (
        (buy < low * (1.0 - bound_tolerance))
        | (buy > high * (1.0 + bound_tolerance))
        | (sell < low * (1.0 - bound_tolerance))
        | (sell > high * (1.0 + bound_tolerance))
    )
    max_error = float(np.nanmax(relative_error[valid])) if valid.any() else float("nan")
    if valid.any() and max_error > tolerance:
        raise ValueError(f"centroid quote reconstruction error {max_error:.3e}")
    if bound_violation.any():
        raise ValueError(f"centroid outside completed-hour range: {bound_violation.sum()}")
    return {
        "decision_hours": int(decision.sum()),
        "valid_centroid_hours": int(valid.sum()),
        "invalid_centroid_hours": int(decision.sum() - valid.sum()),
        "max_quote_reconstruction_relative_error": max_error,
        "range_bound_violations": int(bound_violation.sum()),
    }


def topology_masks(
    state: pd.DataFrame,
    variant: str = "inverted_terminal",
    *,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    buy = state["buy_centroid"].to_numpy(float)
    sell = state["sell_centroid"].to_numpy(float)
    close = state["hourly_close"].to_numpy(float)
    decision = state["decision"].to_numpy(bool)
    finite = decision & np.isfinite(buy) & np.isfinite(sell) & np.isfinite(close)

    if variant == "inverted_terminal":
        long_active = finite & (buy < sell) & (sell < close)
        short_active = finite & (close < buy) & (buy < sell)
    elif variant == "ordinary_terminal":
        long_active = finite & (sell < buy) & (buy < close)
        short_active = finite & (close < sell) & (sell < buy)
    elif variant == "terminal_any_order":
        long_active = finite & (close > np.maximum(buy, sell))
        short_active = finite & (close < np.minimum(buy, sell))
    elif variant == "inverted_midpoint":
        midpoint = np.sqrt(buy * sell)
        long_active = finite & (buy < sell) & (close > midpoint)
        short_active = finite & (buy < sell) & (close < midpoint)
    elif variant == "inversion_plus_vwap_direction":
        vwap = state["market_vwap"].to_numpy(float)
        valid = finite & np.isfinite(vwap) & (buy < sell)
        long_active = valid & (close > vwap)
        short_active = valid & (close < vwap)
    elif variant == "inversion_plus_hourly_return":
        hourly_open = state["hourly_open"].to_numpy(float)
        valid = finite & np.isfinite(hourly_open) & (buy < sell)
        long_active = valid & (close > hourly_open)
        short_active = valid & (close < hourly_open)
    elif variant == "vwap_terminal":
        vwap = state["market_vwap"].to_numpy(float)
        valid = finite & np.isfinite(vwap)
        long_active = valid & (close > vwap)
        short_active = valid & (close < vwap)
    elif variant == "hourly_return":
        hourly_open = state["hourly_open"].to_numpy(float)
        valid = finite & np.isfinite(hourly_open)
        long_active = valid & (close > hourly_open)
        short_active = valid & (close < hourly_open)
    elif variant == "taker_flow":
        flow = state["taker_imbalance"].to_numpy(float)
        valid = finite & np.isfinite(flow)
        long_active = valid & (flow > 0.0)
        short_active = valid & (flow < 0.0)
    else:
        raise KeyError(variant)
    if flip:
        return short_active, long_active
    return long_active, short_active


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


def build_mask_bank(state: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: topology_masks(state, name)
        for name in (
            "inverted_terminal",
            "ordinary_terminal",
            "terminal_any_order",
            "inverted_midpoint",
            "inversion_plus_vwap_direction",
            "inversion_plus_hourly_return",
            "vwap_terminal",
            "hourly_return",
            "taker_flow",
        )
    }


def detailed_support_counts(
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    window: str,
) -> dict[str, int]:
    start, end = WINDOWS[window]
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(
        bool
    )
    long_active = np.asarray(long_active, dtype=bool)
    short_active = np.asarray(short_active, dtype=bool)
    active = long_active | short_active
    candidates = np.arange(0, len(dates) - HOLD_BARS - 2, dtype=np.int64)
    candidates = candidates[period[candidates] & active[candidates]]
    executable_long = executable_short = 0
    next_position = 0
    for position in candidates:
        if position < next_position:
            continue
        entry_position = position + 1
        exit_position = entry_position + HOLD_BARS
        if exit_position >= len(dates) or not period[exit_position]:
            continue
        executable_long += int(long_active[position])
        executable_short += int(short_active[position])
        next_position = exit_position + 1
    return {
        "raw": int((period & active).sum()),
        "raw_long": int((period & long_active).sum()),
        "raw_short": int((period & short_active).sum()),
        "strict_executable": executable_long + executable_short,
        "strict_executable_long": executable_long,
        "strict_executable_short": executable_short,
    }


def build_support_preflight(
    dates: pd.Series,
    mask_bank: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        name: {
            window: detailed_support_counts(
                dates, masks[0], masks[1], window=window
            )
            for window in WINDOWS
        }
        for name, masks in mask_bank.items()
    }


def novelty_audit(
    state: pd.DataFrame,
    mask_bank: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    primary_long, primary_short = mask_bank["inverted_terminal"]
    primary_events = primary_long | primary_short
    event_overlap = {
        name: event_jaccard(primary_events, masks[0] | masks[1])
        for name, masks in mask_bank.items()
        if name != "inverted_terminal"
    }
    hourly_return = np.log(
        state["hourly_close"].to_numpy(float)
        / state["hourly_open"].to_numpy(float)
    )
    close_vwap = np.log(
        state["hourly_close"].to_numpy(float)
        / state["market_vwap"].to_numpy(float)
    )
    signed_score = state["signed_inversion_settlement"].to_numpy(float)
    feature_spearman = {
        "hourly_return": finite_spearman(signed_score, hourly_return),
        "close_minus_market_vwap": finite_spearman(signed_score, close_vwap),
        "taker_quote_imbalance": finite_spearman(
            signed_score, state["taker_imbalance"].to_numpy(float)
        ),
    }
    return {
        "event_jaccard": event_overlap,
        "max_event_jaccard": max(event_overlap.values()),
        "feature_spearman": feature_spearman,
        "max_abs_feature_spearman": max(abs(value) for value in feature_spearman.values()),
        "novelty_pass": bool(
            max(event_overlap.values()) < 0.50
            and max(abs(value) for value in feature_spearman.values()) < 0.80
        ),
        "gate": "max event Jaccard < 0.50 and max absolute Spearman < 0.80 against simple price/VWAP/taker controls",
    }


def run(*, support_only: bool = False) -> dict[str, Any]:
    market, dates = load_pre2024()
    state = build_centroid_state(market, dates)
    accounting = validate_centroid_accounting(state)
    mask_bank = build_mask_bank(state)
    support_preflight = build_support_preflight(dates, mask_bank)
    primary_long, primary_short = mask_bank["inverted_terminal"]
    preflight = {
        "accounting": accounting,
        "support": support_preflight,
        "raw_primary_long_short": [
            int(primary_long.sum()),
            int(primary_short.sum()),
        ],
    }
    if support_only:
        return {"support_only": True, **preflight}

    extremes = (
        _future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"),
        _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"),
    )
    primary_stats = simulate(
        market, dates, primary_long, primary_short, extremes
    )
    print_stats("PRIMARY aggressor_centroid_inversion", primary_stats)

    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(
            market,
            dates,
            *topology_masks(state, "inverted_terminal", flip=True),
            extremes,
        )
    }
    for name, masks in mask_bank.items():
        if name != "inverted_terminal":
            controls[name] = simulate(market, dates, masks[0], masks[1], extremes)
    for name, bars in (
        ("signal_delay_5m", 1),
        ("signal_delay_1h", 12),
        ("signal_delay_24h", 288),
        ("signal_delay_7d", 2016),
    ):
        controls[name] = simulate(
            market,
            dates,
            lag_boolean(primary_long, bars),
            lag_boolean(primary_short, bars),
            extremes,
        )
    for name, stats in controls.items():
        print_stats("CONTROL " + name, stats)

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
    novelty = novelty_audit(state, mask_bank)
    structural_controls = (
        "ordinary_terminal",
        "terminal_any_order",
        "inverted_midpoint",
        "inversion_plus_vwap_direction",
        "inversion_plus_hourly_return",
        "vwap_terminal",
        "hourly_return",
        "taker_flow",
    )
    output = {
        "protocol": {
            "source_cutoff": "returned analysis frame hard-filtered strictly before 2024-01-01",
            "source_io_disclosure": "shared chunk parser may read and immediately discard later rows in the cutoff-crossing chunk; none enters the returned frame or computation",
            "raw_semantics": "completed kline taker-buy base/quote ratios reconstruct buyer centroid; total minus taker-buy quantities reconstruct seller centroid",
            "mechanism": "threshold-free aggressor centroid inversion plus terminal settlement beyond both centroids",
            "grid_size": 1,
            "grid": "one continuation map and one fixed 12h hold; exact reversal is a falsification control",
            "support_only_preflight": {
                "performed_before_returns": True,
                "counts": support_preflight,
            },
            "entry": "completed minute-55 hour enters next minute-00 open; a separate one-row delay control enters minute-05",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
            "contamination_note": "new raw representation but all pre-2024 outcomes are exploratory; 2023 is inspected internal selection and 2024+ excluded",
            "ontology_note": "centroids are observed aggregate execution averages; trapped/informed inventory is an economic hypothesis, not an observed account label",
        },
        "accounting_audit": accounting,
        "state_summary": {
            "raw_primary_events": int((primary_long | primary_short).sum()),
            "raw_primary_long_short": [
                int(primary_long.sum()),
                int(primary_short.sum()),
            ],
        },
        "primary": {
            "prelim_admitted": admission(primary_stats),
            "rank": list(rank_key(primary_stats)),
            "stats": primary_stats,
        },
        "novelty_audit": novelty,
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(
            admission(primary_stats)
            and novelty["novelty_pass"]
            and admission(controls["signal_delay_5m"])
            and not admission(controls["direction_flip"])
            and not any(admission(controls[name]) for name in structural_controls)
        ),
    }
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--support-only",
        action="store_true",
        help="compute accounting and event support without opening return outcomes",
    )
    args = parser.parse_args()
    output = run(support_only=args.support_only)
    if args.support_only:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
