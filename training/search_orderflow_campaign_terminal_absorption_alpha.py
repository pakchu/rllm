"""Search a terminal-absorption fade after a frozen order-flow campaign.

The parent trophic campaign follows repeated sponsor-to-crowd continuation
events.  This experiment changes the economic transition rather than retuning
that family: after a frozen campaign confirmation, it waits for the first
same-side aggressive-flow phase that no longer produces price progress, then
fades the campaign side at the next open.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_orderflow_trophic_campaign_alpha import (
    base_events,
    campaign_signals,
)
from training.search_orderflow_trophic_succession_alpha import (
    SEGMENTS,
    WINDOWS,
    load_pre2024,
)
from training.search_positioning_disagreement_alpha import (
    _future_extreme,
    _simulate_no_stop,
)

PROFILE = (12, 24, 6)
TAIL_QUANTILE = 0.95
CAMPAIGN_LOOKBACK = 144
CAMPAIGN_MIN_EVENTS = 2
CAMPAIGN_MAX_OPPOSITE = 1
VARIANTS = ((72, 72), (144, 144))
SIDE_COST = 0.0006


def terminal_absorption_scores(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Score absorption relative to an externally frozen long/short campaign."""
    flow = features["a_imbalance_z"].to_numpy(float)
    progress = features["a_return_z"].to_numpy(float)
    impact = features["a_impact_z"].to_numpy(float)
    close_location = features["a_clv"].to_numpy(float)
    long_score = flow - progress - impact - close_location
    short_score = -flow + progress - impact + close_location
    return long_score, short_score


def terminal_absorption_signals(
    campaign_long: np.ndarray,
    campaign_short: np.ndarray,
    long_score: np.ndarray,
    short_score: np.ndarray,
    *,
    threshold: float,
    max_wait_bars: int,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fade the first causal absorption phase after each campaign confirmation."""
    campaign_long = np.asarray(campaign_long, dtype=bool)
    campaign_short = np.asarray(campaign_short, dtype=bool)
    long_score = np.asarray(long_score, dtype=float)
    short_score = np.asarray(short_score, dtype=float)
    if not (
        campaign_long.shape
        == campaign_short.shape
        == long_score.shape
        == short_score.shape
    ):
        raise ValueError("campaign and score arrays must have the same shape")
    if np.any(campaign_long & campaign_short):
        raise ValueError("campaign direction must be unambiguous")
    if max_wait_bars <= 0 or not np.isfinite(threshold):
        raise ValueError("max wait and threshold must be valid")

    output_long = np.zeros(len(long_score), dtype=bool)
    output_short = np.zeros(len(long_score), dtype=bool)
    signal_age = np.full(len(long_score), np.nan, dtype=float)
    signal_score = np.full(len(long_score), np.nan, dtype=float)
    active_side = 0
    start = -1
    started = expired = 0
    for position in range(len(long_score)):
        if active_side and position - start > max_wait_bars:
            expired += 1
            active_side = 0
            start = -1
        if active_side and position > start:
            score = long_score[position] if active_side > 0 else short_score[position]
            if np.isfinite(score) and score >= threshold:
                trade_side = -active_side
                if flip:
                    trade_side = -trade_side
                output_long[position] = trade_side > 0
                output_short[position] = trade_side < 0
                signal_age[position] = position - start
                signal_score[position] = score
                active_side = 0
                start = -1
                continue
        if not active_side:
            if campaign_long[position]:
                active_side = 1
                start = position
                started += 1
            elif campaign_short[position]:
                active_side = -1
                start = position
                started += 1
    return output_long, output_short, {
        "started_campaigns": started,
        "expired_campaigns": expired,
        "signal_age": signal_age,
        "signal_score": signal_score,
    }


def standalone_absorption_signals(
    long_score: np.ndarray,
    short_score: np.ndarray,
    *,
    threshold: float,
    cooldown_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Control: fade an absorption onset without requiring a campaign."""
    long_score = np.asarray(long_score, dtype=float)
    short_score = np.asarray(short_score, dtype=float)
    finite = np.isfinite(long_score) & np.isfinite(short_score)
    best = np.maximum(long_score, short_score)
    inferred_side = np.where(long_score > short_score, 1, -1)
    active = finite & (best >= threshold) & (long_score != short_score)
    onset = active & ~np.r_[False, active[:-1]]
    output_long = np.zeros(len(active), dtype=bool)
    output_short = np.zeros(len(active), dtype=bool)
    next_allowed = 0
    for position in np.flatnonzero(onset):
        if position < next_allowed:
            continue
        trade_side = -int(inferred_side[position])
        output_long[position] = trade_side > 0
        output_short[position] = trade_side < 0
        next_allowed = position + cooldown_bars
    return output_long, output_short


def lag_boolean(values: np.ndarray, bars: int) -> np.ndarray:
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
    hold_bars: int,
) -> dict[str, int]:
    """Count raw and canonical non-overlapping/split-contained signals only."""
    start, end = WINDOWS[window]
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    active = np.asarray(long_active, dtype=bool) | np.asarray(short_active, dtype=bool)
    raw = int((period & active).sum())
    candidates = np.arange(0, len(dates) - hold_bars - 2, dtype=np.int64)
    candidates = candidates[period[candidates] & active[candidates]]
    executable = 0
    next_position = 0
    for position in candidates:
        if position < next_position:
            continue
        entry_position = position + 1
        exit_position = entry_position + hold_bars
        if exit_position >= len(dates) or not period[exit_position]:
            continue
        executable += 1
        next_position = exit_position + 1
    return {"raw": raw, "strict_executable": executable}


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    hold_bars: int,
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
            hold_bars=hold_bars,
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
        and stats["select_2023"]["trades"] >= 18
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 6
        and min(stats["fit"]["longs"], stats["fit"]["shorts"]) >= 8
        and min(stats["select_2023"]["longs"], stats["select_2023"]["shorts"]) >= 4
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
    enough = stats["fit"]["trades"] >= 60 and stats["select_2023"]["trades"] >= 18
    core = [
        stats[name]["ratio"]
        for name in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")
    ]
    return (
        admission(stats),
        enough,
        min(core) > 0.0,
        sum(stats[name]["return_pct"] > 0.0 for name in SEGMENTS),
        min(core),
        float(np.median(core)),
        stats["select_2023"]["trades"],
    )


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
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("terminal-absorption search requires a complete 5-minute grid")
    event_long, event_short, features, thresholds = base_events(market, dates, PROFILE)
    campaign_long, campaign_short, _ = campaign_signals(
        event_long,
        event_short,
        lookback_bars=CAMPAIGN_LOOKBACK,
        min_same_events=CAMPAIGN_MIN_EVENTS,
        max_opposite_events=CAMPAIGN_MAX_OPPOSITE,
    )
    long_score, short_score = terminal_absorption_scores(features)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for _, hold in VARIANTS
    }

    rows: list[dict[str, Any]] = []
    signal_bank: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    support_preflight: dict[str, dict[str, dict[str, int]]] = {}
    for max_wait, hold in VARIANTS:
        long_active, short_active, diagnostics = terminal_absorption_signals(
            campaign_long,
            campaign_short,
            long_score,
            short_score,
            threshold=thresholds["absorption_role"],
            max_wait_bars=max_wait,
        )
        signal_bank[max_wait] = (long_active, short_active, diagnostics)
        support_preflight[f"wait{max_wait}_hold{hold}"] = {
            window: support_counts(
                dates,
                long_active,
                short_active,
                window=window,
                hold_bars=hold,
            )
            for window in ("fit", "select_2023")
        }
        stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
        rows.append(
            {
                "profile": list(PROFILE),
                "tail_quantile": TAIL_QUANTILE,
                "campaign_lookback": CAMPAIGN_LOOKBACK,
                "campaign_min_events": CAMPAIGN_MIN_EVENTS,
                "campaign_max_opposite": CAMPAIGN_MAX_OPPOSITE,
                "max_wait": max_wait,
                "hold": hold,
                "absorption_threshold": thresholds["absorption_role"],
                "campaign_events": int((campaign_long | campaign_short).sum()),
                "terminal_events": int((long_active | short_active).sum()),
                "terminal_long_short": [int(long_active.sum()), int(short_active.sum())],
                "started_campaigns": diagnostics["started_campaigns"],
                "expired_campaigns": diagnostics["expired_campaigns"],
                "median_terminal_age": float(np.nanmedian(diagnostics["signal_age"])),
                "prelim_admitted": admission(stats),
                "rank": rank_key(stats),
                "stats": stats,
            }
        )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    for index, row in enumerate(rows, 1):
        print_stats(
            f"RANK {index} wait{row['max_wait']} hold{row['hold']} "
            f"events={row['terminal_events']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    base_long, base_short, _ = signal_bank[top["max_wait"]]
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(
            market, dates, base_short, base_long, top["hold"], extremes[top["hold"]]
        ),
        "parent_campaign_continuation": simulate(
            market,
            dates,
            campaign_long,
            campaign_short,
            top["hold"],
            extremes[top["hold"]],
        ),
    }
    standalone_long, standalone_short = standalone_absorption_signals(
        long_score,
        short_score,
        threshold=thresholds["absorption_role"],
        cooldown_bars=CAMPAIGN_LOOKBACK,
    )
    controls["absorption_without_campaign"] = simulate(
        market,
        dates,
        standalone_long,
        standalone_short,
        top["hold"],
        extremes[top["hold"]],
    )
    swapped_event_long, swapped_event_short, _, _ = base_events(
        market, dates, PROFILE, order_swap=True
    )
    swapped_campaign_long, swapped_campaign_short, _ = campaign_signals(
        swapped_event_long,
        swapped_event_short,
        lookback_bars=CAMPAIGN_LOOKBACK,
        min_same_events=CAMPAIGN_MIN_EVENTS,
        max_opposite_events=CAMPAIGN_MAX_OPPOSITE,
    )
    swapped_long, swapped_short, _ = terminal_absorption_signals(
        swapped_campaign_long,
        swapped_campaign_short,
        long_score,
        short_score,
        threshold=thresholds["absorption_role"],
        max_wait_bars=top["max_wait"],
    )
    controls["phase_order_swap"] = simulate(
        market, dates, swapped_long, swapped_short, top["hold"], extremes[top["hold"]]
    )
    for name, bars in (("signal_delay_1h", 12), ("signal_delay_6h", 72), ("signal_delay_7d", 2016)):
        controls[name] = simulate(
            market,
            dates,
            lag_boolean(base_long, bars),
            lag_boolean(base_short, bars),
            top["hold"],
            extremes[top["hold"]],
        )
    for name, stats in controls.items():
        print_stats("CONTROL " + name, stats)

    cost_stress = {
        str(bp): simulate(
            market,
            dates,
            base_long,
            base_short,
            top["hold"],
            extremes[top["hold"]],
            side_cost=bp / 10_000.0,
        )
        for bp in (0, 1, 3, 6, 10, 15)
    }
    output = {
        "protocol": {
            "source_cutoff": "returned analysis frame hard-filtered strictly before 2024-01-01",
            "source_io_disclosure": "the shared chunked CSV parser can read and immediately discard rows after the cutoff in its crossing chunk; no such row enters the returned frame, features, thresholds, signals or outcomes",
            "parent_usage": "frozen q95 profile (12,24,6), 144-bar/k2/<=1-opposite campaign; no parent retuning",
            "mechanism": "first same-side terminal absorption after a confirmed campaign, then fade campaign side",
            "grid_size": len(rows),
            "grid": "two co-primary policies with wait/hold tied at 6h or 12h; deterministic rank_key selects the reported top and no post-result horizon expansion is allowed",
            "state_machine": {
                "scan": "campaign confirmed on completed bar i; inspect only completed bars j=i+1..i+h inclusive",
                "emission": "first threshold hit emits a fade at completed j and enters j+1 open",
                "expiry": "no hit by i+h expires the campaign",
                "pending_precedence": "one pending campaign; every new confirmation while pending, including at the inclusive endpoint, is ignored",
                "trade_overlap": "signal generation continues, but the canonical simulator executes the first candidate and ignores candidates through exit+1",
            },
            "threshold_semantics": "terminal absorption score evaluated under the frozen campaign side, using the frozen parent q95 numeric absorption threshold; it is not a literal later parent-role event",
            "support_only_preflight": {
                "performed_before_returns": True,
                "counts": support_preflight,
            },
            "selection_rule": "rank by admission, support, positive full/half windows, positive segment count, minimum and median core ratios, then 2023 trades",
            "control_rule": "all controls reuse identical entry/hold/non-overlap accounting and frozen numeric thresholds; no refits",
            "entry": "completed terminal bar enters next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
            "contamination_note": "the parent profile/lookback/count came from an outcome-ranked rejected pre-2024 campaign scan; this recombination is contaminated exploratory research, 2023 is inspected internal selection, and 2024+ remained sealed",
        },
        "thresholds": thresholds,
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(
            top["prelim_admitted"] and not any(admission(stats) for stats in controls.values())
        ),
    }
    Path(
        "results/orderflow_campaign_terminal_absorption_alpha_scan_2026-07-14.json"
    ).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
