"""Search a multiplex visibility-graph layer-irreversibility alpha.

Each completed six-hour price-return and aggressive-flow series is mapped to a
directed horizontal visibility graph (HVG).  The policy follows the current
direction of whichever layer has the larger in/out degree-distribution
irreversibility.  This is a fixed arbitration rule over an observable
time-asymmetry descriptor, not proof of causality or predictability.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_crossmap_flow_causality_alpha import (
    HOLD_BARS,
    MARKET_PATH,
    SEGMENTS,
    SIDE_COST,
    WINDOWS,
    _frame_sha256,
    admission,
    build_completed_blocks,
    build_crossmap_features,
    event_jaccard,
    finite_spearman,
    load_pre2024,
    shift_masks,
    simulate,
    support_counts,
    support_passes,
)
from training.search_positioning_disagreement_alpha import _future_extreme


RESULT_PATH = Path("results/visibility_layer_irreversibility_alpha_scan_2026-07-14.json")
VISIBILITY_BLOCKS = 168
JEFFREYS_PRIOR = 0.5
RATIO_EPSILON = 1e-9
GATE_LOOKBACK = 120
GATE_MIN_OBSERVATIONS = 60
GATE_QUANTILE = 0.80
TREND_BLOCKS = 28
ORDINAL_ORDER = 3


def directed_hvg_degrees(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return directed HVG in/out degrees in linear time.

    An edge ``i -> j`` exists for ``i < j`` when every intermediate value is
    strictly below ``min(values[i], values[j])``.  Replacing an equal-height
    stack top is required because the newer equal node blocks the older one.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("HVG input must be a finite one-dimensional sequence")
    in_degree = np.zeros(len(values), dtype=np.int16)
    out_degree = np.zeros(len(values), dtype=np.int16)
    stack: list[int] = []
    for current, value in enumerate(values):
        while stack and value > values[stack[-1]]:
            previous = stack.pop()
            out_degree[previous] += 1
            in_degree[current] += 1
        if stack:
            previous = stack[-1]
            out_degree[previous] += 1
            in_degree[current] += 1
            if value == values[previous]:
                stack.pop()
        stack.append(current)
    return in_degree, out_degree


def degree_irreversibility(values: np.ndarray, *, prior: float = JEFFREYS_PRIOR) -> float:
    """Symmetric KL mismatch between directed HVG in/out degree laws."""
    if prior <= 0.0:
        raise ValueError("degree-distribution prior must be positive")
    in_degree, out_degree = directed_hvg_degrees(values)
    maximum = int(max(in_degree.max(), out_degree.max()))
    inbound = np.bincount(in_degree, minlength=maximum + 1).astype(float) + prior
    outbound = np.bincount(out_degree, minlength=maximum + 1).astype(float) + prior
    inbound /= inbound.sum()
    outbound /= outbound.sum()
    forward = float(np.sum(inbound * np.log(inbound / outbound)))
    reverse = float(np.sum(outbound * np.log(outbound / inbound)))
    return 0.5 * (forward + reverse)


def permutation_entropy(values: np.ndarray, *, order: int = ORDINAL_ORDER) -> float:
    """Normalized ordinal-pattern entropy used only as a novelty reference."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or order < 2 or len(values) < order or not np.isfinite(values).all():
        return float("nan")
    patterns = {pattern: index for index, pattern in enumerate(itertools.permutations(range(order)))}
    counts = np.zeros(math.factorial(order), dtype=float)
    for window in np.lib.stride_tricks.sliding_window_view(values, order):
        pattern = tuple(np.argsort(window, kind="stable").tolist())
        counts[patterns[pattern]] += 1.0
    probability = counts[counts > 0.0] / counts.sum()
    return float(-np.sum(probability * np.log(probability)) / np.log(float(len(counts))))


def _prior_threshold(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .shift(1)
        .rolling(GATE_LOOKBACK, min_periods=GATE_MIN_OBSERVATIONS)
        .quantile(GATE_QUANTILE)
        .to_numpy(float)
    )


def build_visibility_features(blocks: pd.DataFrame) -> pd.DataFrame:
    output = blocks.copy().reset_index(drop=True)
    price = pd.to_numeric(output["price_return"], errors="coerce").to_numpy(float)
    flow = pd.to_numeric(output["flow_fraction"], errors="coerce").to_numpy(float)
    price_irreversibility = np.full(len(output), np.nan)
    flow_irreversibility = np.full(len(output), np.nan)
    price_ordinal_entropy = np.full(len(output), np.nan)
    for index in range(VISIBILITY_BLOCKS - 1, len(output)):
        source = slice(index - VISIBILITY_BLOCKS + 1, index + 1)
        price_irreversibility[index] = degree_irreversibility(price[source])
        flow_irreversibility[index] = degree_irreversibility(flow[source])
        price_ordinal_entropy[index] = permutation_entropy(price[source])
    layer_log_ratio = np.log(
        (flow_irreversibility + RATIO_EPSILON)
        / (price_irreversibility + RATIO_EPSILON)
    )
    layer_score = np.abs(layer_log_ratio)
    output["price_hvg_irreversibility"] = price_irreversibility
    output["flow_hvg_irreversibility"] = flow_irreversibility
    output["price_ordinal_entropy_o3"] = price_ordinal_entropy
    output["hvg_layer_log_ratio"] = layer_log_ratio
    output["hvg_layer_score"] = layer_score
    output["hvg_layer_threshold"] = _prior_threshold(layer_score)
    output["price_hvg_threshold"] = _prior_threshold(price_irreversibility)
    output["flow_hvg_threshold"] = _prior_threshold(flow_irreversibility)
    output["price_realized_vol"] = (
        pd.Series(price).rolling(VISIBILITY_BLOCKS, min_periods=VISIBILITY_BLOCKS).std(ddof=0)
    )
    output["mean_absolute_flow"] = (
        pd.Series(np.abs(flow)).rolling(VISIBILITY_BLOCKS, min_periods=VISIBILITY_BLOCKS).mean()
    )
    output["price_trend"] = pd.Series(price).rolling(TREND_BLOCKS, min_periods=TREND_BLOCKS).sum()
    return output


def _signals_from_side(
    features: pd.DataFrame,
    rows: int,
    side: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    side = np.asarray(side, dtype=float)
    active = np.asarray(active, dtype=bool) & np.isfinite(side) & (side != 0.0)
    positions = features["position"].to_numpy(np.int64)
    long_active = np.zeros(rows, dtype=bool)
    short_active = np.zeros(rows, dtype=bool)
    long_active[positions[active & (side > 0.0)]] = True
    short_active[positions[active & (side < 0.0)]] = True
    return long_active, short_active


def policy_masks(
    features: pd.DataFrame,
    rows: int,
    *,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = features["hvg_layer_log_ratio"].to_numpy(float)
    score = features["hvg_layer_score"].to_numpy(float)
    threshold = features["hvg_layer_threshold"].to_numpy(float)
    price = features["price_return"].to_numpy(float)
    flow = features["flow_fraction"].to_numpy(float)
    active = np.isfinite(score) & np.isfinite(threshold) & (score > threshold)
    dominant_side = np.where(ratio > 0.0, np.sign(flow), np.sign(price))
    if flip:
        dominant_side = -dominant_side
    return _signals_from_side(features, rows, dominant_side, active)


def control_masks(features: pd.DataFrame, rows: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    score = features["hvg_layer_score"].to_numpy(float)
    threshold = features["hvg_layer_threshold"].to_numpy(float)
    price = features["price_return"].to_numpy(float)
    flow = features["flow_fraction"].to_numpy(float)
    primary_active = np.isfinite(score) & np.isfinite(threshold) & (score > threshold)
    price_irreversibility = features["price_hvg_irreversibility"].to_numpy(float)
    price_threshold = features["price_hvg_threshold"].to_numpy(float)
    price_active = (
        np.isfinite(price_irreversibility)
        & np.isfinite(price_threshold)
        & (price_irreversibility > price_threshold)
    )
    flow_irreversibility = features["flow_hvg_irreversibility"].to_numpy(float)
    flow_threshold = features["flow_hvg_threshold"].to_numpy(float)
    flow_active = (
        np.isfinite(flow_irreversibility)
        & np.isfinite(flow_threshold)
        & (flow_irreversibility > flow_threshold)
    )
    return {
        "same_events_flow_follow": _signals_from_side(features, rows, np.sign(flow), primary_active),
        "same_events_price_follow": _signals_from_side(features, rows, np.sign(price), primary_active),
        "price_hvg_only_follow_price": _signals_from_side(features, rows, np.sign(price), price_active),
        "flow_hvg_only_follow_flow": _signals_from_side(features, rows, np.sign(flow), flow_active),
    }


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


def run(*, market_path: str | Path = MARKET_PATH, support_only: bool = False) -> dict[str, Any]:
    market, dates = load_pre2024(market_path)
    blocks = build_completed_blocks(market, dates)
    features = build_visibility_features(blocks)
    primary = policy_masks(features, len(market))
    controls_mask = control_masks(features, len(market))
    crossmap = build_crossmap_features(blocks)
    crossmap_score = crossmap["crossmap_dominance"].to_numpy(float)
    crossmap_threshold = crossmap["dominance_threshold"].to_numpy(float)
    crossmap_event = (
        np.isfinite(crossmap_score)
        & np.isfinite(crossmap_threshold)
        & (np.abs(crossmap_score) > crossmap_threshold)
    )
    primary_positions = blocks["position"].to_numpy(np.int64)
    primary_event_blocks = (primary[0] | primary[1])[primary_positions]
    layer_ratio = features["hvg_layer_log_ratio"].to_numpy(float)
    layer_score = features["hvg_layer_score"].to_numpy(float)
    novelty = {
        "layer_ratio_vs_crossmap_dominance_spearman": finite_spearman(
            layer_ratio, crossmap_score
        ),
        "layer_score_vs_price_vol_spearman": finite_spearman(
            layer_score, features["price_realized_vol"].to_numpy(float)
        ),
        "layer_score_vs_mean_absolute_flow_spearman": finite_spearman(
            layer_score, features["mean_absolute_flow"].to_numpy(float)
        ),
        "layer_ratio_vs_price_trend_spearman": finite_spearman(
            layer_ratio, features["price_trend"].to_numpy(float)
        ),
        "layer_score_vs_price_ordinal_entropy_spearman": finite_spearman(
            layer_score, features["price_ordinal_entropy_o3"].to_numpy(float)
        ),
        "primary_vs_crossmap_event_jaccard": event_jaccard(
            primary_event_blocks, crossmap_event
        ),
    }
    novelty["passed"] = bool(
        abs(novelty["layer_ratio_vs_crossmap_dominance_spearman"]) < 0.50
        and abs(novelty["layer_score_vs_price_vol_spearman"]) < 0.50
        and abs(novelty["layer_score_vs_mean_absolute_flow_spearman"]) < 0.50
        and abs(novelty["layer_ratio_vs_price_trend_spearman"]) < 0.50
        and abs(novelty["layer_score_vs_price_ordinal_entropy_spearman"]) < 0.50
        and novelty["primary_vs_crossmap_event_jaccard"] < 0.60
    )
    support = {
        window: support_counts(dates, *primary, window=window)
        for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")
    }
    preflight = {
        "support_only": True,
        "support": support,
        "support_passed": support_passes(support),
        "novelty": novelty,
        "preflight_passed": bool(support_passes(support) and novelty["passed"]),
        "completed_blocks": int(len(blocks)),
        "finite_visibility_states": int(np.isfinite(layer_score).sum()),
        "source_latest": str(pd.to_datetime(blocks["source_time"]).max()),
        "outcomes_opened": False,
    }
    if support_only:
        return preflight
    if not preflight["preflight_passed"]:
        raise RuntimeError("support/novelty preflight failed; outcome access remains closed")

    extremes = (
        _future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"),
        _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"),
    )
    primary_stats = simulate(market, dates, *primary, extremes=extremes)
    controls_mask["exact_direction_flip"] = (primary[1].copy(), primary[0].copy())
    controls_mask["signal_delay_6h"] = shift_masks(*primary, 6 * 12)
    controls_mask["signal_delay_7d"] = shift_masks(*primary, 7 * 24 * 12)
    controls = {
        name: simulate(market, dates, *masks, extremes=extremes)
        for name, masks in controls_mask.items()
    }
    cost_stress = {
        str(bp): simulate(
            market,
            dates,
            *primary,
            side_cost=bp / 10_000.0,
            extremes=extremes,
        )
        for bp in (0, 1, 3, 6, 10, 15)
    }
    entry_delay = {
        str(minutes): simulate(
            market,
            dates,
            *shift_masks(*primary, bars),
            extremes=extremes,
        )
        for minutes, bars in ((5, 0), (10, 1), (15, 2))
    }
    hold_diagnostics = {
        str(hours): simulate(market, dates, *primary, hold_bars=hours * 12)
        for hours in (6, 12, 24)
    }
    control_admissions = {name: admission(stats) for name, stats in controls.items()}
    structural_controls = (
        "same_events_flow_follow",
        "same_events_price_follow",
        "price_hvg_only_follow_price",
        "flow_hvg_only_follow_flow",
    )
    output = {
        "protocol": {
            "source_cutoff": "returned market frame strictly before 2024-01-01",
            "source_io_disclosure": "a cutoff-crossing chunk may be physically decoded and immediately discarded; discarded rows never enter returned frames, hashes or computation",
            "mechanism": "separate directed HVGs for completed six-hour price returns and aggressive flow; follow the current sign of the layer with larger symmetric in/out degree-law mismatch",
            "claim_disclaimer": "HVG layer irreversibility is an observable time-asymmetry descriptor, not proof of causal direction or predictive edge",
            "block": "72 completed 5m bars [T-6h,T), ending minute55 at UTC 00/06/12/18 boundary T",
            "visibility_window_blocks": VISIBILITY_BLOCKS,
            "hvg_edge": "i->j iff i<j and every intermediate value is strictly below min(x_i,x_j)",
            "degree_metric": f"0.5*(KL(Pin||Pout)+KL(Pout||Pin)); Jeffreys prior {JEFFREYS_PRIOR}",
            "ordinal_novelty_reference": f"order-{ORDINAL_ORDER} permutation entropy over the identical {VISIBILITY_BLOCKS}-block price window",
            "layer_ratio": f"log((flow_irreversibility+{RATIO_EPSILON})/(price_irreversibility+{RATIO_EPSILON}))",
            "gate": f"abs layer ratio > shift(1) prior {GATE_LOOKBACK}-state q{GATE_QUANTILE:.2f}, minimum {GATE_MIN_OBSERVATIONS}",
            "side": "flow sign when flow irreversibility is larger; otherwise price-return sign",
            "entry": "signal at boundary minute00; enter minute05 open",
            "hold_bars": HOLD_BARS,
            "leverage": 0.5,
            "cost": "6bp/side",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "selection": "fit 2020-06..2022; one-shot 2023 plus fixed H1/H2; no 2024+",
            "support_only_preflight": {"performed_before_returns": True, **preflight},
            "diagnostics_not_selected": "entry 5/10/15 minutes and hold 6/12/24 hours reported without replacement selection",
            "oos_opened": False,
        },
        "source": {
            "market_path": str(market_path),
            "pre2024_frame_sha256": _frame_sha256(market),
            "full_source_hash_computed": False,
        },
        "state_summary": {
            "completed_blocks": int(len(blocks)),
            "finite_visibility_states": int(np.isfinite(layer_score).sum()),
            "raw_signals": int(primary_event_blocks.sum()),
            "raw_long_short": [int(primary[0].sum()), int(primary[1].sum())],
        },
        "primary": {"stats": primary_stats, "prelim_admitted": admission(primary_stats)},
        "controls": controls,
        "control_admissions": control_admissions,
        "cost_stress": cost_stress,
        "entry_delay_diagnostics": entry_delay,
        "hold_diagnostics": hold_diagnostics,
        "novelty_audit": novelty,
        "prelim_admitted": admission(primary_stats),
        "final_admitted": bool(
            admission(primary_stats)
            and novelty["passed"]
            and not any(control_admissions[name] for name in structural_controls)
        ),
        "oos_opened": False,
    }
    _print_stats("PRIMARY visibility-layer irreversibility arbitration", primary_stats)
    for name, stats in controls.items():
        _print_stats("CONTROL " + name, stats)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-path", default=str(MARKET_PATH))
    parser.add_argument("--support-only", action="store_true")
    args = parser.parse_args()
    output = run(market_path=args.market_path, support_only=args.support_only)
    if args.support_only:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
