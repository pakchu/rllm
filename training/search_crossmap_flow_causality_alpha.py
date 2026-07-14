"""Search a nonlinear price/flow cross-map asymmetry alpha.

The experiment estimates nonlinear dynamical coupling from the preceding 120
completed six-hour BTC blocks.  It deliberately calls the statistic a
``cross-map asymmetry`` rather than proof of causality: CCM direction can be
unreliable for noisy or synchronized stochastic systems.

If the price manifold reconstructs prior aggressive flow better than the flow
manifold reconstructs price, the policy follows current completed flow; under
the opposite asymmetry it fades current completed flow.  These are fixed policy
labels, not validated causal regimes.  No realized trade outcome estimates the
state, side or gate.  Returned source frames are strictly pre-2024.
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
RESULT_PATH = Path("results/crossmap_flow_causality_alpha_scan_2026-07-14.json")
CUTOFF = "2024-01-01"

BLOCK_BARS = 6 * 12
LIBRARY_BLOCKS = 120
EMBEDDING_DIMENSION = 3
NEIGHBORS = EMBEDDING_DIMENSION + 1
THEILER_RADIUS = 1
GATE_LOOKBACK = 120
GATE_MIN_OBSERVATIONS = 60
GATE_QUANTILE = 0.80
HOLD_BARS = 12 * 12
LEVERAGE = 0.5
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


def _frame_sha256(frame: pd.DataFrame) -> str:
    """Hash only the physically returned, pre-cutoff analysis frame."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy("<u8").tobytes())
    return digest.hexdigest()


def load_pre2024(path: str | Path = MARKET_PATH) -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(str(path), "date", CUTOFF)
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(CUTOFF):
        raise RuntimeError("post-cutoff market row entered analysis")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("cross-map search requires a complete five-minute grid")
    required = {"open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote"}
    missing = required.difference(market.columns)
    if missing:
        raise ValueError(f"market source is missing columns: {sorted(missing)}")
    return market, dates


def build_completed_blocks(market: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    boundary = dates.dt.minute.eq(0) & dates.dt.hour.mod(6).eq(0)
    positions = np.flatnonzero(boundary.to_numpy(bool))
    positions = positions[positions >= BLOCK_BARS]
    open_price = pd.to_numeric(market["open"], errors="coerce").to_numpy(float)
    close = pd.to_numeric(market["close"], errors="coerce").to_numpy(float)
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce").to_numpy(float)
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce").to_numpy(float)
    rows: list[dict[str, Any]] = []
    for position in positions:
        start = int(position) - BLOCK_BARS
        source = slice(start, int(position))
        values = np.r_[
            open_price[start],
            close[int(position) - 1],
            quote[source],
            taker_buy[source],
        ]
        if not np.isfinite(values).all():
            continue
        if open_price[start] <= 0.0 or close[int(position) - 1] <= 0.0:
            continue
        if (quote[source] <= 0.0).any() or (taker_buy[source] < 0.0).any():
            continue
        quote_sum = float(quote[source].sum())
        if quote_sum <= 0.0 or (taker_buy[source] > quote[source] * (1.0 + 1e-9)).any():
            continue
        source_time = pd.Timestamp(dates.iloc[int(position) - 1])
        effective_time = pd.Timestamp(dates.iloc[int(position)])
        if source_time != effective_time - pd.Timedelta("5min"):
            raise RuntimeError("six-hour block did not end on a completed minute-55 bar")
        signed_flow = float((2.0 * taker_buy[source] - quote[source]).sum())
        rows.append(
            {
                "position": int(position),
                "effective_time": effective_time,
                "source_time": source_time,
                "price_return": float(np.log(close[int(position) - 1] / open_price[start])),
                "flow_fraction": signed_flow / quote_sum,
                "quote_volume": quote_sum,
            }
        )
    blocks = pd.DataFrame(rows).sort_values("effective_time").reset_index(drop=True)
    if blocks.empty:
        raise ValueError("no completed six-hour blocks")
    return blocks


def delay_embedding(values: np.ndarray, dimension: int = EMBEDDING_DIMENSION) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or dimension < 1 or len(values) < dimension:
        raise ValueError("invalid delay-embedding input")
    return np.column_stack(
        [values[dimension - 1 - lag : len(values) - lag] for lag in range(dimension)]
    )


def theiler_distance_matrix(
    embedding: np.ndarray,
    radius: int = THEILER_RADIUS,
) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError("embedding must be a finite matrix")
    difference = embedding[:, None, :] - embedding[None, :, :]
    distance = np.sqrt(np.sum(difference * difference, axis=2))
    index = np.arange(len(embedding))
    distance[np.abs(index[:, None] - index[None, :]) <= int(radius)] = np.inf
    return distance


def cross_map_skill(
    embedding: np.ndarray,
    target: np.ndarray,
    *,
    neighbors: int = NEIGHBORS,
    theiler_radius: int = THEILER_RADIUS,
) -> float:
    """Leave-one-out simplex reconstruction skill on a completed library."""
    embedding = np.asarray(embedding, dtype=float)
    target = np.asarray(target, dtype=float)
    if embedding.ndim != 2 or len(embedding) != len(target):
        raise ValueError("embedding and target must have the same row count")
    if len(target) <= neighbors + 2 * theiler_radius or not np.isfinite(target).all():
        return float("nan")
    distance = theiler_distance_matrix(embedding, theiler_radius)
    if (np.isfinite(distance).sum(axis=1) < neighbors).any():
        return float("nan")
    nearest = np.argpartition(distance, neighbors - 1, axis=1)[:, :neighbors]
    nearest_distance = np.take_along_axis(distance, nearest, axis=1)
    order = np.argsort(nearest_distance, axis=1)
    nearest = np.take_along_axis(nearest, order, axis=1)
    nearest_distance = np.take_along_axis(nearest_distance, order, axis=1)
    first = nearest_distance[:, [0]]
    positive_first = first > 1e-12
    weights = np.zeros_like(nearest_distance)
    weights[positive_first[:, 0]] = np.exp(
        -nearest_distance[positive_first[:, 0]] / first[positive_first[:, 0]]
    )
    zero_rows = ~positive_first[:, 0]
    if zero_rows.any():
        zero_neighbor = nearest_distance[zero_rows] <= 1e-12
        weights[zero_rows] = zero_neighbor.astype(float)
    weight_sum = weights.sum(axis=1, keepdims=True)
    if (weight_sum <= 0.0).any():
        return float("nan")
    prediction = np.sum((weights / weight_sum) * target[nearest], axis=1)
    if np.std(prediction) <= 1e-12 or np.std(target) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(prediction, target)[0, 1])


def crossmap_asymmetry(
    price: np.ndarray,
    flow: np.ndarray,
) -> tuple[float, float, float]:
    price = np.asarray(price, dtype=float)
    flow = np.asarray(flow, dtype=float)
    if len(price) != LIBRARY_BLOCKS or len(flow) != LIBRARY_BLOCKS:
        raise ValueError(f"cross-map library must contain {LIBRARY_BLOCKS} blocks")
    if not np.isfinite(price).all() or not np.isfinite(flow).all():
        return float("nan"), float("nan"), float("nan")
    price_std = float(np.std(price, ddof=0))
    flow_std = float(np.std(flow, ddof=0))
    if price_std <= 1e-12 or flow_std <= 1e-12:
        return float("nan"), float("nan"), float("nan")
    price_z = (price - float(np.mean(price))) / price_std
    flow_z = (flow - float(np.mean(flow))) / flow_std
    price_embedding = delay_embedding(price_z)
    flow_embedding = delay_embedding(flow_z)
    aligned_price = price_z[EMBEDDING_DIMENSION - 1 :]
    aligned_flow = flow_z[EMBEDDING_DIMENSION - 1 :]
    # CCM convention: an effect manifold can reconstruct its putative cause.
    flow_to_price = cross_map_skill(price_embedding, aligned_flow)
    price_to_flow = cross_map_skill(flow_embedding, aligned_price)
    return flow_to_price, price_to_flow, flow_to_price - price_to_flow


def linear_leadlag_asymmetry(price: np.ndarray, flow: np.ndarray) -> float:
    price = np.asarray(price, dtype=float)
    flow = np.asarray(flow, dtype=float)
    if len(price) < 3 or len(price) != len(flow):
        return float("nan")
    flow_leads = np.corrcoef(flow[:-1], price[1:])[0, 1]
    price_leads = np.corrcoef(price[:-1], flow[1:])[0, 1]
    return float(flow_leads - price_leads)


def build_crossmap_features(blocks: pd.DataFrame) -> pd.DataFrame:
    output = blocks.copy().reset_index(drop=True)
    price = pd.to_numeric(output["price_return"], errors="coerce").to_numpy(float)
    flow = pd.to_numeric(output["flow_fraction"], errors="coerce").to_numpy(float)
    flow_to_price = np.full(len(output), np.nan)
    price_to_flow = np.full(len(output), np.nan)
    dominance = np.full(len(output), np.nan)
    linear = np.full(len(output), np.nan)
    correlation = np.full(len(output), np.nan)
    for index in range(LIBRARY_BLOCKS, len(output)):
        source = slice(index - LIBRARY_BLOCKS, index)
        first, second, difference = crossmap_asymmetry(price[source], flow[source])
        flow_to_price[index] = first
        price_to_flow[index] = second
        dominance[index] = difference
        linear[index] = linear_leadlag_asymmetry(price[source], flow[source])
        correlation[index] = float(np.corrcoef(price[source], flow[source])[0, 1])
    output["flow_to_price_skill"] = flow_to_price
    output["price_to_flow_skill"] = price_to_flow
    output["crossmap_dominance"] = dominance
    output["linear_leadlag_dominance"] = linear
    output["price_flow_correlation"] = correlation
    output["dominance_threshold"] = (
        pd.Series(np.abs(dominance))
        .shift(1)
        .rolling(GATE_LOOKBACK, min_periods=GATE_MIN_OBSERVATIONS)
        .quantile(GATE_QUANTILE)
    )
    output["linear_threshold"] = (
        pd.Series(np.abs(linear))
        .shift(1)
        .rolling(GATE_LOOKBACK, min_periods=GATE_MIN_OBSERVATIONS)
        .quantile(GATE_QUANTILE)
    )
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


def policy_masks(features: pd.DataFrame, rows: int, *, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    dominance = features["crossmap_dominance"].to_numpy(float)
    threshold = features["dominance_threshold"].to_numpy(float)
    flow = features["flow_fraction"].to_numpy(float)
    active = np.isfinite(dominance) & np.isfinite(threshold) & (np.abs(dominance) > threshold)
    side = np.sign(dominance) * np.sign(flow) * (-1.0 if flip else 1.0)
    return _signals_from_side(features, rows, side, active)


def control_masks(features: pd.DataFrame, rows: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    dominance = features["crossmap_dominance"].to_numpy(float)
    threshold = features["dominance_threshold"].to_numpy(float)
    flow = features["flow_fraction"].to_numpy(float)
    price = features["price_return"].to_numpy(float)
    primary_active = np.isfinite(dominance) & np.isfinite(threshold) & (np.abs(dominance) > threshold)
    linear = features["linear_leadlag_dominance"].to_numpy(float)
    linear_threshold = features["linear_threshold"].to_numpy(float)
    linear_active = np.isfinite(linear) & np.isfinite(linear_threshold) & (np.abs(linear) > linear_threshold)
    return {
        "same_events_flow_follow": _signals_from_side(features, rows, np.sign(flow), primary_active),
        "same_events_flow_fade": _signals_from_side(features, rows, -np.sign(flow), primary_active),
        "same_events_price_follow": _signals_from_side(features, rows, np.sign(price), primary_active),
        "same_events_price_fade": _signals_from_side(features, rows, -np.sign(price), primary_active),
        "ordinary_linear_leadlag": _signals_from_side(
            features, rows, np.sign(linear) * np.sign(flow), linear_active
        ),
    }


def shift_masks(long_active: np.ndarray, short_active: np.ndarray, bars: int) -> tuple[np.ndarray, np.ndarray]:
    if bars <= 0:
        return np.asarray(long_active, dtype=bool).copy(), np.asarray(short_active, dtype=bool).copy()
    pad = np.zeros(bars, dtype=bool)
    return (
        np.r_[pad, np.asarray(long_active, dtype=bool)[:-bars]],
        np.r_[pad, np.asarray(short_active, dtype=bool)[:-bars]],
    )


def support_counts(
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    window: str,
    hold_bars: int = HOLD_BARS,
) -> dict[str, int]:
    start, end = WINDOWS[window]
    executable = select_executable_positions(
        dates,
        long_active,
        short_active,
        start=start,
        end=end,
        hold_bars=hold_bars,
    )
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    long_active = np.asarray(long_active, dtype=bool)
    short_active = np.asarray(short_active, dtype=bool)
    raw_positions = np.flatnonzero(period & (long_active | short_active))
    selected = np.asarray(executable, dtype=np.int64)
    return {
        "raw": int(len(raw_positions)),
        "raw_long": int((period & long_active & ~short_active).sum()),
        "raw_short": int((period & short_active & ~long_active).sum()),
        "strict_executable": int(len(selected)),
        "strict_executable_long": int(long_active[selected].sum()) if len(selected) else 0,
        "strict_executable_short": int(short_active[selected].sum()) if len(selected) else 0,
    }


def select_executable_positions(
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    start: str,
    end: str,
    hold_bars: int,
) -> list[int]:
    """Mirror the simulator's split-contained, non-overlapping execution rule."""
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    long_active = np.asarray(long_active, dtype=bool)
    short_active = np.asarray(short_active, dtype=bool)
    raw_positions = np.flatnonzero(period & (long_active | short_active))
    executable: list[int] = []
    next_position = 0
    for position in raw_positions:
        entry = int(position) + 1
        exit_position = entry + hold_bars
        if position < next_position or exit_position >= len(dates) or not period[exit_position]:
            continue
        if long_active[position] == short_active[position]:
            continue
        executable.append(int(position))
        next_position = exit_position + 1
    return executable


def support_passes(support: dict[str, dict[str, int]]) -> bool:
    fit = support["fit"]
    select = support["select_2023"]
    h1 = support["select_2023_h1"]
    h2 = support["select_2023_h2"]
    return bool(
        fit["strict_executable"] >= 200
        and select["strict_executable"] >= 60
        and min(h1["strict_executable"], h2["strict_executable"]) >= 25
        and min(fit["strict_executable_long"], fit["strict_executable_short"]) >= 50
        and min(select["strict_executable_long"], select["strict_executable_short"]) >= 15
    )


def event_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = int((left | right).sum())
    return float((left & right).sum() / union) if union else 0.0


def finite_spearman(left: np.ndarray, right: np.ndarray, minimum: int = 100) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < minimum:
        return float("nan")
    return float(pd.Series(left[valid]).corr(pd.Series(right[valid]), method="spearman"))


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    hold_bars: int = HOLD_BARS,
    side_cost: float = SIDE_COST,
    extremes: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, dict[str, Any]]:
    if extremes is None:
        extremes = (
            _future_extreme(market["low"].to_numpy(float), hold_bars, "min"),
            _future_extreme(market["high"].to_numpy(float), hold_bars, "max"),
        )
    return {
        window: _simulate_no_stop(
            market,
            dates,
            long_active,
            short_active,
            window=window,
            hold_bars=hold_bars,
            stride_bars=1,
            leverage=LEVERAGE,
            fee_rate=side_cost,
            slippage_rate=0.0,
            extremes=extremes,
            windows=WINDOWS,
        )
        for window in WINDOWS
    }


def admission(stats: dict[str, dict[str, Any]]) -> bool:
    return bool(
        stats["fit"]["trades"] >= 200
        and stats["select_2023"]["trades"] >= 60
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 25
        and min(stats["fit"]["longs"], stats["fit"]["shorts"]) >= 50
        and min(stats["select_2023"]["longs"], stats["select_2023"]["shorts"]) >= 15
        and stats["fit"]["return_pct"] > 0.0
        and stats["fit"]["ratio"] >= 3.0
        and stats["select_2023"]["return_pct"] > 0.0
        and stats["select_2023"]["ratio"] >= 3.0
        and stats["select_2023_h1"]["return_pct"] > 0.0
        and stats["select_2023_h2"]["return_pct"] > 0.0
    )


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
    features = build_crossmap_features(blocks)
    primary = policy_masks(features, len(market))
    controls_mask = control_masks(features, len(market))
    linear_masks = controls_mask["ordinary_linear_leadlag"]
    primary_event = primary[0] | primary[1]
    linear_event = linear_masks[0] | linear_masks[1]
    novelty = {
        "dominance_vs_linear_spearman": finite_spearman(
            features["crossmap_dominance"].to_numpy(float),
            features["linear_leadlag_dominance"].to_numpy(float),
        ),
        "dominance_vs_same_time_correlation_spearman": finite_spearman(
            features["crossmap_dominance"].to_numpy(float),
            features["price_flow_correlation"].to_numpy(float),
        ),
        "primary_vs_linear_event_jaccard": event_jaccard(primary_event, linear_event),
    }
    novelty["passed"] = bool(
        abs(novelty["dominance_vs_linear_spearman"]) < 0.50
        and abs(novelty["dominance_vs_same_time_correlation_spearman"]) < 0.50
        and novelty["primary_vs_linear_event_jaccard"] < 0.60
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
        "finite_crossmap_states": int(np.isfinite(features["crossmap_dominance"]).sum()),
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
    controls_mask["signal_delay_6h"] = shift_masks(*primary, BLOCK_BARS)
    controls_mask["signal_delay_7d"] = shift_masks(*primary, 7 * 24 * 12)
    controls = {
        name: simulate(market, dates, *masks, extremes=extremes)
        for name, masks in controls_mask.items()
    }
    cost_stress = {
        str(bp): simulate(market, dates, *primary, side_cost=bp / 10_000.0, extremes=extremes)
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
    component_names = (
        "same_events_flow_follow",
        "same_events_flow_fade",
        "same_events_price_follow",
        "same_events_price_fade",
        "ordinary_linear_leadlag",
    )
    output = {
        "protocol": {
            "source_cutoff": "returned market frame strictly before 2024-01-01",
            "source_io_disclosure": "a cutoff-crossing chunk may be physically read and immediately discarded; discarded rows never enter returned frames or computation",
            "mechanism": "nonlinear price/flow cross-map asymmetry over preceding completed six-hour blocks; positive asymmetry follows current completed flow and negative asymmetry fades it as fixed policy labels, not inferred causal regimes",
            "causal_claim_disclaimer": "cross-map asymmetry is an observable nonlinear state descriptor, not proof of causal direction in noisy financial data",
            "block": "72 completed 5m bars ending minute55 at UTC 00/06/12/18 boundaries",
            "library": f"strictly preceding {LIBRARY_BLOCKS} blocks, excluding the current block",
            "embedding": {"dimension": EMBEDDING_DIMENSION, "lag_blocks": 1, "neighbors": NEIGHBORS, "theiler_radius": THEILER_RADIUS},
            "gate": f"abs dominance > shift(1) prior {GATE_LOOKBACK}-state q{GATE_QUANTILE:.2f} of abs dominance, minimum {GATE_MIN_OBSERVATIONS}",
            "primary": "sign(crossmap dominance) times sign(current completed taker flow)",
            "entry": "signal at six-hour boundary minute00; enter minute05 open",
            "hold_bars": HOLD_BARS,
            "leverage": LEVERAGE,
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
            "finite_crossmap_states": int(np.isfinite(features["crossmap_dominance"]).sum()),
            "raw_signals": int(primary_event.sum()),
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
            and not any(control_admissions[name] for name in component_names)
        ),
        "oos_opened": False,
    }
    _print_stats("PRIMARY nonlinear cross-map flow asymmetry", primary_stats)
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
