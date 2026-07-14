"""Search within-hour execution-metronome absorption as a BTC alpha.

The signal looks for a repeated average-ticket rhythm, unusually large tickets,
coherent one-sided taker flow and poor price acceptance inside the same completed
hour.  Such a state is interpreted as algorithmic execution being absorbed by
passive liquidity and is faded at the next conservative 5-minute open.
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
RESULT_PATH = Path("results/execution_metronome_absorption_alpha_scan_2026-07-14.json")
CUTOFF = "2024-01-01"

HOUR_BARS = 12
NORMALIZATION_HOURS = 30 * 24
NORMALIZATION_MIN_HOURS = 15 * 24
FIT_TAIL = 0.90
DENOMINATOR_FLOOR_QUANTILE = 0.01
SPECTRAL_POWER_EPS = 1e-10
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
SCORE_COLUMNS = (
    "metronome_absorption_score",
    "absorption_without_regularity",
    "absorption_without_ticket_pressure",
    "metronome_flow_without_nonacceptance",
    "plain_flow_pressure",
    "regularity_only",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
        raise ValueError("metronome search requires a complete 5-minute market grid")
    return market, dates


def prior_zscore(
    values: pd.Series | np.ndarray,
    *,
    window: int = NORMALIZATION_HOURS,
    min_periods: int = NORMALIZATION_MIN_HOURS,
) -> np.ndarray:
    series = pd.Series(values, dtype=float).reset_index(drop=True)
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=min_periods).mean()
    std = prior.rolling(window, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).to_numpy(float)


def spectral_regularity(log_ticket: np.ndarray) -> float:
    """Return one minus normalized non-zero-frequency spectral entropy."""
    values = np.asarray(log_ticket, dtype=float)
    if values.ndim != 1 or len(values) != HOUR_BARS or not np.isfinite(values).all():
        return float("nan")
    time = np.arange(len(values), dtype=float)
    centered_time = time - time.mean()
    centered_value = values - values.mean()
    denominator = float(np.dot(centered_time, centered_time))
    slope = float(np.dot(centered_time, centered_value) / denominator) if denominator > 0.0 else 0.0
    residual = centered_value - slope * centered_time
    power = np.abs(np.fft.rfft(residual)[1:]) ** 2
    total = float(power.sum())
    if total <= SPECTRAL_POWER_EPS:
        return float("nan")
    probability = power / total
    entropy = -float(np.sum(probability * np.log(probability + 1e-18))) / np.log(float(len(power)))
    return float(np.clip(1.0 - entropy, 0.0, 1.0))


def _hour_metrics(
    open_price: np.ndarray,
    close: np.ndarray,
    quote: np.ndarray,
    trades: np.ndarray,
    taker_buy: np.ndarray,
) -> dict[str, float]:
    arrays = (open_price, close, quote, trades, taker_buy)
    if any(len(values) != HOUR_BARS for values in arrays):
        raise ValueError("hour metrics require exactly 12 completed 5-minute bars")
    if not all(np.isfinite(values).all() for values in arrays):
        return {}
    if (open_price <= 0.0).any() or (close <= 0.0).any() or (quote <= 0.0).any() or (trades <= 0.0).any():
        return {}
    if (taker_buy < 0.0).any() or (taker_buy > quote * (1.0 + 1e-9)).any():
        return {}

    ticket = quote / trades
    log_ticket = np.log(ticket)
    regularity = spectral_regularity(log_ticket)
    signed_flow = 2.0 * taker_buy - quote
    total_quote = float(quote.sum())
    signed_sum = float(signed_flow.sum())
    absolute_signed = float(np.abs(signed_flow).sum())
    flow_imbalance = signed_sum / total_quote
    flow_direction = float(np.sign(signed_sum))
    if absolute_signed <= 1e-12 * total_quote:
        return {}
    flow_coherence = abs(signed_sum) / absolute_signed
    flow_pressure = abs(flow_imbalance) * flow_coherence

    increments = np.empty(HOUR_BARS, dtype=float)
    increments[0] = np.log(close[0] / open_price[0])
    increments[1:] = np.log(close[1:] / close[:-1])
    path_length = float(np.abs(increments).sum())
    if path_length <= 1e-12:
        return {}
    net_return = float(increments.sum())
    signed_acceptance = (
        float(np.clip(flow_direction * net_return / path_length, -1.0, 1.0))
        if path_length > 0.0 and flow_direction != 0.0
        else 0.0
    )
    nonacceptance = 0.5 * (1.0 - signed_acceptance)
    return {
        "hour_ticket": total_quote / float(trades.sum()),
        "hour_quote": total_quote,
        "hour_trades": float(trades.sum()),
        "absolute_flow_fraction": absolute_signed / total_quote,
        "price_path_length": path_length,
        "ticket_regularity": regularity,
        "flow_imbalance": flow_imbalance,
        "flow_coherence": flow_coherence,
        "flow_pressure": flow_pressure,
        "flow_direction": flow_direction,
        "signed_price_acceptance": signed_acceptance,
        "price_nonacceptance": nonacceptance,
    }


def build_state(market: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    decision_positions = np.flatnonzero(dates.dt.minute.eq(0).to_numpy(bool))
    decision_positions = decision_positions[decision_positions >= HOUR_BARS]
    open_price = pd.to_numeric(market["open"], errors="coerce").to_numpy(float)
    close = pd.to_numeric(market["close"], errors="coerce").to_numpy(float)
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce").to_numpy(float)
    trades = pd.to_numeric(market["number_of_trades"], errors="coerce").to_numpy(float)
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce").to_numpy(float)

    hourly_rows: list[dict[str, float | int | pd.Timestamp]] = []
    for position in decision_positions:
        start = int(position) - HOUR_BARS
        metrics = _hour_metrics(
            open_price[start:position],
            close[start:position],
            quote[start:position],
            trades[start:position],
            taker_buy[start:position],
        )
        hourly_rows.append(
            {
                "position": int(position),
                "decision_time": dates.iloc[position],
                "source_time": dates.iloc[position - 1],
                **metrics,
            }
        )
    hourly = pd.DataFrame(hourly_rows)
    if hourly.empty or "hour_ticket" not in hourly:
        raise ValueError("no valid completed execution hours")
    if (pd.to_datetime(hourly["source_time"]) > pd.to_datetime(hourly["decision_time"]) - pd.Timedelta("5min")).any():
        raise RuntimeError("hourly source is not complete before decision")
    fit_time = (
        (pd.to_datetime(hourly["decision_time"]) >= pd.Timestamp(WINDOWS["fit"][0]))
        & (pd.to_datetime(hourly["decision_time"]) < pd.Timestamp(WINDOWS["fit"][1]))
    )
    denominator_columns = (
        "hour_quote",
        "hour_trades",
        "absolute_flow_fraction",
        "price_path_length",
    )
    denominator_floors: dict[str, float] = {}
    for column in denominator_columns:
        reference = pd.to_numeric(hourly.loc[fit_time, column], errors="coerce").dropna()
        if len(reference) < 1_000:
            raise ValueError(f"insufficient fit denominator observations for {column}")
        denominator_floors[column] = float(reference.quantile(DENOMINATOR_FLOOR_QUANTILE))
    hourly["ticket_level_z"] = prior_zscore(np.log(hourly["hour_ticket"]))
    ticket_pressure = np.clip(hourly["ticket_level_z"].to_numpy(float), 0.0, None)
    regularity = hourly["ticket_regularity"].to_numpy(float)
    flow_pressure = hourly["flow_pressure"].to_numpy(float)
    nonacceptance = hourly["price_nonacceptance"].to_numpy(float)
    eligible = (
        np.isfinite(ticket_pressure)
        & np.isfinite(regularity)
        & np.isfinite(flow_pressure)
        & np.isfinite(nonacceptance)
        & (hourly["flow_direction"].to_numpy(float) != 0.0)
        & (ticket_pressure > 0.0)
    )
    for column, floor in denominator_floors.items():
        eligible &= hourly[column].to_numpy(float) >= floor
    hourly["eligible"] = eligible
    hourly["metronome_absorption_score"] = np.where(
        eligible,
        regularity * ticket_pressure * flow_pressure * nonacceptance,
        np.nan,
    )
    hourly["absorption_without_regularity"] = np.where(
        eligible, ticket_pressure * flow_pressure * nonacceptance, np.nan
    )
    hourly["absorption_without_ticket_pressure"] = np.where(
        eligible, regularity * flow_pressure * nonacceptance, np.nan
    )
    hourly["metronome_flow_without_nonacceptance"] = np.where(
        eligible, regularity * ticket_pressure * flow_pressure, np.nan
    )
    hourly["plain_flow_pressure"] = np.where(eligible, flow_pressure, np.nan)
    hourly["regularity_only"] = np.where(eligible, regularity, np.nan)

    state = pd.DataFrame(
        {
            "decision": np.zeros(len(market), dtype=bool),
            "source_time": pd.Series(pd.NaT, index=np.arange(len(market)), dtype="datetime64[ns]"),
            "eligible": np.zeros(len(market), dtype=bool),
            "flow_direction": np.zeros(len(market), dtype=float),
            "hour_ticket": np.full(len(market), np.nan),
            "hour_quote": np.full(len(market), np.nan),
            "hour_trades": np.full(len(market), np.nan),
            "absolute_flow_fraction": np.full(len(market), np.nan),
            "price_path_length": np.full(len(market), np.nan),
            "ticket_level_z": np.full(len(market), np.nan),
            "ticket_regularity": np.full(len(market), np.nan),
            "flow_imbalance": np.full(len(market), np.nan),
            "flow_coherence": np.full(len(market), np.nan),
            "flow_pressure": np.full(len(market), np.nan),
            "signed_price_acceptance": np.full(len(market), np.nan),
            "price_nonacceptance": np.full(len(market), np.nan),
            **{column: np.full(len(market), np.nan) for column in SCORE_COLUMNS},
        }
    )
    positions = hourly["position"].to_numpy(np.int64)
    state.loc[positions, "decision"] = True
    for column in state.columns.difference(["decision"]):
        state.loc[positions, column] = hourly[column].to_numpy()
    state.attrs["denominator_floors"] = denominator_floors
    return state


def fit_threshold(state: pd.DataFrame, dates: pd.Series, column: str) -> float:
    start, end = WINDOWS["fit"]
    fit = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    values = pd.to_numeric(state[column], errors="coerce").to_numpy(float)
    reference = values[fit & np.isfinite(values)]
    if len(reference) < 1_000:
        raise ValueError(f"insufficient fit observations for {column}: {len(reference)}")
    return float(np.quantile(reference, FIT_TAIL))


def policy_masks(state: pd.DataFrame, column: str, threshold: float, *, fade: bool = True) -> tuple[np.ndarray, np.ndarray]:
    values = state[column].to_numpy(float)
    active = state["eligible"].to_numpy(bool) & np.isfinite(values) & (values >= threshold)
    side = state["flow_direction"].to_numpy(float)
    if fade:
        side = -side
    return active & (side > 0.0), active & (side < 0.0)


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
    raw_positions = np.flatnonzero(period & (long_active | short_active))
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
    chosen = np.asarray(executable, dtype=np.int64)
    return {
        "raw": int(len(raw_positions)),
        "raw_long": int((period & long_active & ~short_active).sum()),
        "raw_short": int((period & short_active & ~long_active).sum()),
        "strict_executable": int(len(chosen)),
        "strict_executable_long": int(long_active[chosen].sum()) if len(chosen) else 0,
        "strict_executable_short": int(short_active[chosen].sum()) if len(chosen) else 0,
    }


def support_passes(support: dict[str, dict[str, int]]) -> bool:
    fit = support["fit"]
    select = support["select_2023"]
    return bool(
        fit["strict_executable"] >= 80
        and select["strict_executable"] >= 24
        and min(
            support["select_2023_h1"]["strict_executable"],
            support["select_2023_h2"]["strict_executable"],
        ) >= 8
        and min(fit["strict_executable_long"], fit["strict_executable_short"]) >= 15
        and min(select["strict_executable_long"], select["strict_executable_short"]) >= 4
        and min(
            support["select_2023_h1"]["strict_executable_long"],
            support["select_2023_h1"]["strict_executable_short"],
            support["select_2023_h2"]["strict_executable_long"],
            support["select_2023_h2"]["strict_executable_short"],
        ) >= 4
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
        and min(
            stats["select_2023_h1"]["longs"],
            stats["select_2023_h1"]["shorts"],
            stats["select_2023_h2"]["longs"],
            stats["select_2023_h2"]["shorts"],
        ) >= 4
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


def legacy_orderflow_event_masks(
    market: pd.DataFrame,
    dates: pd.Series,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Rebuild fixed pre-existing order-flow event representatives."""
    from training.search_orderflow_trophic_campaign_alpha import campaign_signals
    from training.search_orderflow_trophic_chirp_alpha import chirp_signals
    from training.search_orderflow_trophic_succession_alpha import (
        build_profile_features,
        fit_policy_thresholds,
        sequence_signals,
    )

    profile = (12, 24, 6)
    features = build_profile_features(market, profile)
    thresholds = fit_policy_thresholds(features, dates, 0.95)
    continuation_long, continuation_short, _ = sequence_signals(
        features, thresholds, "continuation"
    )
    absorption_long, absorption_short, _ = sequence_signals(
        features, thresholds, "absorption_reversal"
    )
    campaign_long, campaign_short, _ = campaign_signals(
        continuation_long,
        continuation_short,
        lookback_bars=144,
        min_same_events=2,
        max_opposite_events=1,
    )

    chirp_profile = (6, 12, 6)
    chirp_features = build_profile_features(market, chirp_profile)
    chirp_thresholds = fit_policy_thresholds(chirp_features, dates, 0.95)
    chirp_parent_long, chirp_parent_short, _ = sequence_signals(
        chirp_features, chirp_thresholds, "continuation"
    )
    chirp_long, chirp_short, _ = chirp_signals(
        chirp_parent_long,
        chirp_parent_short,
        max_gap_bars=288,
        branch="acceleration_continuation",
    )
    return {
        "trophic_continuation_q95_12_24_6": (continuation_long, continuation_short),
        "trophic_terminal_absorption_q95_12_24_6": (absorption_long, absorption_short),
        "trophic_campaign_lb144_k2": (campaign_long, campaign_short),
        "trophic_chirp_6_12_6_gap288": (chirp_long, chirp_short),
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
    state = build_state(market, dates)
    thresholds = {column: fit_threshold(state, dates, column) for column in SCORE_COLUMNS}
    primary_long, primary_short = policy_masks(
        state, "metronome_absorption_score", thresholds["metronome_absorption_score"]
    )
    support = {
        window: support_counts(dates, primary_long, primary_short, window=window)
        for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")
    }
    preflight = {
        "support_only": True,
        "thresholds": thresholds,
        "support": support,
        "support_passed": support_passes(support),
        "valid_decision_hours": int(state["decision"].sum()),
        "eligible_hours": int(state["eligible"].sum()),
        "denominator_floor_quantile": DENOMINATOR_FLOOR_QUANTILE,
        "denominator_floors": state.attrs["denominator_floors"],
    }
    if support_only:
        return preflight

    extremes = (
        _future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"),
        _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"),
    )
    primary_stats = simulate(market, dates, primary_long, primary_short, extremes)
    control_masks = {
        column: policy_masks(state, column, thresholds[column])
        for column in SCORE_COLUMNS
        if column != "metronome_absorption_score"
    }
    control_masks["exact_direction_flip"] = (primary_short.copy(), primary_long.copy())
    for name, bars in (("signal_delay_1h", 12), ("signal_delay_24h", 288), ("signal_delay_7d", 2016)):
        control_masks[name] = (lag_mask(primary_long, bars), lag_mask(primary_short, bars))
    controls = {
        name: simulate(market, dates, long_active, short_active, extremes)
        for name, (long_active, short_active) in control_masks.items()
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
    component_names = [column for column in SCORE_COLUMNS if column != "metronome_absorption_score"]
    event_overlap = {
        name: event_jaccard(primary_events, control_masks[name][0] | control_masks[name][1])
        for name in component_names
    }
    legacy_masks = legacy_orderflow_event_masks(market, dates)
    legacy_overlap = {
        name: event_jaccard(primary_events, long_active | short_active)
        for name, (long_active, short_active) in legacy_masks.items()
    }
    primary_score = state["metronome_absorption_score"].to_numpy(float)
    feature_spearman = {
        name: finite_spearman(primary_score, state[column].to_numpy(float))
        for name, column in {
            "ticket_regularity": "ticket_regularity",
            "ticket_level_z": "ticket_level_z",
            "flow_pressure": "flow_pressure",
            "price_nonacceptance": "price_nonacceptance",
        }.items()
    }
    all_overlap = {**event_overlap, **legacy_overlap}
    novelty_pass = bool(max(all_overlap.values(), default=1.0) < 0.60)
    control_admissions = {name: admission(stats) for name, stats in controls.items()}
    no_regularity = controls["absorption_without_regularity"]
    regularity_necessary = bool(
        primary_stats["fit"]["return_pct"] > no_regularity["fit"]["return_pct"]
        and primary_stats["select_2023"]["return_pct"]
        > no_regularity["select_2023"]["return_pct"]
        and primary_stats["fit"]["ratio"] > no_regularity["fit"]["ratio"]
        and primary_stats["select_2023"]["ratio"]
        > no_regularity["select_2023"]["ratio"]
    )
    output = {
        "protocol": {
            "source_cutoff": "returned market frame hard-filtered strictly before 2024-01-01",
            "source_io_disclosure": "shared chunk parser may read and immediately discard later rows in the cutoff-crossing chunk; none enters the returned frame or computation",
            "mechanism": "within-hour average-ticket spectral regularity x positive prior ticket-size pressure x persistent taker-flow pressure x failed signed price acceptance",
            "grid": "one fixed q90 fade policy; component ablations are falsification controls, not candidate selection",
            "hour_completion": "12 bars ending minute-55; signal assigned minute-00; entry minute-05",
            "normalization": "hour-average ticket log z-score from prior 30d only, minimum 15d",
            "denominator_floors": "fit-only q01 hour quote, trade count, absolute-flow fraction and price-path length",
            "fit_tail": FIT_TAIL,
            "hold_bars": HOLD_BARS,
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "support_only_preflight": {"performed_before_returns": True, **preflight},
            "oos_opened": False,
            "contamination_note": "pre-2024 exploratory mechanism; 2023 inspected internally and 2024+ excluded",
        },
        "source": {"market_path": str(market_path), "market_sha256": _sha256(market_path)},
        "state_summary": {
            "valid_decision_hours": int(state["decision"].sum()),
            "eligible_hours": int(state["eligible"].sum()),
            "primary_raw_events": int(primary_events.sum()),
            "primary_raw_long_short": [int(primary_long.sum()), int(primary_short.sum())],
        },
        "primary": {
            "threshold": thresholds["metronome_absorption_score"],
            "stats": primary_stats,
            "prelim_admitted": admission(primary_stats),
        },
        "controls": controls,
        "control_admissions": control_admissions,
        "cost_stress": cost_stress,
        "novelty_overlap_audit": {
            "event_jaccard": event_overlap,
            "legacy_orderflow_event_jaccard": legacy_overlap,
            "feature_spearman": feature_spearman,
            "max_event_jaccard": max(all_overlap.values(), default=1.0),
            "novelty_pass": novelty_pass,
            "gate": "all fixed component-control event Jaccards below 0.60",
            "regularity_necessary": regularity_necessary,
            "regularity_gate": "primary must beat no-regularity return and ratio in both fit and 2023",
        },
        "prelim_admitted": admission(primary_stats),
        "final_admitted": bool(
            admission(primary_stats)
            and novelty_pass
            and regularity_necessary
            and not any(control_admissions.values())
        ),
        "oos_opened": False,
    }
    _print_stats("PRIMARY execution metronome absorption", primary_stats)
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
