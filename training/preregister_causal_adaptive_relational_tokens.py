"""Support-only preregistration for CARTA.

CARTA converts a broad capital-versus-crowd fracture into causal symbolic
relations observed after a fixed transition window.  This module intentionally
contains no future-return labels, action rewards, or backtest calculation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.preregister_metaorder_fragmentation_impact_curvature import (
    Config as SourceConfig,
    load_causal_frame,
    nonoverlapping_schedule,
)
from training.preregister_notional_event_topology_fracture import (
    _lagged_clean_quantile,
)


SUPPORT_CALIBRATION_GRID = (0.95, 0.96, 0.975, 0.98, 0.99)
RANK_QUANTILES = (0.20, 0.40, 0.60, 0.80)
STRUCTURE_FEATURES = (
    ("arrival_burst", "interarrival_burstiness"),
    ("notional_concentration", "event_notional_hhi"),
    ("trade_id_span", "underlying_trades_per_agg_event"),
)
RANK_FEATURES = (
    ("tension", "topology_tension"),
    ("arrival", "interarrival_burstiness"),
    ("concentration", "event_notional_hhi"),
    ("trade_span", "underlying_trades_per_agg_event"),
    ("coherence", "flow_coherence"),
    ("effective_events", "normalized_effective_event_count"),
    ("flip_rate", "sign_flip_rate"),
    ("event_imbalance", "event_imbalance_magnitude"),
    ("size_asymmetry", "size_asymmetry_magnitude"),
    ("price_response", "price_response_magnitude"),
    ("volatility", "realized_volatility_24h"),
    ("drawdown", "drawdown_from_high_24h"),
)


@dataclass(frozen=True)
class Config:
    output: str = "results/causal_adaptive_relational_tokens_support_2026-07-14.json"
    setup_tension_quantile: float = 0.975
    structure_quantile: float = 0.80
    baseline_bars: int = 8_640
    baseline_min_periods: int = 2_016
    confirmation_bars: int = 9
    hold_bars: int = 72
    minimum_agg_trade_count: int = 64
    minimum_nonoverlap_total: int = 500
    minimum_nonoverlap_per_year: int = 50
    minimum_nonoverlap_per_2023_half: int = 80
    minimum_reference_side_share: float = 0.25


def _direction_relation(value: pd.Series) -> pd.Series:
    relation = pd.Series("FLAT", index=value.index, dtype="string")
    relation.loc[value.gt(0.0)] = "WITH_REFERENCE"
    relation.loc[value.lt(0.0)] = "AGAINST_REFERENCE"
    return relation


def _lagged_rank_bucket(
    values: pd.Series,
    clean: pd.Series,
    *,
    window: int,
    minimum: int,
) -> pd.Series:
    baselines = [
        _lagged_clean_quantile(
            values,
            clean,
            quantile=quantile,
            window=window,
            minimum=minimum,
        )
        for quantile in RANK_QUANTILES
    ]
    available = pd.concat(baselines, axis=1).notna().all(axis=1)
    bucket = sum(values.ge(baseline).astype(np.int8) for baseline in baselines)
    return bucket.where(available, -1).astype(np.int8)


def _rank_transition(origin: pd.Series, current: pd.Series) -> pd.Series:
    transition = pd.Series("UNAVAILABLE", index=current.index, dtype="string")
    available = origin.ge(0) & current.ge(0)
    transition.loc[available & current.gt(origin)] = "RISE"
    transition.loc[available & current.lt(origin)] = "FALL"
    transition.loc[available & current.eq(origin)] = "STABLE"
    return transition


def _structure_bits(marks: list[pd.Series]) -> pd.Series:
    values = np.column_stack([mark.fillna(False).to_numpy(bool) for mark in marks])
    return pd.Series(
        ["".join("1" if value else "0" for value in row) for row in values],
        index=marks[0].index,
        dtype="string",
    )


def compute_carta_state(
    frame: pd.DataFrame,
    cfg: Config,
    *,
    include_tokens: bool = True,
) -> pd.DataFrame:
    if cfg.confirmation_bars < 1 or cfg.hold_bars < 1:
        raise ValueError("confirmation and hold bars must be positive")
    if not 0.0 <= cfg.setup_tension_quantile <= 1.0:
        raise ValueError("setup tension quantile must be in [0, 1]")

    clean = ~frame["quarantined"].astype(bool)
    capital_direction = pd.Series(
        np.sign(frame["signed_quote_notional"].astype(float)),
        index=frame.index,
        dtype=float,
    )
    crowd_direction = pd.Series(
        np.sign(frame["signed_event_imbalance"].astype(float)),
        index=frame.index,
        dtype=float,
    )
    immediate_price_direction = pd.Series(
        np.sign(frame["micro_log_return"].astype(float)),
        index=frame.index,
        dtype=float,
    )
    topology_tension = (
        np.sqrt(
            frame["flow_coherence"].clip(lower=0.0).astype(float)
            * frame["signed_event_imbalance"].abs().astype(float)
        )
        * frame["buy_sell_event_size_log_ratio"].abs().astype(float)
    )
    tension_baseline = _lagged_clean_quantile(
        topology_tension,
        clean,
        quantile=cfg.setup_tension_quantile,
        window=cfg.baseline_bars,
        minimum=cfg.baseline_min_periods,
    )

    structure_marks: list[pd.Series] = []
    for _, column in STRUCTURE_FEATURES:
        baseline = _lagged_clean_quantile(
            frame[column].astype(float),
            clean,
            quantile=cfg.structure_quantile,
            window=cfg.baseline_bars,
            minimum=cfg.baseline_min_periods,
        )
        structure_marks.append(frame[column].astype(float).ge(baseline))
    structure_count = sum(mark.astype(np.int8) for mark in structure_marks)

    setup = (
        clean
        & frame["agg_trade_count"].fillna(0).ge(cfg.minimum_agg_trade_count)
        & capital_direction.ne(0.0)
        & crowd_direction.ne(0.0)
        & capital_direction.eq(-crowd_direction)
        & topology_tension.ge(tension_baseline)
        & structure_count.ge(1)
    )

    confirmation = cfg.confirmation_bars
    origin_setup = setup.shift(confirmation, fill_value=False)
    reference_direction = capital_direction.shift(confirmation)
    origin_position = pd.Series(
        np.arange(len(frame), dtype=np.int64) - confirmation,
        index=frame.index,
        dtype=np.int64,
    )
    clean_transition = (
        clean.astype(np.int16)
        .rolling(confirmation + 1, min_periods=confirmation + 1)
        .sum()
        .eq(confirmation + 1)
    )
    context_bars = confirmation + 289
    clean_context = (
        clean.astype(np.int16)
        .rolling(context_bars, min_periods=context_bars)
        .sum()
        .eq(context_bars)
    )
    rank_history_ready = pd.Series(
        np.arange(len(frame))
        >= cfg.baseline_min_periods + 288 + confirmation,
        index=frame.index,
        dtype=bool,
    )
    candidate = (
        origin_setup & clean_transition & clean_context & rank_history_ready
    )
    side = reference_direction.where(candidate, 0.0).fillna(0.0).astype(np.int8)
    hold = pd.Series(0, index=frame.index, dtype=np.int16)
    hold.loc[candidate] = cfg.hold_bars
    branch = pd.Series("none", index=frame.index, dtype="string")
    branch.loc[candidate] = "carta_candidate"

    output = pd.DataFrame(
        {
            "date": frame["date"],
            "setup": setup,
            "candidate": candidate,
            "origin_position": origin_position,
            "reference_direction": reference_direction,
            "side": side,
            "hold_bars": hold,
            "branch": branch,
            "quarantined": frame["quarantined"],
            "topology_tension": topology_tension,
        }
    )
    if not include_tokens:
        return output

    post_capital = reference_direction * (
        frame["signed_quote_notional"]
        .astype(float)
        .rolling(confirmation, min_periods=confirmation)
        .sum()
    )
    post_crowd = reference_direction * (
        frame["signed_event_imbalance"]
        .astype(float)
        .rolling(confirmation, min_periods=confirmation)
        .mean()
    )
    close = frame["close"].astype(float)
    origin_close = close.shift(confirmation)
    post_price = reference_direction * np.log(close / origin_close)
    origin_price = reference_direction * immediate_price_direction.shift(confirmation)
    origin_trend_24h = reference_direction * np.log(
        close.shift(confirmation) / close.shift(confirmation + 288)
    )
    rolling_high_24h = close.rolling(288, min_periods=288).max()
    rolling_low_24h = close.rolling(288, min_periods=288).min()
    realized_volatility_24h = (
        frame["micro_log_return"].astype(float).rolling(288, min_periods=288).std()
    )
    drawdown_from_high_24h = np.log(rolling_high_24h / close)
    range_width = (rolling_high_24h - rolling_low_24h).replace(0.0, np.nan)
    range_position = (close - rolling_low_24h).divide(range_width)

    transition_high = frame["high"].astype(float).rolling(
        confirmation, min_periods=confirmation
    ).max()
    transition_low = frame["low"].astype(float).rolling(
        confirmation, min_periods=confirmation
    ).min()
    favorable = pd.Series(
        np.where(
            reference_direction.gt(0.0),
            np.log(transition_high / origin_close),
            np.log(origin_close / transition_low),
        ),
        index=frame.index,
        dtype=float,
    )
    adverse = pd.Series(
        np.where(
            reference_direction.gt(0.0),
            np.log(origin_close / transition_low),
            np.log(transition_high / origin_close),
        ),
        index=frame.index,
        dtype=float,
    )
    path_balance = pd.Series("BALANCED", index=frame.index, dtype="string")
    path_balance.loc[favorable.gt(adverse)] = "REFERENCE_DOMINANT"
    path_balance.loc[favorable.lt(adverse)] = "OPPOSITE_DOMINANT"

    output["reference_side_token"] = pd.Series(
        np.where(reference_direction.gt(0.0), "LONG", "SHORT"),
        index=frame.index,
        dtype="string",
    )
    output["origin_price_relation"] = _direction_relation(origin_price)
    output["capital_transition_relation"] = _direction_relation(post_capital)
    output["crowd_transition_relation"] = _direction_relation(post_crowd)
    output["price_transition_relation"] = _direction_relation(post_price)
    output["origin_trend_24h_relation"] = _direction_relation(origin_trend_24h)
    output["transition_path_balance"] = path_balance
    output["range_location_24h"] = pd.Series(
        np.where(
            range_position.lt(1.0 / 3.0),
            "LOWER_THIRD",
            np.where(range_position.gt(2.0 / 3.0), "UPPER_THIRD", "MIDDLE_THIRD"),
        ),
        index=frame.index,
        dtype="string",
    )
    origin_bits = _structure_bits(
        [mark.shift(confirmation, fill_value=False) for mark in structure_marks]
    )
    signal_bits = _structure_bits(structure_marks)
    output["origin_structure_bits"] = origin_bits
    output["signal_structure_bits"] = signal_bits

    rank_inputs = {
        "topology_tension": topology_tension,
        "interarrival_burstiness": frame["interarrival_burstiness"].astype(float),
        "event_notional_hhi": frame["event_notional_hhi"].astype(float),
        "underlying_trades_per_agg_event": frame[
            "underlying_trades_per_agg_event"
        ].astype(float),
        "flow_coherence": frame["flow_coherence"].astype(float),
        "normalized_effective_event_count": frame[
            "normalized_effective_event_count"
        ].astype(float),
        "sign_flip_rate": frame["sign_flip_rate"].astype(float),
        "event_imbalance_magnitude": frame["signed_event_imbalance"].abs().astype(float),
        "size_asymmetry_magnitude": frame[
            "buy_sell_event_size_log_ratio"
        ].abs().astype(float),
        "price_response_magnitude": frame["signed_price_response"].abs().astype(float),
        "realized_volatility_24h": realized_volatility_24h,
        "drawdown_from_high_24h": drawdown_from_high_24h,
    }
    for token_name, column in RANK_FEATURES:
        rank = _lagged_rank_bucket(
            rank_inputs[column],
            clean,
            window=cfg.baseline_bars,
            minimum=cfg.baseline_min_periods,
        )
        origin_rank = rank.shift(confirmation, fill_value=-1).astype(np.int8)
        output[f"{token_name}_rank"] = rank
        output[f"{token_name}_transition"] = _rank_transition(origin_rank, rank)

    alignment = sum(
        output[column].eq("WITH_REFERENCE").astype(np.int8)
        for column in (
            "capital_transition_relation",
            "crowd_transition_relation",
            "price_transition_relation",
        )
    )
    output["transition_alignment_count"] = alignment.astype(np.int8)
    output["position_state"] = "FLAT"
    return output


TOKEN_COLUMNS = (
    "reference_side_token",
    "origin_price_relation",
    "capital_transition_relation",
    "crowd_transition_relation",
    "price_transition_relation",
    "origin_trend_24h_relation",
    "transition_path_balance",
    "range_location_24h",
    "origin_structure_bits",
    "signal_structure_bits",
    "transition_alignment_count",
    "position_state",
) + tuple(
    column
    for token_name, _ in RANK_FEATURES
    for column in (f"{token_name}_rank", f"{token_name}_transition")
)


def relational_tokens(row: pd.Series) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for column in TOKEN_COLUMNS:
        value = row[column]
        if column.endswith("_rank") or column == "transition_alignment_count":
            value = str(int(value))
        else:
            value = str(value)
        tokens[column] = value
    return tokens


def nonoverlapping_carta_schedule(
    state: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp = "2024-01-01",
) -> pd.DataFrame:
    start_timestamp = frame["date"].min() if start is None else pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    period = frame["date"].ge(start_timestamp) & frame["date"].lt(end_timestamp)
    origin = state["origin_position"].to_numpy(np.int64)
    valid_origin = np.zeros(len(state), dtype=bool)
    in_range = (origin >= 0) & (origin < len(frame))
    valid_origin[in_range] = period.to_numpy(bool)[origin[in_range]]
    eligible = state.copy()
    eligible.loc[~valid_origin, "side"] = 0
    schedule = nonoverlapping_schedule(
        eligible,
        frame,
        start=start_timestamp,
        end=end_timestamp,
    )
    if not schedule.empty:
        schedule["origin_position"] = [
            int(state.loc[position, "origin_position"])
            for position in schedule["signal_position"]
        ]
    return schedule


def _period_count(
    state: pd.DataFrame,
    frame: pd.DataFrame,
    start: str,
    end: str,
) -> int:
    return len(
        nonoverlapping_carta_schedule(state, frame, start=start, end=end)
    )


def support_summary(
    state: pd.DataFrame,
    frame: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    annual_schedules = {
        str(year): nonoverlapping_carta_schedule(
            state,
            frame,
            start=f"{year}-01-01",
            end=f"{year + 1}-01-01",
        )
        for year in range(2020, 2024)
    }
    schedule = pd.concat(annual_schedules.values(), ignore_index=True)
    years = {year: len(period) for year, period in annual_schedules.items()}
    h1 = _period_count(state, frame, "2023-01-01", "2023-07-01")
    h2 = _period_count(state, frame, "2023-07-01", "2024-01-01")
    total = len(schedule)
    long_share = float(schedule["side"].gt(0).mean()) if total else 0.0
    short_share = float(schedule["side"].lt(0).mean()) if total else 0.0
    passes = (
        total >= cfg.minimum_nonoverlap_total
        and all(value >= cfg.minimum_nonoverlap_per_year for value in years.values())
        and h1 >= cfg.minimum_nonoverlap_per_2023_half
        and h2 >= cfg.minimum_nonoverlap_per_2023_half
        and min(long_share, short_share) >= cfg.minimum_reference_side_share
    )
    return {
        "nonoverlap_total": int(total),
        "by_year": years,
        "2023_h1": int(h1),
        "2023_h2": int(h2),
        "reference_long_share": long_share,
        "reference_short_share": short_share,
        "passes_support": bool(passes),
    }


def _selected_support_quantile(trials: list[dict[str, Any]]) -> float | None:
    passing = [
        float(trial["setup_tension_quantile"])
        for trial in trials
        if trial["passes_support"]
    ]
    return max(passing) if passing else None


def _token_vocabulary(
    state: pd.DataFrame,
    schedule: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    rows = state.loc[schedule["signal_position"], TOKEN_COLUMNS]
    return {
        column: dict(sorted(Counter(str(value) for value in rows[column]).items()))
        for column in TOKEN_COLUMNS
    }


def _token_signatures(
    state: pd.DataFrame,
    schedule: pd.DataFrame,
) -> list[str]:
    return [
        json.dumps(
            relational_tokens(state.loc[position]),
            sort_keys=True,
            separators=(",", ":"),
        )
        for position in schedule["signal_position"]
    ]


def signature_support(
    state: pd.DataFrame,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    train = nonoverlapping_carta_schedule(
        state, frame, start="2020-01-01", end="2023-01-01"
    )
    selection = nonoverlapping_carta_schedule(
        state, frame, start="2023-01-01", end="2024-01-01"
    )
    train_signatures = _token_signatures(state, train)
    selection_signatures = _token_signatures(state, selection)
    train_unique = set(train_signatures)
    selection_unique = set(selection_signatures)
    selection_unseen = [
        signature for signature in selection_signatures if signature not in train_unique
    ]
    return {
        "train_2020_2022_candidates": len(train_signatures),
        "train_unique_signatures": len(train_unique),
        "select_2023_candidates": len(selection_signatures),
        "select_unique_signatures": len(selection_unique),
        "select_exact_signature_seen_in_train": len(selection_signatures)
        - len(selection_unseen),
        "select_unseen_signature_count": len(selection_unseen),
        "select_unseen_signature_share": (
            len(selection_unseen) / len(selection_signatures)
            if selection_signatures
            else 0.0
        ),
    }


def run_support(cfg: Config) -> dict[str, Any]:
    frame, source = load_causal_frame(SourceConfig())
    state = compute_carta_state(frame, cfg, include_tokens=True)
    support = support_summary(state, frame, cfg)
    schedule = pd.concat(
        [
            nonoverlapping_carta_schedule(
                state,
                frame,
                start=f"{year}-01-01",
                end=f"{year + 1}-01-01",
            )
            for year in range(2020, 2024)
        ],
        ignore_index=True,
    )

    trials: list[dict[str, Any]] = []
    for quantile in SUPPORT_CALIBRATION_GRID:
        trial_cfg = replace(cfg, setup_tension_quantile=quantile)
        trial_state = (
            state
            if quantile == cfg.setup_tension_quantile
            else compute_carta_state(frame, trial_cfg, include_tokens=False)
        )
        trial_support = support_summary(trial_state, frame, trial_cfg)
        trials.append(
            {
                "setup_tension_quantile": quantile,
                **trial_support,
            }
        )
    selected = _selected_support_quantile(trials)
    if selected != cfg.setup_tension_quantile:
        raise ValueError("configured CARTA quantile violates support stopping rule")

    return {
        "protocol": {
            "name": "CARTA — Causal Adaptive Relational Token Abstainer",
            "support_only": True,
            "outcomes_opened_for_carta": False,
            "selection_end_exclusive": "2024-01-01 00:00:00",
            "candidate_clock": "fixed before policy action; abstention does not release skipped candidates",
            "signal_availability": "after 9 completed 5m transition bars; enter next 5m open",
            "actions": ["ABSTAIN", "FOLLOW", "FADE"],
            "raw_timestamps_in_model_prompt": False,
            "raw_numeric_values_in_model_prompt": False,
            "future_rewards_in_model_prompt": False,
            "source_gap_policy": "verified full-day/missing-slot plus 24-bar quarantine",
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
        },
        "config": asdict(cfg),
        "source": source,
        "support_calibration": {
            "outcomes_opened_for_carta": False,
            "tested_setup_tension_quantiles": list(SUPPORT_CALIBRATION_GRID),
            "all_other_parameters_fixed": True,
            "stopping_rule": "highest tested quantile passing every frozen support floor",
            "selected_setup_tension_quantile": selected,
            "further_support_repairs_allowed": False,
            "trials": trials,
        },
        "setup_count": int(state["setup"].sum()),
        "raw_candidate_count": int(state["candidate"].sum()),
        "support": support,
        "token_schema": list(TOKEN_COLUMNS),
        "observed_token_vocabulary": _token_vocabulary(state, schedule),
        "signature_support": signature_support(state, frame),
        "all_support_gates_pass": bool(support["passes_support"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Config.output)
    args = parser.parse_args()
    cfg = Config(output=args.output)
    result = run_support(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "outcomes_opened_for_carta": result["protocol"][
                    "outcomes_opened_for_carta"
                ],
                "setup_count": result["setup_count"],
                "raw_candidate_count": result["raw_candidate_count"],
                **result["support"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
