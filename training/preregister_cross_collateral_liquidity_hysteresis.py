"""Support-only preregistration for CCLH persistent liquidity transitions.

CCLH derives direction from a persistent, dimensionless difference between
Binance coin-margined and USD-margined full displayed-depth geometry. It does
not use a preceding price shock and contains no future-return calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_cross_collateral_liquidity_void_refill as clvr
from training import preregister_cross_collateral_liquidity_vacuum as clv
from training.preregister_metaorder_fragmentation_impact_curvature import (
    nonoverlapping_schedule,
)


PREREGISTRATION_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_hysteresis.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cross-collateral-liquidity-hysteresis-preregistration-2026-07-14.md"
)
FEATURE_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_void_refill.py"
)
CLV_SOURCE = Path("training/preregister_cross_collateral_liquidity_vacuum.py")
CLV_SUPPORT = Path(
    "results/cross_collateral_liquidity_vacuum_support_2026-07-14.json"
)
SCHEDULER_SOURCE = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)


@dataclass(frozen=True)
class Config:
    depth_manifest: str = (
        "results/binance_cross_collateral_book_depth_btc_2023_manifest.json"
    )
    market_manifest: str = (
        "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
    )
    output: str = (
        "results/cross_collateral_liquidity_hysteresis_support_2026-07-14.json"
    )
    robust_baseline_bars: int = 8_640
    robust_min_periods: int = 2_016
    entry_z: float = 0.50
    exit_z: float = 0.25
    confirmation_bars: int = 12
    exit_confirmation_bars: int = 12
    hold_bars: int = 144
    minimum_nonoverlap_total: int = 120
    minimum_nonoverlap_per_half: int = 45
    minimum_nonoverlap_per_quarter: int = 20
    minimum_side_share: float = 0.35
    maximum_quarter_share: float = 0.40
    clv_overlap_tolerance_bars: int = 12
    maximum_clv_event_jaccard: float = 0.35


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _log_depth_slope(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("depth slope requires exactly five cumulative levels")
    log_values = np.log(values.astype(float))
    x = np.log(np.arange(1.0, 6.0))
    centered_x = x - x.mean()
    centered_values = log_values - log_values.mean(axis=1, keepdims=True)
    return centered_values.dot(centered_x) / centered_x.dot(centered_x)


def cross_collateral_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    geometry: dict[str, np.ndarray] = {}
    for venue in ("um", "cm"):
        bids = frame[
            [f"{venue}_depth_m{distance}" for distance in range(1, 6)]
        ].to_numpy(float)
        asks = frame[
            [f"{venue}_depth_p{distance}" for distance in range(1, 6)]
        ].to_numpy(float)
        if np.any((bids <= 0.0) & np.isfinite(bids)) or np.any(
            (asks <= 0.0) & np.isfinite(asks)
        ):
            raise ValueError("CCLH depth must be positive when finite")
        geometry[f"{venue}_pressure"] = np.mean(
            np.log(bids) - np.log(asks),
            axis=1,
        )
        geometry[f"{venue}_elasticity"] = (
            _log_depth_slope(asks) - _log_depth_slope(bids)
        )
    return pd.DataFrame(
        {
            "cross_pressure": (
                geometry["cm_pressure"] - geometry["um_pressure"]
            ),
            "cross_elasticity": (
                geometry["cm_elasticity"] - geometry["um_elasticity"]
            ),
        }
    )


def hysteresis_state_machine(
    pressure_z: pd.Series,
    elasticity_z: pd.Series,
    clean: pd.Series,
    cfg: Config,
) -> pd.DataFrame:
    if cfg.entry_z <= 0.0 or not 0.0 <= cfg.exit_z < cfg.entry_z:
        raise ValueError("CCLH hysteresis thresholds are invalid")
    if cfg.confirmation_bars < 1 or cfg.exit_confirmation_bars < 1:
        raise ValueError("CCLH confirmation lengths must be positive")
    if not (len(pressure_z) == len(elasticity_z) == len(clean)):
        raise ValueError("CCLH state inputs must have equal length")

    clean_values = clean.astype(bool).to_numpy()
    pressure = pressure_z.astype(float).to_numpy()
    elasticity = elasticity_z.astype(float).to_numpy()
    observed = clean_values & np.isfinite(pressure) & np.isfinite(elasticity)
    provisional = np.zeros(len(pressure), dtype=np.int8)
    provisional[
        observed & (pressure >= cfg.entry_z) & (elasticity >= cfg.entry_z)
    ] = 1
    provisional[
        observed & (pressure <= -cfg.entry_z) & (elasticity <= -cfg.entry_z)
    ] = -1
    weak = (
        ~observed
        | (np.abs(pressure) < cfg.exit_z)
        | (np.abs(elasticity) < cfg.exit_z)
    )

    active_values = np.zeros(len(pressure), dtype=np.int8)
    event_values = np.zeros(len(pressure), dtype=np.int8)
    confirmation_values = np.zeros(len(pressure), dtype=np.int32)
    weak_values = np.zeros(len(pressure), dtype=np.int32)
    active = 0
    confirmation_side = 0
    confirmation_count = 0
    weak_count = 0
    for position, (state, is_weak) in enumerate(zip(provisional, weak)):
        if state == 0:
            confirmation_side = 0
            confirmation_count = 0
        elif state == confirmation_side:
            confirmation_count += 1
        else:
            confirmation_side = int(state)
            confirmation_count = 1

        weak_count = weak_count + 1 if is_weak else 0
        event = 0
        if active == 0 and confirmation_count >= cfg.confirmation_bars:
            active = confirmation_side
            event = active
            weak_count = 0
        elif (
            active != 0
            and confirmation_side == -active
            and confirmation_count >= cfg.confirmation_bars
        ):
            active = confirmation_side
            event = active
            weak_count = 0
        elif active != 0 and weak_count >= cfg.exit_confirmation_bars:
            active = 0
            weak_count = 0

        active_values[position] = active
        event_values[position] = event
        confirmation_values[position] = confirmation_count
        weak_values[position] = weak_count

    return pd.DataFrame(
        {
            "provisional_state": provisional,
            "active_state": active_values,
            "event_side": event_values,
            "confirmation_count": confirmation_values,
            "weak_count": weak_values,
        },
        index=pressure_z.index,
    )


def build_signal(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    geometry = cross_collateral_geometry(frame)
    clean = frame["source_complete"].astype(bool)
    pressure_z = clvr.lagged_robust_zscore(
        geometry["cross_pressure"].where(clean),
        window=cfg.robust_baseline_bars,
        minimum=cfg.robust_min_periods,
    )
    elasticity_z = clvr.lagged_robust_zscore(
        geometry["cross_elasticity"].where(clean),
        window=cfg.robust_baseline_bars,
        minimum=cfg.robust_min_periods,
    )
    state = hysteresis_state_machine(pressure_z, elasticity_z, clean, cfg)
    side = state["event_side"].astype(np.int8)
    candidate = side.ne(0)
    branch = pd.Series("none", index=frame.index, dtype="string")
    branch.loc[side.gt(0)] = "bullish_hysteresis"
    branch.loc[side.lt(0)] = "bearish_hysteresis"
    return pd.DataFrame(
        {
            "date": frame["date"],
            "candidate": candidate,
            "cross_pressure": geometry["cross_pressure"],
            "cross_elasticity": geometry["cross_elasticity"],
            "pressure_z": pressure_z,
            "elasticity_z": elasticity_z,
            **state.to_dict("series"),
            "side": side,
            "branch": branch,
            "hold_bars": np.where(candidate, cfg.hold_bars, 0).astype(
                np.int16
            ),
        }
    )


def support_summary(
    signal: pd.DataFrame,
    market: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    periods = {
        "q1": ("2023-01-01", "2023-04-01"),
        "q2": ("2023-04-01", "2023-07-01"),
        "q3": ("2023-07-01", "2023-10-01"),
        "q4": ("2023-10-01", "2024-01-01"),
    }
    quarterly = {
        name: nonoverlapping_schedule(signal, market, start=start, end=end)
        for name, (start, end) in periods.items()
    }
    schedule = pd.concat(quarterly.values(), ignore_index=True)
    h1 = nonoverlapping_schedule(
        signal,
        market,
        start="2023-01-01",
        end="2023-07-01",
    )
    h2 = nonoverlapping_schedule(
        signal,
        market,
        start="2023-07-01",
        end="2024-01-01",
    )
    total = len(schedule)
    by_quarter = {name: len(rows) for name, rows in quarterly.items()}
    long_share = float(schedule["side"].gt(0).mean()) if total else 0.0
    short_share = float(schedule["side"].lt(0).mean()) if total else 0.0
    maximum_observed_quarter_share = (
        max(by_quarter.values()) / total if total else 1.0
    )
    passes = (
        total >= cfg.minimum_nonoverlap_total
        and len(h1) >= cfg.minimum_nonoverlap_per_half
        and len(h2) >= cfg.minimum_nonoverlap_per_half
        and all(
            value >= cfg.minimum_nonoverlap_per_quarter
            for value in by_quarter.values()
        )
        and min(long_share, short_share) >= cfg.minimum_side_share
        and maximum_observed_quarter_share <= cfg.maximum_quarter_share
    )
    return {
        "nonoverlap_total": int(total),
        "by_quarter": by_quarter,
        "h1": int(len(h1)),
        "h2": int(len(h2)),
        "long_share": long_share,
        "short_share": short_share,
        "maximum_observed_quarter_share": float(
            maximum_observed_quarter_share
        ),
        "passes_support": bool(passes),
    }


def _quarterly_schedule(
    signal: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    return pd.concat(
        [
            nonoverlapping_schedule(signal, market, start=start, end=end)
            for start, end in (
                ("2023-01-01", "2023-04-01"),
                ("2023-04-01", "2023-07-01"),
                ("2023-07-01", "2023-10-01"),
                ("2023-10-01", "2024-01-01"),
            )
        ],
        ignore_index=True,
    )


def tolerant_event_jaccard(
    first_positions: list[int],
    second_positions: list[int],
    *,
    tolerance_bars: int,
) -> dict[str, Any]:
    if tolerance_bars < 0:
        raise ValueError("event-overlap tolerance must be non-negative")
    first = sorted(int(value) for value in first_positions)
    second = sorted(int(value) for value in second_positions)
    first_cursor = 0
    second_cursor = 0
    matches = 0
    while first_cursor < len(first) and second_cursor < len(second):
        left = first[first_cursor]
        right = second[second_cursor]
        if abs(left - right) <= tolerance_bars:
            matches += 1
            first_cursor += 1
            second_cursor += 1
        elif left < right:
            first_cursor += 1
        else:
            second_cursor += 1
    union = len(first) + len(second) - matches
    return {
        "first_event_count": int(len(first)),
        "second_event_count": int(len(second)),
        "matched_event_count": int(matches),
        "tolerance_bars": int(tolerance_bars),
        "jaccard": float(matches / union) if union else 1.0,
    }


def run_support(cfg: Config) -> dict[str, Any]:
    market, source = clvr.load_sources(cfg)
    signal = build_signal(market, cfg)
    support = support_summary(signal, market, cfg)
    schedule = _quarterly_schedule(signal, market)

    clv_cfg = clv.Config(
        depth_manifest=cfg.depth_manifest,
        market_manifest=cfg.market_manifest,
    )
    clv_features = clvr.build_features(market, clv_cfg)
    clv_signal = clv.classify_vacuum(clv_features, clv_cfg)
    clv_schedule = _quarterly_schedule(clv_signal, market)
    overlap = tolerant_event_jaccard(
        schedule["signal_position"].astype(int).tolist(),
        clv_schedule["signal_position"].astype(int).tolist(),
        tolerance_bars=cfg.clv_overlap_tolerance_bars,
    )
    overlap["maximum_allowed_jaccard"] = cfg.maximum_clv_event_jaccard
    overlap["passes_independence"] = bool(
        overlap["jaccard"] <= cfg.maximum_clv_event_jaccard
    )
    passes_all = support["passes_support"] and overlap["passes_independence"]
    return {
        "protocol": {
            "name": "CCLH — Cross-Collateral Liquidity Hysteresis",
            "support_only": True,
            "outcomes_opened_for_cclh": False,
            "support_rejected": not passes_all,
            "selection_end_exclusive": "2024-01-01 00:00:00",
            "event_clock": (
                "confirmed transition into a persistent cross-collateral "
                "full-depth geometry state"
            ),
            "signal_availability": (
                "current 5m depth median complete; enter next 5m open"
            ),
            "action_rule": (
                "positive pressure/elasticity state long; negative state short"
            ),
            "candidate_clock": (
                "one event on flat-to-active or confirmed opposite-state flip"
            ),
            "holding_rule": "144 completed 5m bars; scheduled-open exit",
            "support_parameters_searched": False,
            "source_gap_policy": (
                "missing current depth is unavailable state; future depth "
                "gaps do not cancel an already entered trade"
            ),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
        },
        "config": asdict(cfg),
        "frozen_artifacts": {
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": _sha256(PREREGISTRATION_SOURCE),
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": _sha256(
                PREREGISTRATION_DOCUMENT
            ),
            "feature_source": str(FEATURE_SOURCE),
            "feature_source_sha256": _sha256(FEATURE_SOURCE),
            "clv_source": str(CLV_SOURCE),
            "clv_source_sha256": _sha256(CLV_SOURCE),
            "clv_support": str(CLV_SUPPORT),
            "clv_support_sha256": _sha256(CLV_SUPPORT),
            "scheduler_source": str(SCHEDULER_SOURCE),
            "scheduler_source_sha256": _sha256(SCHEDULER_SOURCE),
            "depth_manifest_sha256": _sha256(cfg.depth_manifest),
            "market_manifest_sha256": _sha256(cfg.market_manifest),
        },
        "source": source,
        "feature": {
            "provisional_bullish_rows": int(
                signal["provisional_state"].gt(0).sum()
            ),
            "provisional_bearish_rows": int(
                signal["provisional_state"].lt(0).sum()
            ),
            "raw_transition_count": int(signal["candidate"].sum()),
            "standardization": (
                "strictly lagged rolling median and recursive MAD; "
                "clip [-12, 12]"
            ),
        },
        "support_calibration": {
            "outcomes_opened_for_cclh": False,
            "parameters_searched": False,
            "all_parameters_fixed": True,
            "further_support_repairs_allowed": False,
        },
        "raw_candidate_count": int(signal["candidate"].sum()),
        "scheduled_side_counts": {
            "long": int(schedule["side"].gt(0).sum()),
            "short": int(schedule["side"].lt(0).sum()),
        },
        "scheduled_branch_counts": {
            name: int(value)
            for name, value in schedule["branch"].value_counts().items()
        },
        "clv_event_overlap": overlap,
        "support": support,
        "all_support_gates_pass": bool(passes_all),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Config.output)
    args = parser.parse_args()
    result = run_support(Config(output=args.output))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "outcomes_opened_for_cclh": False,
                "support_rejected": result["protocol"]["support_rejected"],
                "support": result["support"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
