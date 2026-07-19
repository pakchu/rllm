"""Outcome-blind support preregistration for RQCI-24.

RQCI-24 trades the innovation in outer-versus-inner directional average-quote
curvature after removing its strictly-prior linear response to the quote-center
move.  This module never loads OHLC, funding, returns, PnL, equity, or any
post-2023 source row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training import preregister_residual_notional_centroid_migration as shared


POLICY_ID = "RQCI-24"
SHARED_SOURCE = Path(
    "training/preregister_residual_notional_centroid_migration.py"
)
SHARED_SOURCE_SHA256 = (
    "733ef4c3aaa823f19c8fe9303d3405def0c86f593c35bb2556a69edc3f67ad6f"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_residual_quote_curvature_impulse.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/residual-quote-curvature-impulse-preregistration-2026-07-20.md"
)
SOURCE_DECISION_DOCUMENT = Path(
    "docs/rqci-source-mechanism-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "fe92ac70536171e62280bba8fac1cf5f950332cb02f7f77106e242d8f0595476"
)


@dataclass(frozen=True)
class Config:
    support_output: str = (
        "results/residual_quote_curvature_impulse_support_2026-07-20.json"
    )
    event_clock_output: str = (
        "results/residual_quote_curvature_impulse_event_clock_2026-07-20.json"
    )
    impulse_bars: int = 6
    baseline_window_bars: int = 8_640
    baseline_minimum_bars: int = 4_032
    threshold_quantiles: tuple[float, ...] = (
        0.995,
        0.99,
        0.985,
        0.975,
        0.95,
    )
    quiet_center_quantile: float = 0.50
    minimum_residual_dominance: float = 0.25
    hold_bars: int = 24
    minimum_nonoverlap_total: int = 180
    minimum_nonoverlap_per_half: int = 70
    minimum_nonoverlap_per_quarter: int = 30
    minimum_side_share: float = 0.35
    maximum_quarter_share: float = 0.40
    overlap_tolerance_bars: int = 12
    maximum_comparator_jaccard: float = 0.35
    maximum_synthetic_false_events: int = 0


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_config(cfg: Config) -> None:
    expected = Config(
        support_output=cfg.support_output,
        event_clock_output=cfg.event_clock_output,
    )
    if cfg != expected:
        raise ValueError("RQCI-24 signal and support configuration is frozen")
    if cfg.threshold_quantiles != tuple(
        sorted(cfg.threshold_quantiles, reverse=True)
    ):
        raise ValueError("RQCI support quantiles must be strictest first")
    if sha256_file(SHARED_SOURCE) != SHARED_SOURCE_SHA256:
        raise ValueError("RQCI shared causal utility source hash mismatch")
    if sha256_file(SOURCE_DECISION_DOCUMENT) != SOURCE_DECISION_SHA256:
        raise ValueError("RQCI source decision hash mismatch")


def curvature(skew: pd.DataFrame) -> pd.Series:
    """Outer directional slope minus inner directional slope."""
    return cast(
        pd.Series,
        (skew["skew_5_median"] - skew["skew_4_median"])
        - (skew["skew_3_median"] - skew["skew_2_median"]),
    )


def curvature_features(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    complete = cast(pd.Series, frame["source_complete"]).astype(bool)
    streak = (
        complete.rolling(
            cfg.impulse_bars + 1,
            min_periods=cfg.impulse_bars + 1,
        )
        .sum()
        .eq(cfg.impulse_bars + 1)
    )
    skew = cast(pd.DataFrame, frame[list(shared.SKEW_COLUMNS)]).where(complete)
    level = curvature(skew)
    raw_impulse = (level - level.shift(cfg.impulse_bars)).where(streak)
    center = cast(pd.Series, frame["center_quote_median"]).where(complete)
    center_move = cast(
        pd.Series,
        np.log(center / center.shift(cfg.impulse_bars)).where(streak),
    )
    beta = shared.prior_beta(
        raw_impulse,
        center_move,
        window=cfg.baseline_window_bars,
        minimum=cfg.baseline_minimum_bars,
    )
    residual = raw_impulse - beta * center_move
    dominance = residual.abs() / raw_impulse.abs().replace(0.0, np.nan)
    quiet_center_threshold = shared.prior_quantile(
        center_move.abs(),
        quantile=cfg.quiet_center_quantile,
        window=cfg.baseline_window_bars,
        minimum=cfg.baseline_minimum_bars,
    )
    return pd.DataFrame(
        {
            "source_streak_complete": streak,
            "curvature": level,
            "raw_impulse": raw_impulse,
            "center_move_30m": center_move,
            "center_beta": beta,
            "residual_impulse": residual,
            "residual_dominance": dominance,
            "quiet_center_threshold": quiet_center_threshold,
        }
    )


def build_signal(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    cfg: Config,
    *,
    quantile: float,
) -> pd.DataFrame:
    residual = cast(pd.Series, features["residual_impulse"])
    threshold = shared.prior_quantile(
        residual.abs(),
        quantile=quantile,
        window=cfg.baseline_window_bars,
        minimum=cfg.baseline_minimum_bars,
    )
    above = (
        residual.ne(0.0)
        & features["residual_dominance"].ge(cfg.minimum_residual_dominance)
        & features["center_move_30m"].abs().le(
            features["quiet_center_threshold"]
        )
        & residual.abs().ge(threshold)
    )
    crossed_from_below = residual.abs().shift(1).lt(threshold.shift(1))
    candidate = above & crossed_from_below
    side = pd.Series(0, index=frame.index, dtype=np.int8)
    side.loc[candidate & residual.gt(0.0)] = 1
    side.loc[candidate & residual.lt(0.0)] = -1
    branch = pd.Series("none", index=frame.index, dtype="string")
    branch.loc[side.gt(0)] = "outer_ask_curvature_impulse"
    branch.loc[side.lt(0)] = "outer_bid_curvature_impulse"
    return pd.DataFrame(
        {
            "date": frame["date"],
            "candidate": side.ne(0),
            "threshold": threshold,
            "side": side,
            "branch": branch,
            "hold_bars": np.where(side.ne(0), cfg.hold_bars, 0).astype(
                np.int16
            ),
        }
    )


def support_summary(schedule: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    by_quarter = {
        quarter: int(schedule["quarter"].eq(quarter).sum())
        if not schedule.empty
        else 0
        for quarter in shared.QUARTERS
    }
    total = int(len(schedule))
    h1 = by_quarter["q1"] + by_quarter["q2"]
    h2 = by_quarter["q3"] + by_quarter["q4"]
    long_share = float(schedule["side"].gt(0).mean()) if total else 0.0
    short_share = float(schedule["side"].lt(0).mean()) if total else 0.0
    maximum_quarter_share = max(by_quarter.values()) / total if total else 1.0
    passes = (
        total >= cfg.minimum_nonoverlap_total
        and h1 >= cfg.minimum_nonoverlap_per_half
        and h2 >= cfg.minimum_nonoverlap_per_half
        and all(
            count >= cfg.minimum_nonoverlap_per_quarter
            for count in by_quarter.values()
        )
        and min(long_share, short_share) >= cfg.minimum_side_share
        and maximum_quarter_share <= cfg.maximum_quarter_share
    )
    return {
        "nonoverlap_total": total,
        "by_quarter": by_quarter,
        "h1": int(h1),
        "h2": int(h2),
        "long_share": long_share,
        "short_share": short_share,
        "maximum_quarter_share": float(maximum_quarter_share),
        "passes_incidence": bool(passes),
    }


def synthetic_control(cfg: Config) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    passes = True
    for name, frame in shared.synthetic_null_suite().items():
        features = curvature_features(frame, cfg)
        trials: dict[str, Any] = {}
        for quantile in cfg.threshold_quantiles:
            signal = build_signal(frame, features, cfg, quantile=quantile)
            schedule = shared.quarterly_nonoverlap_schedule(signal, frame)
            raw_count = int(signal["candidate"].sum())
            count = int(len(schedule))
            trials[str(quantile)] = {
                "raw_events": raw_count,
                "nonoverlap_events": count,
            }
            passes &= (
                raw_count <= cfg.maximum_synthetic_false_events
                and count <= cfg.maximum_synthetic_false_events
            )
        scenarios[name] = trials
    return {
        "mechanism": (
            "five fixed absolute-book moving-band nulls; only radial sampling "
            "changes while the underlying book remains stationary"
        ),
        "maximum_allowed_nonoverlap_events_each_quantile": (
            cfg.maximum_synthetic_false_events
        ),
        "scenarios": scenarios,
        "passes": bool(passes),
    }


def event_records(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    rows = cast(
        list[dict[str, Any]],
        schedule[
            [
                "quarter",
                "signal_position",
                "entry_position",
                "exit_position",
                "signal_date",
                "entry_date",
                "exit_date",
                "side",
                "branch",
                "hold_bars",
            ]
        ].to_dict(orient="records"),
    )
    return [
        {
            "quarter": str(row["quarter"]),
            "signal_position": int(row["signal_position"]),
            "entry_position": int(row["entry_position"]),
            "exit_position": int(row["exit_position"]),
            "signal_date": str(row["signal_date"]),
            "entry_date": str(row["entry_date"]),
            "exit_date": str(row["exit_date"]),
            "side": int(row["side"]),
            "branch": str(row["branch"]),
            "hold_bars": int(row["hold_bars"]),
        }
        for row in rows
    ]


def event_clock_hash(
    schedule: pd.DataFrame,
    *,
    selected_quantile: float,
) -> str:
    return canonical_hash(
        {
            "policy_id": POLICY_ID,
            "selected_quantile": float(selected_quantile),
            "events": event_records(schedule),
        }
    )


def frozen_artifacts() -> dict[str, Any]:
    return {
        "preregistration_source": str(PREREGISTRATION_SOURCE),
        "preregistration_source_sha256": sha256_file(PREREGISTRATION_SOURCE),
        "preregistration_document": str(PREREGISTRATION_DOCUMENT),
        "preregistration_document_sha256": sha256_file(
            PREREGISTRATION_DOCUMENT
        ),
        "source_decision_document": str(SOURCE_DECISION_DOCUMENT),
        "source_decision_document_sha256": SOURCE_DECISION_SHA256,
        "shared_causal_utility_source": str(SHARED_SOURCE),
        "shared_causal_utility_source_sha256": SHARED_SOURCE_SHA256,
        "source_panel_sha256": shared.SOURCE_PANEL_SHA256,
        "source_manifest_sha256": shared.SOURCE_MANIFEST_SHA256,
        "comparators": shared.COMPARATORS,
    }


def protocol(cfg: Config) -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "support_only": True,
        "outcomes_opened": False,
        "external_ohlc_funding_return_or_equity_loaded": False,
        "selection_end_exclusive": "2024-01-01 00:00:00",
        "feature": (
            "30m impulse of outer-minus-inner directional average-quote "
            "curvature, residualized against the 30m inner quote-center move by "
            "strictly-prior rolling OLS"
        ),
        "signal": (
            "residual/raw dominance >=25%, quote-center move no larger than its "
            "strictly-prior rolling median, and absolute residual crosses its "
            "strictly-prior rolling threshold from below"
        ),
        "side": "positive outer-ask curvature impulse long; negative short",
        "clock": (
            "completed 5m source bar; enter next 5m open; exit after "
            f"{cfg.hold_bars} completed 5m bars"
        ),
        "scheduler": (
            "one position; four quarter-contained schedules with non-overlap "
            "state reset at known UTC quarter boundaries"
        ),
        "source_gap_policy": (
            "signal requires source_complete for t-6..t; future source gaps never "
            "cancel an already selected event"
        ),
        "selection": (
            "strictest-first frozen quantile stopping on first incidence pass; "
            "no fallback after comparator novelty"
        ),
        "sealed_windows": ["test2024", "eval2025", "recent2026"],
    }


def _base_result(cfg: Config, synthetic: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": protocol(cfg),
        "config": asdict(cfg),
        "frozen_artifacts": frozen_artifacts(),
        "synthetic_control": synthetic,
    }


def run_support(cfg: Config) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _validate_config(cfg)
    synthetic = synthetic_control(cfg)
    base = _base_result(cfg, synthetic)
    if not synthetic["passes"]:
        return (
            {
                **base,
                "source_loaded": False,
                "threshold_trials": [],
                "selected_quantile": None,
                "all_support_gates_pass": False,
                "rejection_reason": "moving-band synthetic control failed",
            },
            None,
        )

    frame, source = shared.load_source()
    features = curvature_features(frame, cfg)
    public_trials: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for quantile in cfg.threshold_quantiles:
        signal = build_signal(frame, features, cfg, quantile=quantile)
        schedule = shared.quarterly_nonoverlap_schedule(signal, frame)
        support = support_summary(schedule, cfg)
        public_trials.append(
            {
                "quantile": quantile,
                "raw_event_count": int(signal["candidate"].sum()),
                "support": support,
            }
        )
        if support["passes_incidence"]:
            selected = {
                "quantile": quantile,
                "schedule": schedule,
                "support": support,
            }
            break
    if selected is None:
        return (
            {
                **base,
                "source_loaded": True,
                "source": source,
                "threshold_trials": public_trials,
                "selected_quantile": None,
                "all_support_gates_pass": False,
                "rejection_reason": "no frozen quantile passed incidence",
            },
            None,
        )

    schedule = cast(pd.DataFrame, selected["schedule"])
    entries = schedule["entry_position"].astype(int).tolist()
    overlaps: dict[str, Any] = {}
    novelty_passes = True
    for name in shared.COMPARATORS:
        overlap = shared.tolerant_event_jaccard(
            entries,
            shared.comparator_entries(name),
            tolerance_bars=cfg.overlap_tolerance_bars,
        )
        overlap["maximum_allowed_jaccard"] = cfg.maximum_comparator_jaccard
        overlap["passes"] = bool(
            overlap["jaccard"] <= cfg.maximum_comparator_jaccard
        )
        overlaps[name] = overlap
        novelty_passes &= overlap["passes"]
    quantile = float(selected["quantile"])
    clock_hash = event_clock_hash(schedule, selected_quantile=quantile)
    result = {
        **base,
        "source_loaded": True,
        "source": source,
        "threshold_trials": public_trials,
        "selected_quantile": quantile,
        "selected_support": selected["support"],
        "selected_event_clock_sha256": clock_hash,
        "comparator_overlap": overlaps,
        "all_support_gates_pass": bool(novelty_passes),
        "rejection_reason": None if novelty_passes else "comparator novelty failed",
    }
    if not novelty_passes:
        return result, None
    clock = {
        "protocol": "RQCI-24 canonical outcome-blind event-clock freeze",
        "outcomes_opened": False,
        "external_ohlc_funding_return_or_equity_loaded": False,
        "selection_end_exclusive": "2024-01-01 00:00:00",
        "selected_quantile": quantile,
        "canonical_fields": [
            "quarter",
            "signal_position",
            "entry_position",
            "exit_position",
            "signal_date",
            "entry_date",
            "exit_date",
            "side",
            "branch",
            "hold_bars",
        ],
        "event_count": int(len(schedule)),
        "quarter_counts": selected["support"]["by_quarter"],
        "side_counts": {
            "long": int(schedule["side"].gt(0).sum()),
            "short": int(schedule["side"].lt(0).sum()),
        },
        "event_clock_sha256": clock_hash,
        "events": event_records(schedule),
    }
    return result, clock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-output", default=Config.support_output)
    parser.add_argument("--event-clock-output", default=Config.event_clock_output)
    args = parser.parse_args()
    cfg = Config(
        support_output=args.support_output,
        event_clock_output=args.event_clock_output,
    )
    result, clock = run_support(cfg)
    support_path = Path(cfg.support_output)
    support_path.parent.mkdir(parents=True, exist_ok=True)
    support_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    clock_output: str | None = None
    if clock is not None:
        clock_path = Path(cfg.event_clock_output)
        clock_path.parent.mkdir(parents=True, exist_ok=True)
        clock_path.write_text(
            json.dumps(clock, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
        clock_output = str(clock_path)
    print(
        json.dumps(
            {
                "outcomes_opened": False,
                "all_support_gates_pass": result["all_support_gates_pass"],
                "selected_quantile": result.get("selected_quantile"),
                "support_output": str(support_path),
                "event_clock_output": clock_output,
                "rejection_reason": result.get("rejection_reason"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
