"""One-shot strict 2026H1 evaluator for frozen CRES-1.

The module refuses to load 2026 market/funding outcomes until its exact source
has been committed and frozen by
``freeze_causal_residual_expert_switcher_2026_evaluator``.  Within the walk,
each action is materialized before that event's counterfactual returns are
computed; those returns become eligible only for later signals.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import develop_causal_residual_expert_switcher_pre2026 as development
from training.build_causal_residual_expert_switcher_2026_support import (
    DEFAULT_CLOCK,
    DEFAULT_MANIFEST as SUPPORT_MANIFEST,
    END,
    EXPECTED_SOURCE_MANIFEST_HASH,
    SOURCE_MANIFEST,
    START,
    SYMBOLS,
    assert_clock_contract,
)
from training.export_leave_one_out_residual_exhaustion_sources import sha256_file
from training.preregister_causal_residual_expert_switcher_2026 import canonical_hash
from training.select_leave_one_out_residual_exhaustion_pre2025 import (
    MarketBundle,
    weekly_cluster_signflip,
)


CONFIRMATION_START = pd.Timestamp("2026-01-01 00:00:00")
CONFIRMATION_END = pd.Timestamp("2026-07-01 00:00:00")
Q2_START = pd.Timestamp("2026-04-01 00:00:00")

SOURCE_DIR = Path("data/binance_um_cres_2025_2026h1")
SEED_PATH = Path("data/cres_v1_training_seed_2023_2025.csv.gz")
EVALUATION_SOURCE = Path("training/evaluate_causal_residual_expert_switcher_2026.py")
TEST_PATH = Path("tests/test_evaluate_causal_residual_expert_switcher_2026.py")
EVALUATION_FREEZE = Path(
    "results/causal_residual_expert_switcher_2026_evaluator_freeze_2026-07-17.json"
)
DEFAULT_OUTPUT = Path("results/causal_residual_expert_switcher_2026_evaluation_2026-07-17.json")
DEFAULT_DOCS = Path("docs/causal-residual-expert-switcher-2026-evaluation-2026-07-17.md")

PREREGISTRATION_SHA256 = "6f3f66d3c58ce0a4f8dd7481e98bd74bf07f5115babbadedc01ab28094667456"
SOURCE_MANIFEST_SHA256 = "c3fc16e703bdcd9d9fb0095b4c1922b9acfd37343042a7f4601f76880b2ade3f"
SUPPORT_MANIFEST_SHA256 = "a00be68b193eadd484ce7a1fd7a06ee3d47c8c51b7a7164a71013f9e7b9959c5"
EXPECTED_SUPPORT_MANIFEST_HASH = "0f7e9000e2a578d46a01219695b52b51f67ef8fc99d3e68074b82905e950dc60"
CLOCK_SHA256 = "62b40c2474399595acd5c48f2fecb0b8f6b0f96cfb3fce1ec63da3a1c7522088"
SEED_SHA256 = "cdcd7719b0f3c1e40bcd4610c836fa7ca3f8dd83223e36c4b8a5840db202dec9"
DEVELOPMENT_SOURCE_SHA256 = "1d7387213fd497628cfd956af320d0e82372d965d874aee9019322b4f49463b1"
DEVELOPMENT_TEST_SHA256 = "d0624b93b01955dee1061087e370b645cf24c2f45b8ccb94d7882501d9a7f47c"

CURRENT_FEATURES = development.CURRENT_FEATURES
EDGE_FEATURES = development.EDGE_FEATURES
MODEL_FEATURES = development.MODEL_FEATURES
OUTCOME_LAG = development.OUTCOME_LAG


@dataclass(frozen=True)
class EvaluationConfig:
    minimum_history: int = 52
    maximum_history: int = 104
    ridge_alpha: float = 300.0
    confidence_quantile: float = 0.825
    risk_reference_events: int = 52
    minimum_gross_scale: float = 0.25
    base_cost_bp_per_side: float = 6.0
    stress_cost_bp_per_side: float = 10.0
    cluster_signflip_samples: int = 20_000
    cluster_signflip_seed: int = 20_260_717


CONFIG = EvaluationConfig()


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seal(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "manifest_hash": canonical_hash(core)}


def verify_support_and_clock() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    dependencies = (
        (Path("results/causal_residual_expert_switcher_2026_preregistration_2026-07-17.json"), PREREGISTRATION_SHA256),
        (Path(SOURCE_MANIFEST), SOURCE_MANIFEST_SHA256),
        (Path(SUPPORT_MANIFEST), SUPPORT_MANIFEST_SHA256),
        (Path(DEFAULT_CLOCK), CLOCK_SHA256),
        (SEED_PATH, SEED_SHA256),
        (Path(development.SCRIPT_PATH), DEVELOPMENT_SOURCE_SHA256),
        (Path(development.TEST_PATH), DEVELOPMENT_TEST_SHA256),
    )
    for path, expected in dependencies:
        if _sha256(path) != expected:
            raise ValueError(f"frozen CRES dependency changed: {path}")
    support = json.loads(Path(SUPPORT_MANIFEST).read_text())
    body = {key: value for key, value in support.items() if key not in {"manifest_hash", "created_at"}}
    if canonical_hash(body) != EXPECTED_SUPPORT_MANIFEST_HASH:
        raise ValueError("CRES support manifest body changed")
    if support.get("manifest_hash") != EXPECTED_SUPPORT_MANIFEST_HASH:
        raise ValueError("CRES support manifest identity changed")
    if support.get("source_manifest_hash") != EXPECTED_SOURCE_MANIFEST_HASH:
        raise ValueError("CRES source manifest binding changed")
    if support.get("post_entry_2026_strategy_returns_calculated") is not False:
        raise ValueError("CRES support artifact already opened outcomes")
    if support.get("support", {}).get("passes_support") is not True:
        raise ValueError("CRES support gate did not pass")
    clock = pd.read_csv(DEFAULT_CLOCK)
    assert_clock_contract(clock)
    for column in ("signal_time", "feature_available_time", "entry_time", "exit_time"):
        clock[column] = pd.to_datetime(clock[column], errors="raise")
    seed = pd.read_csv(SEED_PATH)
    for column in ("signal_time", "entry_time", "exit_time"):
        seed[column] = pd.to_datetime(seed[column], errors="raise")
    if seed.empty or not (seed["exit_time"] < CONFIRMATION_START).all():
        raise ValueError("CRES historical seed crossed the confirmation boundary")
    return support, clock, seed


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.exists():
        raise ValueError("CRES evaluator freeze is missing")
    payload = json.loads(EVALUATION_FREEZE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise ValueError("CRES evaluator freeze hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("CRES evaluator freeze already opened outcomes")
    if payload.get("evaluation_source") != str(EVALUATION_SOURCE):
        raise ValueError("CRES evaluator freeze source path changed")
    if payload.get("evaluation_source_sha256") != _sha256(EVALUATION_SOURCE):
        raise ValueError("CRES evaluator differs from its pre-outcome freeze")
    if payload.get("test_sha256") != _sha256(TEST_PATH):
        raise ValueError("CRES evaluator tests differ from their pre-outcome freeze")
    if payload.get("support_manifest_sha256") != SUPPORT_MANIFEST_SHA256:
        raise ValueError("CRES evaluator freeze support binding changed")
    if payload.get("clock_sha256") != CLOCK_SHA256 or payload.get("seed_sha256") != SEED_SHA256:
        raise ValueError("CRES evaluator freeze data binding changed")
    if payload.get("evaluation_config") != asdict(CONFIG):
        raise ValueError("CRES evaluator configuration changed")
    if payload.get("mutable_parameters") != []:
        raise ValueError("CRES evaluator freeze permits mutable parameters")
    for key in (
        "labels_constructed_during_freeze",
        "market_rows_parsed_during_freeze",
        "funding_rows_loaded_during_freeze",
        "execution_simulation_run_during_freeze",
    ):
        expected: Any = False if key in {"labels_constructed_during_freeze", "execution_simulation_run_during_freeze"} else 0
        if payload.get(key) != expected:
            raise ValueError(f"CRES evaluator freeze violated outcome boundary: {key}")
    return payload


def _load_source_manifest() -> dict[str, Any]:
    payload = json.loads(Path(SOURCE_MANIFEST).read_text())
    if payload.get("manifest_hash") != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("CRES source manifest hash changed")
    body = {key: value for key, value in payload.items() if key not in {"manifest_hash", "created_at"}}
    if canonical_hash(body) != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("CRES source manifest body changed")
    return payload


def load_bundle() -> MarketBundle:
    verify_evaluation_freeze()
    source = _load_source_manifest()
    records = {str(row["symbol"]): row for row in source["records"]}
    market: dict[str, dict[str, np.ndarray]] = {}
    funding: dict[str, pd.DataFrame] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    dates: pd.DatetimeIndex | None = None
    for symbol in sorted(SYMBOLS):
        market_path = SOURCE_DIR / f"{symbol}_5m_2025_2026h1.csv.gz"
        funding_path = SOURCE_DIR / f"{symbol}_funding_2025_2026h1.csv.gz"
        market_hash = sha256_file(market_path)
        funding_hash = sha256_file(funding_path)
        if market_hash != records[symbol]["output_market_sha256"]:
            raise RuntimeError(f"{symbol} CRES market source changed")
        if funding_hash != records[symbol]["output_funding_sha256"]:
            raise RuntimeError(f"{symbol} CRES funding source changed")
        frame = pd.read_csv(
            market_path,
            usecols=["date", "open", "high", "low", "close", "tic"],
            parse_dates=["date"],
        ).sort_values("date")
        if frame["date"].duplicated().any() or not frame["tic"].astype(str).eq(symbol).all():
            raise RuntimeError(f"{symbol} CRES market identity/grid failure")
        current_dates = pd.DatetimeIndex(frame["date"])
        if dates is None:
            dates = current_dates
        elif not dates.equals(current_dates):
            raise RuntimeError("CRES symbol market grids differ")
        arrays = {
            column: pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            for column in ("open", "high", "low", "close")
        }
        if not np.isfinite(np.column_stack(list(arrays.values()))).all():
            raise RuntimeError(f"{symbol} CRES non-finite market source")
        market[symbol] = arrays
        fund = pd.read_csv(funding_path)
        fund["event_time"] = pd.to_datetime(
            pd.to_numeric(fund["funding_time"], errors="raise"), unit="ms"
        )
        fund["funding_rate"] = pd.to_numeric(fund["funding_rate"], errors="raise")
        if fund["event_time"].duplicated().any() or not fund["event_time"].is_monotonic_increasing:
            raise RuntimeError(f"{symbol} CRES funding order failure")
        funding[symbol] = fund[["event_time", "funding_rate"]].copy()
        source_hashes[symbol] = {"market": market_hash, "funding": funding_hash}
    assert dates is not None
    expected = pd.date_range(START, END - pd.Timedelta(minutes=5), freq="5min")
    if not dates.equals(expected):
        raise RuntimeError("CRES execution grid is not the exact physical prefix")
    return MarketBundle(dates, market, funding, source_hashes)


def enrich_lag_features(events: pd.DataFrame) -> pd.DataFrame:
    enriched = events.sort_values("signal_time").reset_index(drop=True).copy()
    for column in EDGE_FEATURES:
        enriched[column] = float("nan")
    for index, row in enriched.iterrows():
        history = enriched.iloc[:index]
        history = history.loc[history["exit_time"] + OUTCOME_LAG <= row["signal_time"]]
        for window in (8, 16, 24):
            values = history["edge"].tail(window)
            enriched.loc[index, f"edge_mean_{window}"] = (
                float(values.mean()) if len(values) == window else float("nan")
            )
        values = history["edge"].tail(24)
        enriched.loc[index, "edge_std_24"] = (
            float(values.std(ddof=1)) if len(values) == 24 else float("nan")
        )
    return enriched


def _current_event(row: pd.Series, history: pd.DataFrame) -> dict[str, Any]:
    event: dict[str, Any] = {
        "signal_time": pd.Timestamp(row["signal_time"]),
        "entry_time": pd.Timestamp(row["entry_time"]),
        "exit_time": pd.Timestamp(row["exit_time"]),
        "loser_residual_z": float(row["loser_residual_z"]),
        "winner_residual_z": float(row["winner_residual_z"]),
        "loser_flow_z": float(row["loser_flow_z"]),
        "winner_flow_z": float(row["winner_flow_z"]),
        "setup_score": float(row["setup_score"]),
        "long_weight": float(row["continuation_long_weight_gross1"]),
        "short_weight_abs": float(row["continuation_short_weight_abs_gross1"]),
        "long_beta": float(row["continuation_long_beta"]),
        "short_beta": float(row["continuation_short_beta"]),
        "range_risk": float(row["range_risk"]),
    }
    eligible = history.loc[history["exit_time"] + OUTCOME_LAG <= event["signal_time"]]
    for window in (8, 16, 24):
        values = eligible["edge"].tail(window)
        event[f"edge_mean_{window}"] = (
            float(values.mean()) if len(values) == window else float("nan")
        )
    values = eligible["edge"].tail(24)
    event["edge_std_24"] = float(values.std(ddof=1)) if len(values) == 24 else float("nan")
    return event


def decide_event(history: pd.DataFrame, current: dict[str, Any]) -> dict[str, Any]:
    eligible = history.loc[
        history["exit_time"] + OUTCOME_LAG <= pd.Timestamp(current["signal_time"])
    ]
    training = eligible.loc[np.isfinite(eligible[list(MODEL_FEATURES)]).all(axis=1)].tail(
        CONFIG.maximum_history
    )
    risk_history = history.loc[
        history["signal_time"] < pd.Timestamp(current["signal_time"]), "range_risk"
    ].dropna().tail(CONFIG.risk_reference_events)
    if len(risk_history) == CONFIG.risk_reference_events and float(current["range_risk"]) > 0.0:
        gross_scale = float(
            np.clip(
                float(risk_history.median()) / float(current["range_risk"]),
                CONFIG.minimum_gross_scale,
                1.0,
            )
        )
    else:
        gross_scale = 0.0
    output = {
        "choice": "flat",
        "predicted_edge": float("nan"),
        "confidence_threshold": float("nan"),
        "training_rows": int(len(training)),
        "gross_scale": gross_scale,
    }
    vector = np.asarray([current[column] for column in MODEL_FEATURES], dtype=float)
    if len(training) < CONFIG.minimum_history or not np.isfinite(vector).all() or gross_scale <= 0.0:
        return output
    design = training[list(MODEL_FEATURES)].to_numpy(dtype=float)
    target = training["edge"].to_numpy(dtype=float)
    mean = design.mean(axis=0)
    std = design.std(axis=0, ddof=1)
    usable = std > 1e-12
    standardized = np.zeros_like(design)
    standardized_current = np.zeros_like(vector)
    standardized[:, usable] = (design[:, usable] - mean[usable]) / std[usable]
    standardized_current[usable] = (vector[usable] - mean[usable]) / std[usable]
    centered_target = target - target.mean()
    coefficients = np.linalg.solve(
        standardized.T @ standardized
        + CONFIG.ridge_alpha * np.eye(standardized.shape[1]),
        standardized.T @ centered_target,
    )
    fitted = standardized @ coefficients
    prediction = float(standardized_current @ coefficients)
    threshold = float(np.quantile(np.abs(fitted), CONFIG.confidence_quantile))
    choice = "flat"
    if abs(prediction) > max(threshold, 1e-12):
        choice = "continuation" if prediction > 0.0 else "reversion"
    output.update(
        {
            "choice": choice,
            "predicted_edge": prediction,
            "confidence_threshold": threshold,
        }
    )
    return output


def _canonical_event_clock(row: pd.Series, direction: str) -> pd.DataFrame:
    long_symbol = str(row["continuation_long_symbol"])
    short_symbol = str(row["continuation_short_symbol"])
    long_weight = float(row["continuation_long_weight_gross1"])
    short_weight = float(row["continuation_short_weight_abs_gross1"])
    long_beta = float(row["continuation_long_beta"])
    short_beta = float(row["continuation_short_beta"])
    if direction == "reversion":
        long_symbol, short_symbol = short_symbol, long_symbol
        long_weight, short_weight = short_weight, long_weight
        long_beta, short_beta = short_beta, long_beta
    elif direction != "continuation":
        raise ValueError(direction)
    return pd.DataFrame(
        [
            {
                "policy_id": "CRES01_COUNTERFACTUAL",
                "signal_time": pd.Timestamp(row["signal_time"]),
                "entry_time": pd.Timestamp(row["entry_time"]),
                "exit_time": pd.Timestamp(row["exit_time"]),
                "long_symbol": long_symbol,
                "short_symbol": short_symbol,
                "long_weight": long_weight,
                "short_weight_abs": short_weight,
                "long_beta": long_beta,
                "short_beta": short_beta,
                "choice": direction,
                "gross_scale": 1.0,
                "predicted_edge": 0.0,
                "confidence_threshold": 0.0,
            }
        ]
    )


def _counterfactual_outcomes(bundle: MarketBundle, row: pd.Series) -> dict[str, float]:
    outcomes: dict[str, float] = {}
    for direction in ("continuation", "reversion"):
        clock = _canonical_event_clock(row, direction)
        stats = development.simulate_segments(
            [
                development.Segment(
                    "counterfactual",
                    bundle,
                    clock,
                    CONFIRMATION_START,
                    CONFIRMATION_END,
                )
            ],
            calendar_start=str(CONFIRMATION_START.date()),
            calendar_end=str(CONFIRMATION_END.date()),
            cost_bp=CONFIG.base_cost_bp_per_side,
        )
        if stats["trades"] != 1:
            raise RuntimeError("CRES counterfactual event did not execute exactly once")
        outcomes[f"{direction}_net_log_return"] = float(stats["trade_rows"][0]["net_log_return"])
    outcomes["edge"] = outcomes["continuation_net_log_return"] - outcomes[
        "reversion_net_log_return"
    ]
    return outcomes


def walk_forward_decisions(
    bundle: MarketBundle,
    base_clock: pd.DataFrame,
    seed: pd.DataFrame,
    *,
    outcome_provider: Callable[[MarketBundle, pd.Series], dict[str, float]] = _counterfactual_outcomes,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    verify_evaluation_freeze()
    history = enrich_lag_features(seed)
    decisions: list[dict[str, Any]] = []
    for _, row in base_clock.sort_values("signal_time").iterrows():
        current = _current_event(row, history)
        decision = decide_event(history, current)
        # Materialize the immutable decision before the event path is opened.
        decision_row = {
            **{column: row[column] for column in row.index},
            **decision,
            "decision_materialized_before_outcome": True,
        }
        decisions.append(decision_row)
        outcomes = outcome_provider(bundle, row)
        history_row = {**current, **outcomes}
        history = pd.concat([history, pd.DataFrame([history_row])], ignore_index=True)
    decision_frame = pd.DataFrame(decisions)
    if len(decision_frame) != len(base_clock) or not decision_frame[
        "decision_materialized_before_outcome"
    ].all():
        raise RuntimeError("CRES walk-forward decision/outcome ordering failed")
    return decision_frame, history


def selected_clock(decisions: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "policy_id",
        "signal_time",
        "entry_time",
        "exit_time",
        "long_symbol",
        "short_symbol",
        "long_weight",
        "short_weight_abs",
        "long_beta",
        "short_beta",
        "choice",
        "gross_scale",
        "predicted_edge",
        "confidence_threshold",
    )
    rows: list[dict[str, Any]] = []
    for row in decisions.itertuples(index=False):
        choice = str(row.choice)
        if choice == "flat" or float(row.gross_scale) <= 0.0:
            continue
        long_symbol = str(row.continuation_long_symbol)
        short_symbol = str(row.continuation_short_symbol)
        long_weight = float(row.continuation_long_weight_gross1)
        short_weight = float(row.continuation_short_weight_abs_gross1)
        long_beta = float(row.continuation_long_beta)
        short_beta = float(row.continuation_short_beta)
        if choice == "reversion":
            long_symbol, short_symbol = short_symbol, long_symbol
            long_weight, short_weight = short_weight, long_weight
            long_beta, short_beta = short_beta, long_beta
        scale = float(row.gross_scale)
        rows.append(
            {
                "policy_id": "CRES01",
                "signal_time": pd.Timestamp(row.signal_time),
                "entry_time": pd.Timestamp(row.entry_time),
                "exit_time": pd.Timestamp(row.exit_time),
                "long_symbol": long_symbol,
                "short_symbol": short_symbol,
                "long_weight": long_weight * scale,
                "short_weight_abs": short_weight * scale,
                "long_beta": long_beta,
                "short_beta": short_beta,
                "choice": choice,
                "gross_scale": scale,
                "predicted_edge": float(row.predicted_edge),
                "confidence_threshold": float(row.confidence_threshold),
            }
        )
    selected = pd.DataFrame(rows, columns=columns)
    if selected.empty:
        return selected
    exposure = selected["long_weight"] * selected["long_beta"] - selected[
        "short_weight_abs"
    ] * selected["short_beta"]
    if not np.allclose(exposure, 0.0, atol=1e-12):
        raise RuntimeError("CRES selected clock lost factor-beta neutrality")
    if (selected["long_weight"] + selected["short_weight_abs"] > 1.0 + 1e-12).any():
        raise RuntimeError("CRES selected clock exceeded gross one")
    return selected.sort_values("entry_time").reset_index(drop=True)


def _simulate(
    bundle: MarketBundle,
    clock: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cost_bp: float,
) -> dict[str, Any]:
    return development.simulate_segments(
        [development.Segment("2026", bundle, clock, start, end)],
        calendar_start=str(start.date()),
        calendar_end=str(end.date()),
        cost_bp=cost_bp,
    )


def _slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key != "trade_rows"}


def _strategy_metrics(bundle: MarketBundle, clock: pd.DataFrame, cost_bp: float) -> dict[str, Any]:
    return {
        "h1": _simulate(bundle, clock, CONFIRMATION_START, CONFIRMATION_END, cost_bp=cost_bp),
        "q1": _simulate(bundle, clock, CONFIRMATION_START, Q2_START, cost_bp=cost_bp),
        "q2": _simulate(bundle, clock, Q2_START, CONFIRMATION_END, cost_bp=cost_bp),
    }


def _markdown(result: dict[str, Any]) -> str:
    primary = result["primary"]
    lines = [
        "# CRES-1 2026 one-shot evaluation",
        "",
        "## Decision",
        "",
        f"- Strategy gate: **{'PASS' if result['strategy_gate']['passes'] else 'FAIL'}**.",
        f"- Disposition: **{result['disposition']}**.",
        "- This file is the first and only CRES-1 opening of 2026 post-entry outcomes.",
        "- All decisions were materialized before each event's outcome was computed.",
        "",
        "## Primary metrics",
        "",
        "| Window | Absolute return | Full-calendar CAGR | Strict MDD | CAGR/MDD | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("h1", "2026H1"), ("q1", "2026 Q1"), ("q2", "2026 Q2")):
        stats = primary[key]
        lines.append(
            f"| {label} | {stats['absolute_return_pct']:.2f}% | {stats['cagr_pct']:.2f}% | "
            f"{stats['strict_mdd_pct']:.2f}% | {stats['cagr_to_strict_mdd']:.2f} | {stats['trades']} |"
        )
    lines.extend(
        [
            "",
            "## Controls (2026H1)",
            "",
            "| Control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("stress_10bp", "10 bp/side"),
        ("delay_five_minutes", "Entry/exit +5m"),
        ("direction_flip", "Direction flip"),
    ):
        stats = result[key]["h1"]
        lines.append(
            f"| {label} | {stats['absolute_return_pct']:.2f}% | {stats['cagr_pct']:.2f}% | "
            f"{stats['strict_mdd_pct']:.2f}% | {stats['cagr_to_strict_mdd']:.2f} | {stats['trades']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            f"- base events: {result['decision_counts']['base_events']}; executed: {result['decision_counts']['executed']}; continuation: {result['decision_counts']['continuation']}; reversion: {result['decision_counts']['reversion']}; flat: {result['decision_counts']['flat']};",
            f"- weekly-cluster sign-flip p-value: {result['weekly_cluster_signflip']['raw_p_value']:.5f};",
            "- 2023-2025 were development; no 2026 threshold/sign/model repair is permitted.",
            "",
            "Portfolio orthogonality is evaluated only if the strategy gate passes. Live use additionally requires an atomic two-leg alt executor and partial-fill neutralization.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    output: str | Path = DEFAULT_OUTPUT,
    docs_output: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    output_path = Path(output)
    if output_path.exists():
        raise RuntimeError("refusing to overwrite the one-shot CRES 2026 result")
    support, base_clock, seed = verify_support_and_clock()
    evaluator_freeze = verify_evaluation_freeze()
    bundle = load_bundle()
    decisions, history = walk_forward_decisions(bundle, base_clock, seed)
    execution_clock = selected_clock(decisions)
    primary_raw = _strategy_metrics(bundle, execution_clock, CONFIG.base_cost_bp_per_side)
    stress_raw = _strategy_metrics(bundle, execution_clock, CONFIG.stress_cost_bp_per_side)
    delayed_clock = development._delay_clock(execution_clock, 5)
    delayed_raw = _strategy_metrics(bundle, delayed_clock, CONFIG.base_cost_bp_per_side)
    opposite_clock = development._direction_flip(execution_clock)
    opposite_raw = _strategy_metrics(bundle, opposite_clock, CONFIG.base_cost_bp_per_side)
    primary = {key: _slim(value) for key, value in primary_raw.items()}
    stress = {key: _slim(value) for key, value in stress_raw.items()}
    delayed = {key: _slim(value) for key, value in delayed_raw.items()}
    opposite = {key: _slim(value) for key, value in opposite_raw.items()}
    signflip = weekly_cluster_signflip(
        primary_raw["h1"]["trade_rows"],
        seed=CONFIG.cluster_signflip_seed,
        samples=CONFIG.cluster_signflip_samples,
    )
    h1 = primary["h1"]
    checks = {
        "absolute_return_positive": h1["absolute_return_pct"] > 0.0,
        "annualized_cagr_to_strict_mdd_at_least_3": h1["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15": h1["strict_mdd_pct"] <= 15.0,
        "executed_trades_at_least_10": h1["trades"] >= 10,
        "q1_absolute_return_positive": primary["q1"]["absolute_return_pct"] > 0.0,
        "q2_absolute_return_positive": primary["q2"]["absolute_return_pct"] > 0.0,
        "ten_bp_cost_stress_positive": stress["h1"]["absolute_return_pct"] > 0.0,
        "entry_delay_plus_5m_positive": delayed["h1"]["absolute_return_pct"] > 0.0,
        "direction_flip_cagr_lower": opposite["h1"]["cagr_pct"] < h1["cagr_pct"],
    }
    strategy_pass = all(checks.values())
    trace_columns = [
        "signal_time",
        "entry_time",
        "exit_time",
        "choice",
        "gross_scale",
        "predicted_edge",
        "confidence_threshold",
        "training_rows",
        "decision_materialized_before_outcome",
    ]
    trace = decisions.loc[:, trace_columns].copy()
    for column in ("signal_time", "entry_time", "exit_time"):
        trace[column] = trace[column].astype(str)
    core: dict[str, Any] = {
        "protocol_version": "cres_v1_2026_one_shot_evaluation_2026-07-17",
        "outcomes_opened": True,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_source_commit": evaluator_freeze["evaluation_source_commit"],
        "evaluation_source_sha256": evaluator_freeze["evaluation_source_sha256"],
        "evaluator_freeze_sha256": _sha256(EVALUATION_FREEZE),
        "support_manifest_hash": support["manifest_hash"],
        "clock_sha256": CLOCK_SHA256,
        "seed_sha256": SEED_SHA256,
        "config": asdict(CONFIG),
        "decision_counts": {
            "base_events": int(len(decisions)),
            "executed": int(len(execution_clock)),
            "continuation": int(decisions["choice"].eq("continuation").sum()),
            "reversion": int(decisions["choice"].eq("reversion").sum()),
            "flat": int(decisions["choice"].eq("flat").sum()),
        },
        "primary": primary,
        "stress_10bp": stress,
        "delay_five_minutes": delayed,
        "direction_flip": opposite,
        "weekly_cluster_signflip": signflip,
        "strategy_gate": {"checks": checks, "passes": strategy_pass},
        "disposition": (
            "strategy_pass_pending_orthogonality_and_execution"
            if strategy_pass
            else "retire_cres1_no_2026_repair"
        ),
        "decision_trace": trace.to_dict(orient="records"),
        "executed_trade_rows": primary_raw["h1"]["trade_rows"],
        "history_rows_after_walk": int(len(history)),
    }
    result = _seal(core)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    docs_path = Path(docs_output)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(_markdown(result))
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
