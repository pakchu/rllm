"""Develop the pre-2026 causal residual expert switcher (CRES-1).

This module is deliberately a *development* evaluator.  The 2023-2025
outcomes were already visible to the research process, so they may select one
new policy but can never be relabelled out of sample.  The selected policy must
be frozen before any 2026 post-entry return is read.

CRES-1 trades a factor-beta-neutral pair among six liquid altcoin perpetuals.
At each already-frozen LORE/LORC event it chooses between the continuation and
mean-reversion directions with an online ridge model trained only on events
whose exits have been observable for at least five minutes.  A causal
range-risk scaler reduces gross exposure when either proposed leg has unusually
large recent intrabar ranges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import evaluate_leave_one_out_residual_continuation_2025 as lorc
from training import select_leave_one_out_residual_exhaustion_pre2025 as lore


POLICY_ID = "CRES01"
OUTPUT = "results/causal_residual_expert_switcher_development_2026-07-17.json"
DOCS_OUTPUT = "docs/causal-residual-expert-switcher-development-2026-07-17.md"
SCRIPT_PATH = "training/develop_causal_residual_expert_switcher_pre2026.py"
TEST_PATH = "tests/test_develop_causal_residual_expert_switcher_pre2026.py"

BASE_COST_BP = 6.0
STRESS_COST_BP = 10.0
OUTCOME_LAG = pd.Timedelta(minutes=5)
MIN_HISTORY = 52
MAX_HISTORY = 104
RIDGE_ALPHA = 300.0
CONFIDENCE_QUANTILE = 0.825
RISK_LOOKBACK_BARS = 3 * 24 * 12
RISK_HISTORY_EVENTS = 52
MIN_GROSS_SCALE = 0.25

CURRENT_FEATURES = (
    "loser_residual_z",
    "winner_residual_z",
    "loser_flow_z",
    "winner_flow_z",
    "setup_score",
    "long_weight",
    "short_weight_abs",
    "long_beta",
    "short_beta",
)
EDGE_FEATURES = ("edge_mean_8", "edge_mean_16", "edge_mean_24", "edge_std_24")
MODEL_FEATURES = CURRENT_FEATURES + EDGE_FEATURES

MULTIPLICITY_DISCLOSURE = {
    "status": "all 2023-2025 windows are development and globally research-seen",
    "families_screened_before_freeze": [
        "rolling best expert over 4/8/12/16/24/32/48 completed events",
        "rolling ridge windows 52/78/104/156/208 and ridge penalties 1/10/100",
        "confidence abstention quantiles from 0 through 0.9",
        "exponentially weighted ridge half-lives 13/26/52/104",
        "causal factor momentum and moving-average regime rules",
        "fixed next-open stop-loss diagnostics",
        "causal close-volatility/range-risk scaling diagnostics",
        "final narrow refinement: windows 78/104/130, penalties 30/100/300, confidence 0.55..0.85",
    ],
    "selection_consequence": "2026 is the first possible confirmatory window for this new family",
}


@dataclass(frozen=True)
class Segment:
    name: str
    bundle: Any
    clock: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class SimulationState:
    equity: float = 1.0
    peak: float = 1.0
    close_peak: float = 1.0
    strict_mdd: float = 0.0
    close_mdd: float = 0.0
    total_funding: float = 0.0
    total_cost: float = 0.0
    trade_rows: list[dict[str, Any]] = field(default_factory=list)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_attestation() -> dict[str, str]:
    status = subprocess.check_output(
        ["git", "status", "--short"], text=True, stderr=subprocess.STDOUT
    ).strip()
    if status:
        raise RuntimeError("CRES development evaluator must run from a clean repository")
    for path in (SCRIPT_PATH, TEST_PATH):
        subprocess.check_call(
            ["git", "ls-files", "--error-unmatch", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "test_sha256": sha256_file(TEST_PATH),
    }


def _expert_rows(
    continuation_stats: dict[str, Any],
    reversion_stats: dict[str, Any],
    clock: pd.DataFrame,
) -> pd.DataFrame:
    continuation = pd.DataFrame(continuation_stats["trade_rows"])[
        ["signal_time", "net_log_return"]
    ].rename(columns={"net_log_return": "continuation_net_log_return"})
    reversion = pd.DataFrame(reversion_stats["trade_rows"])[
        ["signal_time", "net_log_return"]
    ].rename(columns={"net_log_return": "reversion_net_log_return"})
    continuation["signal_time"] = pd.to_datetime(continuation["signal_time"])
    reversion["signal_time"] = pd.to_datetime(reversion["signal_time"])
    timing = clock[["signal_time", "entry_time", "exit_time"]].copy()
    for column in timing.columns:
        timing[column] = pd.to_datetime(timing[column])
    merged = timing.merge(continuation, on="signal_time", validate="one_to_one")
    merged = merged.merge(reversion, on="signal_time", validate="one_to_one")
    if len(merged) != len(clock):
        raise RuntimeError("expert simulations did not cover the frozen event clock")
    merged["edge"] = (
        merged["continuation_net_log_return"] - merged["reversion_net_log_return"]
    )
    return merged


def _completed_range_risk(bundle: Any, clock: pd.DataFrame) -> pd.Series:
    range_logs = {
        symbol: np.log(bundle.market[symbol]["high"] / bundle.market[symbol]["low"])
        for symbol in bundle.market
    }
    values: list[float] = []
    for row in clock.itertuples(index=False):
        signal = pd.Timestamp(row.signal_time)
        end_index = int(bundle.dates.searchsorted(signal, side="left"))
        start_index = end_index - RISK_LOOKBACK_BARS
        if start_index < 0:
            values.append(float("nan"))
            continue
        long_rms = float(
            np.sqrt(np.mean(range_logs[str(row.long_symbol)][start_index:end_index] ** 2))
        )
        short_rms = float(
            np.sqrt(np.mean(range_logs[str(row.short_symbol)][start_index:end_index] ** 2))
        )
        values.append(max(long_rms, short_rms))
    return pd.Series(values, index=pd.to_datetime(clock["signal_time"]), name="range_risk")


def _with_setup_score(clock: pd.DataFrame, source_column: str) -> pd.DataFrame:
    columns = list(CURRENT_FEATURES)
    columns[columns.index("setup_score")] = source_column
    features = clock[["signal_time", *columns]].copy()
    features = features.rename(columns={source_column: "setup_score"})
    features["signal_time"] = pd.to_datetime(features["signal_time"])
    return features


def build_development_events() -> tuple[pd.DataFrame, list[Segment], dict[str, str]]:
    bundle_2023_2024 = lore.load_bundle()
    lore_clock = lore.load_clock()
    lore_clock = lore_clock.loc[lore_clock["policy_id"].eq("L03")].copy().reset_index(drop=True)
    if lore_clock.empty or not lore_clock["residual_horizon_hours"].eq(12).all():
        raise RuntimeError("CRES expected the frozen 12-hour L03 clock")
    continuation_2023_2024 = lore._transform_same_clock(lore_clock, "direction_flip")

    bundle_2025 = lorc.load_bundle()
    continuation_2025 = lorc.load_clock().copy().reset_index(drop=True)
    reversion_2025 = lorc._transform_clock(continuation_2025, "direction_flip")

    expert_2023_2024 = _expert_rows(
        lore.simulate(
            bundle_2023_2024,
            continuation_2023_2024,
            start="2023-01-01",
            end="2025-01-01",
            cost_bp=BASE_COST_BP,
        ),
        lore.simulate(
            bundle_2023_2024,
            lore_clock,
            start="2023-01-01",
            end="2025-01-01",
            cost_bp=BASE_COST_BP,
        ),
        continuation_2023_2024,
    )
    expert_2025 = _expert_rows(
        lorc.simulate(
            bundle_2025,
            continuation_2025,
            start="2025-01-01",
            end="2026-01-01",
            cost_bp=BASE_COST_BP,
        ),
        lorc.simulate(
            bundle_2025,
            reversion_2025,
            start="2025-01-01",
            end="2026-01-01",
            cost_bp=BASE_COST_BP,
        ),
        continuation_2025,
    )

    features_2023_2024 = _with_setup_score(continuation_2023_2024, "exhaustion_score")
    features_2025 = _with_setup_score(continuation_2025, "continuation_score")
    risks = pd.concat(
        [
            _completed_range_risk(bundle_2023_2024, continuation_2023_2024),
            _completed_range_risk(bundle_2025, continuation_2025),
        ]
    ).sort_index()
    features = pd.concat([features_2023_2024, features_2025], ignore_index=True)
    expert = pd.concat([expert_2023_2024, expert_2025], ignore_index=True)
    events = expert.merge(features, on="signal_time", validate="one_to_one")
    events["range_risk"] = events["signal_time"].map(risks)
    events = events.sort_values("signal_time").reset_index(drop=True)
    if events["signal_time"].duplicated().any():
        raise RuntimeError("CRES event times are not unique")
    if not (
        events["exit_time"].shift(1).dropna() + OUTCOME_LAG
        <= events["signal_time"].iloc[1:].to_numpy()
    ).all():
        raise RuntimeError("CRES frozen event clocks do not leave the outcome publication lag")

    for index, row in events.iterrows():
        eligible = events.iloc[:index]
        eligible = eligible.loc[eligible["exit_time"] + OUTCOME_LAG <= row["signal_time"]]
        for window in (8, 16, 24):
            values = eligible["edge"].tail(window)
            events.loc[index, f"edge_mean_{window}"] = (
                float(values.mean()) if len(values) == window else float("nan")
            )
        values_24 = eligible["edge"].tail(24)
        events.loc[index, "edge_std_24"] = (
            float(values_24.std(ddof=1)) if len(values_24) == 24 else float("nan")
        )
        risk_history = events.iloc[:index]["range_risk"].dropna().tail(RISK_HISTORY_EVENTS)
        if len(risk_history) == RISK_HISTORY_EVENTS and float(row["range_risk"]) > 0.0:
            raw_scale = float(risk_history.median()) / float(row["range_risk"])
            events.loc[index, "gross_scale"] = float(np.clip(raw_scale, MIN_GROSS_SCALE, 1.0))
        else:
            events.loc[index, "gross_scale"] = 0.0

    segments = [
        Segment(
            "2023_2024",
            bundle_2023_2024,
            continuation_2023_2024,
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2025-01-01"),
        ),
        Segment(
            "2025",
            bundle_2025,
            continuation_2025,
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2026-01-01"),
        ),
    ]
    hashes = {
        "lore_clock_sha256": lore.EXPECTED_CLOCK_HASH,
        "lore_support_manifest_sha256": lore.EXPECTED_SUPPORT_MANIFEST_HASH,
        "lorc_clock_sha256": lorc.EXPECTED_CLOCK_HASH,
        "lorc_support_manifest_sha256": lorc.EXPECTED_SUPPORT_MANIFEST_HASH,
    }
    return events, segments, hashes


def online_ridge_choices(events: pd.DataFrame) -> pd.DataFrame:
    decisions: list[dict[str, Any]] = []
    for index, row in events.iterrows():
        history = events.iloc[:index]
        history = history.loc[history["exit_time"] + OUTCOME_LAG <= row["signal_time"]]
        history = history.loc[np.isfinite(history[list(MODEL_FEATURES)]).all(axis=1)]
        history = history.tail(MAX_HISTORY)
        decision: dict[str, Any] = {
            "choice": "flat",
            "predicted_edge": float("nan"),
            "confidence_threshold": float("nan"),
            "training_rows": int(len(history)),
        }
        current = row[list(MODEL_FEATURES)].to_numpy(dtype=float)
        if len(history) < MIN_HISTORY or not np.isfinite(current).all():
            decisions.append(decision)
            continue
        design = history[list(MODEL_FEATURES)].to_numpy(dtype=float)
        target = history["edge"].to_numpy(dtype=float)
        mean = design.mean(axis=0)
        std = design.std(axis=0, ddof=1)
        usable = std > 1e-12
        standardized = np.zeros_like(design)
        standardized_current = np.zeros_like(current)
        standardized[:, usable] = (design[:, usable] - mean[usable]) / std[usable]
        standardized_current[usable] = (current[usable] - mean[usable]) / std[usable]
        centered_target = target - target.mean()
        coefficients = np.linalg.solve(
            standardized.T @ standardized + RIDGE_ALPHA * np.eye(standardized.shape[1]),
            standardized.T @ centered_target,
        )
        fitted = standardized @ coefficients
        prediction = float(standardized_current @ coefficients)
        threshold = float(np.quantile(np.abs(fitted), CONFIDENCE_QUANTILE))
        decision.update(
            {
                "predicted_edge": prediction,
                "confidence_threshold": threshold,
                "choice": (
                    "continuation" if prediction > 0.0 else "reversion"
                )
                if abs(prediction) > max(threshold, 1e-12)
                else "flat",
            }
        )
        decisions.append(decision)
    return pd.concat([events.reset_index(drop=True), pd.DataFrame(decisions)], axis=1)


def selected_clock(base_clock: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    lookup = decisions.set_index("signal_time")
    selected = base_clock.copy()
    selected["signal_time"] = pd.to_datetime(selected["signal_time"])
    selected["choice"] = selected["signal_time"].map(lookup["choice"])
    selected["gross_scale"] = selected["signal_time"].map(lookup["gross_scale"])
    selected["predicted_edge"] = selected["signal_time"].map(lookup["predicted_edge"])
    selected["confidence_threshold"] = selected["signal_time"].map(
        lookup["confidence_threshold"]
    )
    selected["training_rows"] = selected["signal_time"].map(lookup["training_rows"])
    selected = selected.loc[selected["choice"].ne("flat") & selected["gross_scale"].gt(0.0)].copy()
    reverse = selected["choice"].eq("reversion")
    selected.loc[reverse, ["long_symbol", "short_symbol"]] = selected.loc[
        reverse, ["short_symbol", "long_symbol"]
    ].to_numpy()
    selected.loc[reverse, ["long_weight", "short_weight_abs"]] = selected.loc[
        reverse, ["short_weight_abs", "long_weight"]
    ].to_numpy()
    selected.loc[reverse, ["long_beta", "short_beta"]] = selected.loc[
        reverse, ["short_beta", "long_beta"]
    ].to_numpy()
    selected["long_weight"] *= selected["gross_scale"]
    selected["short_weight_abs"] *= selected["gross_scale"]
    selected["policy_id"] = POLICY_ID
    exposure = (
        selected["long_weight"] * selected["long_beta"]
        - selected["short_weight_abs"] * selected["short_beta"]
    )
    if not np.allclose(exposure, 0.0, atol=1e-12):
        raise RuntimeError("CRES selected clock lost factor-beta neutrality")
    return selected.sort_values("entry_time").reset_index(drop=True)


def _funding_events(bundle: Any) -> dict[str, dict[int, list[tuple[pd.Timestamp, float, float]]]]:
    dates_ns = bundle.dates.to_numpy(dtype="datetime64[ns]")
    completed_ns = (bundle.dates + pd.Timedelta(minutes=5)).to_numpy(dtype="datetime64[ns]")
    output: dict[str, dict[int, list[tuple[pd.Timestamp, float, float]]]] = {}
    for symbol, frame in bundle.funding.items():
        mapped: dict[int, list[tuple[pd.Timestamp, float, float]]] = {}
        for row in frame.itertuples(index=False):
            event = pd.Timestamp(row.event_time)
            event64 = event.to_datetime64()
            bar_index = int(np.searchsorted(dates_ns, event64, side="right") - 1)
            mark_index = int(np.searchsorted(completed_ns, event64, side="right") - 1)
            if bar_index < 0 or mark_index < 0 or bar_index >= len(bundle.dates):
                continue
            mark = float(bundle.market[symbol]["close"][mark_index])
            mapped.setdefault(bar_index, []).append((event, float(row.funding_rate), mark))
        output[symbol] = mapped
    return output


def _process_segment(
    state: SimulationState,
    segment: Segment,
    *,
    cost_bp: float,
) -> None:
    if cost_bp < 0.0:
        raise ValueError("negative execution cost")
    rate = cost_bp / 10_000.0
    selected = segment.clock.loc[
        (segment.clock["entry_time"] >= segment.start)
        & (segment.clock["exit_time"] < segment.end)
    ].sort_values("entry_time")
    funding_by_bar = _funding_events(segment.bundle)
    previous_exit: pd.Timestamp | None = None
    for row in selected.itertuples(index=False):
        entry_time = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if previous_exit is not None and entry_time < previous_exit:
            raise RuntimeError("CRES execution clock overlaps")
        previous_exit = exit_time
        entry_index = int(segment.bundle.dates.searchsorted(entry_time))
        exit_index = int(segment.bundle.dates.searchsorted(exit_time))
        if (
            entry_index >= len(segment.bundle.dates)
            or segment.bundle.dates[entry_index] != entry_time
            or exit_index >= len(segment.bundle.dates)
            or segment.bundle.dates[exit_index] != exit_time
        ):
            raise RuntimeError("CRES exact entry/exit open is missing")
        if exit_index <= entry_index:
            raise RuntimeError("CRES non-positive hold")

        long_symbol = str(row.long_symbol)
        short_symbol = str(row.short_symbol)
        long_weight = float(row.long_weight)
        short_weight = float(row.short_weight_abs)
        gross = long_weight + short_weight
        if not (0.0 < gross <= 1.0 + 1e-12):
            raise RuntimeError("CRES gross exposure escaped (0, 1]")
        start_equity = state.equity
        long_entry = float(segment.bundle.market[long_symbol]["open"][entry_index])
        short_entry = float(segment.bundle.market[short_symbol]["open"][entry_index])
        long_qty = long_weight * start_equity / long_entry
        short_qty = short_weight * start_equity / short_entry
        entry_cost = rate * (long_qty * long_entry + short_qty * short_entry)
        state.total_cost += entry_cost
        cumulative_funding = 0.0
        equity_after_entry = start_equity - entry_cost
        state.strict_mdd = max(state.strict_mdd, 1.0 - equity_after_entry / state.peak)

        def settle_funding(bar_index: int) -> None:
            nonlocal cumulative_funding
            for symbol, signed_qty in (
                (long_symbol, long_qty),
                (short_symbol, -short_qty),
            ):
                for event_time, funding_rate, mark in funding_by_bar[symbol].get(bar_index, []):
                    if entry_time < event_time <= exit_time:
                        cash = -signed_qty * mark * funding_rate
                        cumulative_funding += cash
                        state.total_funding += cash

        for bar_index in range(entry_index, exit_index):
            settle_funding(bar_index)
            long_high = float(segment.bundle.market[long_symbol]["high"][bar_index])
            long_low = float(segment.bundle.market[long_symbol]["low"][bar_index])
            short_high = float(segment.bundle.market[short_symbol]["high"][bar_index])
            short_low = float(segment.bundle.market[short_symbol]["low"][bar_index])
            favorable_pnl = long_qty * (long_high - long_entry) + short_qty * (
                short_entry - short_low
            )
            adverse_pnl = long_qty * (long_low - long_entry) + short_qty * (
                short_entry - short_high
            )
            favorable_liquidation = rate * (
                long_qty * long_high + short_qty * short_low
            )
            adverse_liquidation = rate * (long_qty * long_low + short_qty * short_high)
            favorable_equity = (
                start_equity
                - entry_cost
                + cumulative_funding
                + favorable_pnl
                - favorable_liquidation
            )
            state.peak = max(state.peak, favorable_equity)
            adverse_equity = (
                start_equity
                - entry_cost
                + cumulative_funding
                + adverse_pnl
                - adverse_liquidation
            )
            state.strict_mdd = max(state.strict_mdd, 1.0 - adverse_equity / state.peak)

            long_close = float(segment.bundle.market[long_symbol]["close"][bar_index])
            short_close = float(segment.bundle.market[short_symbol]["close"][bar_index])
            close_pnl = long_qty * (long_close - long_entry) + short_qty * (
                short_entry - short_close
            )
            close_liquidation = rate * (
                long_qty * long_close + short_qty * short_close
            )
            close_equity = (
                start_equity
                - entry_cost
                + cumulative_funding
                + close_pnl
                - close_liquidation
            )
            state.close_peak = max(state.close_peak, close_equity)
            state.close_mdd = max(state.close_mdd, 1.0 - close_equity / state.close_peak)

        settle_funding(exit_index)
        long_exit = float(segment.bundle.market[long_symbol]["open"][exit_index])
        short_exit = float(segment.bundle.market[short_symbol]["open"][exit_index])
        exit_cost = rate * (long_qty * long_exit + short_qty * short_exit)
        state.total_cost += exit_cost
        pnl = long_qty * (long_exit - long_entry) + short_qty * (short_entry - short_exit)
        state.equity = start_equity - entry_cost + cumulative_funding + pnl - exit_cost
        state.strict_mdd = max(state.strict_mdd, 1.0 - state.equity / state.peak)
        state.close_peak = max(state.close_peak, state.equity)
        state.close_mdd = max(state.close_mdd, 1.0 - state.equity / state.close_peak)
        state.peak = max(state.peak, state.equity)
        state.trade_rows.append(
            {
                "signal_time": str(pd.Timestamp(row.signal_time)),
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "long_symbol": long_symbol,
                "short_symbol": short_symbol,
                "choice": str(row.choice),
                "gross_scale": float(row.gross_scale),
                "predicted_edge": float(row.predicted_edge),
                "confidence_threshold": float(row.confidence_threshold),
                "net_return": float(state.equity / start_equity - 1.0),
                "net_log_return": float(
                    math.log(max(state.equity, 1e-15) / max(start_equity, 1e-15))
                ),
                "funding_cash": float(cumulative_funding),
            }
        )
        if state.equity <= 0.0:
            state.strict_mdd = 1.0
            break


def simulate_segments(
    segments: Iterable[Segment],
    *,
    calendar_start: str,
    calendar_end: str,
    cost_bp: float = BASE_COST_BP,
) -> dict[str, Any]:
    state = SimulationState()
    used = list(segments)
    for segment in used:
        _process_segment(state, segment, cost_bp=cost_bp)
        if state.equity <= 0.0:
            break
    start = pd.Timestamp(calendar_start)
    end = pd.Timestamp(calendar_end)
    years = (end - start).total_seconds() / (365.25 * 86_400.0)
    if years <= 0.0:
        raise ValueError("non-positive CRES reporting calendar")
    absolute = (state.equity - 1.0) * 100.0
    cagr = (state.equity ** (1.0 / years) - 1.0) * 100.0 if state.equity > 0 else -100.0
    strict_pct = min(max(state.strict_mdd * 100.0, 0.0), 100.0)
    close_pct = min(max(state.close_mdd * 100.0, 0.0), 100.0)
    returns = np.asarray([row["net_return"] for row in state.trade_rows], dtype=float)
    ratio = (
        cagr / strict_pct
        if strict_pct > 1e-12
        else (100.0 if cagr > 0.0 else 0.0)
    )
    return {
        "absolute_return_pct": float(absolute),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_pct),
        "close_mdd_pct": float(close_pct),
        "cagr_to_strict_mdd": float(ratio),
        "trades": len(state.trade_rows),
        "mean_net_bps": float(returns.mean() * 10_000.0) if len(returns) else 0.0,
        "win_rate": float(np.mean(returns > 0.0)) if len(returns) else 0.0,
        "funding_cash_pct_initial": float(state.total_funding * 100.0),
        "transaction_cost_pct_initial": float(state.total_cost * 100.0),
        "calendar_start": str(start.date()),
        "calendar_end_exclusive": str(end.date()),
        "trade_rows": state.trade_rows,
    }


def _slice_segment(segment: Segment, start: str, end: str) -> Segment:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    return Segment(segment.name, segment.bundle, segment.clock, start_ts, end_ts)


def _slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key != "trade_rows"}


def _direction_flip(clock: pd.DataFrame) -> pd.DataFrame:
    out = clock.copy()
    out[["long_symbol", "short_symbol"]] = out[["short_symbol", "long_symbol"]].to_numpy()
    out[["long_weight", "short_weight_abs"]] = out[
        ["short_weight_abs", "long_weight"]
    ].to_numpy()
    out[["long_beta", "short_beta"]] = out[["short_beta", "long_beta"]].to_numpy()
    out["choice"] = out["choice"].map(
        {"continuation": "reversion", "reversion": "continuation"}
    )
    return out


def _delay_clock(clock: pd.DataFrame, minutes: int) -> pd.DataFrame:
    out = clock.copy()
    delta = pd.Timedelta(minutes=minutes)
    out["entry_time"] = pd.to_datetime(out["entry_time"]) + delta
    out["exit_time"] = pd.to_datetime(out["exit_time"]) + delta
    return out


def _evaluate_clock(segments: list[Segment], cost_bp: float) -> dict[str, Any]:
    by_name = {segment.name: segment for segment in segments}
    annual = {
        "2023": simulate_segments(
            [_slice_segment(by_name["2023_2024"], "2023-01-01", "2024-01-01")],
            calendar_start="2023-01-01",
            calendar_end="2024-01-01",
            cost_bp=cost_bp,
        ),
        "2024": simulate_segments(
            [_slice_segment(by_name["2023_2024"], "2024-01-01", "2025-01-01")],
            calendar_start="2024-01-01",
            calendar_end="2025-01-01",
            cost_bp=cost_bp,
        ),
        "2025": simulate_segments(
            [_slice_segment(by_name["2025"], "2025-01-01", "2026-01-01")],
            calendar_start="2025-01-01",
            calendar_end="2026-01-01",
            cost_bp=cost_bp,
        ),
    }
    combined = simulate_segments(
        [
            _slice_segment(by_name["2023_2024"], "2024-01-01", "2025-01-01"),
            _slice_segment(by_name["2025"], "2025-01-01", "2026-01-01"),
        ],
        calendar_start="2024-01-01",
        calendar_end="2026-01-01",
        cost_bp=cost_bp,
    )
    return {"annual": annual, "combined_2024_2025": combined}


def _markdown(result: dict[str, Any]) -> str:
    primary = result["primary"]
    lines = [
        "# Causal residual expert switcher development (CRES-1)",
        "",
        "## Decision",
        "",
        f"- Development gate: **{'PASS' if result['development_gate']['passes'] else 'FAIL'}**.",
        "- This is not OOS evidence and not deployable. Every 2023-2025 outcome was already research-seen.",
        "- 2026 post-entry outcomes remained unopened by this evaluator; one exact policy must be frozen first.",
        "- The strategy is market-neutral across six alt perpetuals and holds no BTC leg.",
        "",
        "## Frozen selected development policy",
        "",
        f"- Online ridge: min history {MIN_HISTORY}, last {MAX_HISTORY}, alpha {RIDGE_ALPHA:g}, no target-mean/intercept drift.",
        f"- Confidence: trade only above the {CONFIDENCE_QUANTILE:.3f} in-sample absolute fitted-edge quantile.",
        f"- Outcome publication lag: prior exit + {int(OUTCOME_LAG.total_seconds() // 60)} minutes <= signal.",
        f"- Risk scale: 3-day completed 5m max-leg log-range RMS versus prior {RISK_HISTORY_EVENTS}-event median, clipped [{MIN_GROSS_SCALE:.2f}, 1.00].",
        f"- Cost: {BASE_COST_BP:g} bp/side base, {STRESS_COST_BP:g} bp/side stress; funding included.",
        "- Strict MDD: global pre-entry HWM, funding cash, favorable-before-adverse held OHLC, hypothetical liquidation cost.",
        "",
        "## Development metrics",
        "",
        "| Window | Absolute return | Full-calendar CAGR | Strict MDD | CAGR/MDD | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for window in ("2023", "2024", "2025"):
        stats = primary["annual"][window]
        lines.append(
            f"| {window}{' warm-up' if window == '2023' else ''} | {stats['absolute_return_pct']:.2f}% | "
            f"{stats['cagr_pct']:.2f}% | {stats['strict_mdd_pct']:.2f}% | "
            f"{stats['cagr_to_strict_mdd']:.2f} | {stats['trades']} |"
        )
    stats = primary["combined_2024_2025"]
    lines.append(
        f"| 2024-2025 combined | {stats['absolute_return_pct']:.2f}% | {stats['cagr_pct']:.2f}% | "
        f"{stats['strict_mdd_pct']:.2f}% | {stats['cagr_to_strict_mdd']:.2f} | {stats['trades']} |"
    )
    stress = result["stress_10bp"]["combined_2024_2025"]
    opposite = result["direction_flip_control"]["combined_2024_2025"]
    delayed = result["delay_five_minutes_control"]["combined_2024_2025"]
    lines.extend(
        [
            "",
            "## Controls (2024-2025 combined)",
            "",
            "| Control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |",
            "|---|---:|---:|---:|---:|---:|",
            f"| 10 bp/side | {stress['absolute_return_pct']:.2f}% | {stress['cagr_pct']:.2f}% | {stress['strict_mdd_pct']:.2f}% | {stress['cagr_to_strict_mdd']:.2f} | {stress['trades']} |",
            f"| Direction flip | {opposite['absolute_return_pct']:.2f}% | {opposite['cagr_pct']:.2f}% | {opposite['strict_mdd_pct']:.2f}% | {opposite['cagr_to_strict_mdd']:.2f} | {opposite['trades']} |",
            f"| Entry/exit +5m | {delayed['absolute_return_pct']:.2f}% | {delayed['cagr_pct']:.2f}% | {delayed['strict_mdd_pct']:.2f}% | {delayed['cagr_to_strict_mdd']:.2f} | {delayed['trades']} |",
            "",
            "## Multiple-testing and execution warning",
            "",
            "The policy is a post-hoc successor to failed LORE/LORC studies. The disclosed development search included rolling/weighted experts, ridge windows and penalties, confidence levels, regime rules, stop diagnostics, and causal risk scalers. Therefore only a preregistered one-shot 2026 replay can confirm it.",
            "",
            "The current live executor is BTC single-symbol oriented. CRES-1 remains research/shadow-only until atomic two-leg alt execution, partial-fill neutralization, per-leg min-notional/slippage, and pair-level reservation are implemented and parity-tested.",
            "",
            "## Artifacts",
            "",
            f"- `{OUTPUT}`",
            f"- `{SCRIPT_PATH}`",
            f"- `{TEST_PATH}`",
            "",
        ]
    )
    return "\n".join(lines)


def run(output: str = OUTPUT, docs_output: str = DOCS_OUTPUT) -> dict[str, Any]:
    attestation = _git_attestation()
    events, base_segments, source_hashes = build_development_events()
    decisions = online_ridge_choices(events)
    selected_segments = [
        Segment(
            segment.name,
            segment.bundle,
            selected_clock(segment.clock, decisions),
            segment.start,
            segment.end,
        )
        for segment in base_segments
    ]
    primary_raw = _evaluate_clock(selected_segments, BASE_COST_BP)
    stress_raw = _evaluate_clock(selected_segments, STRESS_COST_BP)
    opposite_segments = [
        Segment(s.name, s.bundle, _direction_flip(s.clock), s.start, s.end)
        for s in selected_segments
    ]
    delayed_segments = [
        Segment(s.name, s.bundle, _delay_clock(s.clock, 5), s.start, s.end)
        for s in selected_segments
    ]
    opposite_raw = _evaluate_clock(opposite_segments, BASE_COST_BP)
    delayed_raw = _evaluate_clock(delayed_segments, BASE_COST_BP)

    primary = {
        "annual": {key: _slim(value) for key, value in primary_raw["annual"].items()},
        "combined_2024_2025": _slim(primary_raw["combined_2024_2025"]),
    }
    stress = {
        "annual": {key: _slim(value) for key, value in stress_raw["annual"].items()},
        "combined_2024_2025": _slim(stress_raw["combined_2024_2025"]),
    }
    opposite = {
        "annual": {key: _slim(value) for key, value in opposite_raw["annual"].items()},
        "combined_2024_2025": _slim(opposite_raw["combined_2024_2025"]),
    }
    delayed = {
        "annual": {key: _slim(value) for key, value in delayed_raw["annual"].items()},
        "combined_2024_2025": _slim(delayed_raw["combined_2024_2025"]),
    }
    combined = primary["combined_2024_2025"]
    annual_2024 = primary["annual"]["2024"]
    annual_2025 = primary["annual"]["2025"]
    checks = {
        "combined_positive_absolute_return": combined["absolute_return_pct"] > 0.0,
        "combined_cagr_to_strict_mdd_at_least_3": combined["cagr_to_strict_mdd"] >= 3.0,
        "combined_strict_mdd_at_most_15": combined["strict_mdd_pct"] <= 15.0,
        "combined_at_least_40_trades": combined["trades"] >= 40,
        "both_report_years_positive": all(
            stats["absolute_return_pct"] > 0.0 for stats in (annual_2024, annual_2025)
        ),
        "both_report_years_strict_mdd_at_most_15": all(
            stats["strict_mdd_pct"] <= 15.0 for stats in (annual_2024, annual_2025)
        ),
        "both_report_years_at_least_20_trades": all(
            stats["trades"] >= 20 for stats in (annual_2024, annual_2025)
        ),
        "stress_10bp_positive": stress["combined_2024_2025"]["absolute_return_pct"] > 0.0,
        "direction_flip_worse": opposite["combined_2024_2025"]["cagr_pct"] < combined["cagr_pct"],
    }
    result: dict[str, Any] = {
        "protocol_version": "cres_v1_pre2026_development_2026-07-17",
        "policy_id": POLICY_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_status": "development_only_2023_2025_seen_2026_outcomes_unopened",
        "attestation": attestation,
        "source_hashes": source_hashes,
        "policy": {
            "symbols": list(lore.SYMBOLS),
            "base_event": "frozen 12h LORE/LORC factor-beta-neutral winner/loser pair",
            "model_features": list(MODEL_FEATURES),
            "target": "continuation net log return minus reversion net log return after costs/funding",
            "outcome_lag_minutes": 5,
            "min_history": MIN_HISTORY,
            "max_history": MAX_HISTORY,
            "ridge_alpha": RIDGE_ALPHA,
            "target_intercept": "none; target mean is removed for fit and not added to prediction",
            "confidence_quantile": CONFIDENCE_QUANTILE,
            "risk": {
                "lookback_bars_5m": RISK_LOOKBACK_BARS,
                "measure": "max leg RMS of completed 5m log(high/low)",
                "reference_events": RISK_HISTORY_EVENTS,
                "reference_statistic": "causal prior-event median",
                "gross_scale_clip": [MIN_GROSS_SCALE, 1.0],
            },
            "base_cost_bp_per_side": BASE_COST_BP,
            "stress_cost_bp_per_side": STRESS_COST_BP,
        },
        "multiplicity_disclosure": MULTIPLICITY_DISCLOSURE,
        "event_counts": {
            "all_frozen_events": int(len(decisions)),
            "flat": int(decisions["choice"].eq("flat").sum()),
            "continuation": int(decisions["choice"].eq("continuation").sum()),
            "reversion": int(decisions["choice"].eq("reversion").sum()),
        },
        "primary": primary,
        "stress_10bp": stress,
        "direction_flip_control": opposite,
        "delay_five_minutes_control": delayed,
        "development_gate": {"checks": checks, "passes": all(checks.values())},
        "selected_trade_rows": primary_raw["combined_2024_2025"]["trade_rows"],
        "next_step": "freeze one exact 2026 support clock and evaluator before reading any 2026 post-entry return",
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    docs_path = Path(docs_output)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(_markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--docs-output", default=DOCS_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.docs_output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
