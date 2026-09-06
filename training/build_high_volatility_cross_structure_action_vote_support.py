"""Materialize deterministic, outcome-blind source support for frozen HVCAV-8."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from training import build_high_volatility_state_ordered_filter_support as hvsof
from training import preregister_high_volatility_cross_structure_action_vote as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PREREG_SHA = "340627ccd4928acb6297f0959fd001cc07066e05e00bfde98db88ae0cb0c550e"
CLOCK = Path("data/high_volatility_cross_structure_action_vote_clocks_2023_2026.csv.gz")
RESULT = Path("results/high_volatility_cross_structure_action_vote_support_2026-08-16.json")
HOLD = pd.Timedelta("8h")
ENTRY_DELAY = pd.Timedelta("5m")
ACTION_COLUMNS = hvsof.ACTION_COLUMNS
STATE_COLUMNS = ("decision_time", "source_valid", "concentration_rank")
OUTPUT_COLUMNS = (
    "candidate",
    "control",
    "split",
    "eligibility_id",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "active_action_count",
    "long_vote_count",
    "short_vote_count",
)

sha256_file = hvsof.sha256_file
canonical_hash = hvsof.canonical_hash


def _read_registration() -> Mapping[str, Any]:
    if sha256_file(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVCAV-8 preregistration artifact drift")
    value = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    if not isinstance(value, dict):
        raise TypeError("HVCAV-8 expected preregistration JSON object")
    prereg.validate(value)
    return value


def verify_frozen_inputs() -> dict[str, Any]:
    """Verify preregistration and component pass artifacts without loading sealed rows."""
    _read_registration()
    verified: dict[str, Any] = {}
    bindings = {**prereg.ACTION_ARTIFACTS, **prereg.ELIGIBILITY_ARTIFACTS}
    for component, artifacts in bindings.items():
        component_result: dict[str, Any] = {}
        for artifact_type, artifact in artifacts.items():
            actual = sha256_file(artifact["path"])
            if actual != artifact["sha256"]:
                raise RuntimeError(
                    f"HVCAV-8 {component} {artifact_type} artifact drift"
                )
            component_result[artifact_type] = {
                "path": artifact["path"],
                "sha256": actual,
                "verified": True,
            }

        support = hvsof._read_top_level_scalars(
            Path(artifacts["support"]["path"]), ("policy_id", "support_passed")
        )
        if support != {"policy_id": component, "support_passed": True}:
            raise RuntimeError(f"HVCAV-8 {component} frozen source support did not pass")
        gross9 = hvsof._read_top_level_scalars(
            Path(artifacts["gross9"]["path"]),
            (
                "policy_id",
                "source_support_passed",
                "every_gross9_sleeve_passed",
                "gross9_novelty_status",
            ),
        )
        if gross9 != {
            "policy_id": component,
            "source_support_passed": True,
            "every_gross9_sleeve_passed": True,
            "gross9_novelty_status": "passed",
        }:
            raise RuntimeError(f"HVCAV-8 {component} frozen Gross9 did not pass")
        component_result["source_support_passed"] = True
        component_result["gross9_passed"] = True
        verified[component] = component_result
    return verified


def load_action_clock(action: str) -> pd.DataFrame:
    """Load only the frozen primary action-clock fields."""
    if action not in prereg.ACTION_ARTIFACTS:
        raise ValueError(f"unknown HVCAV-8 action: {action}")
    frame = pd.read_csv(
        prereg.ACTION_ARTIFACTS[action]["clock"]["path"],
        usecols=list(ACTION_COLUMNS),
    )
    for column in (
        "decision_time",
        "feature_available_time",
        "entry_time",
        "exit_time",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["side"] = pd.to_numeric(frame["side"], errors="raise").astype(int)
    return _validate_action_clock(frame, action)


def _stage_for_decision(decision_time: pd.Timestamp) -> str | None:
    entry_time = decision_time + ENTRY_DELAY
    exit_time = entry_time + HOLD
    for stage, bounds in prereg.build()["stages"].items():
        start, end = (pd.Timestamp(value) for value in bounds)
        if entry_time >= start and exit_time <= end:
            return stage
    return None


def _validate_action_clock(frame: pd.DataFrame, action: str) -> pd.DataFrame:
    ordered = hvsof._validate_reserved_action_clock(frame.copy(), action)
    decision = ordered["decision_time"]
    if not decision.dt.minute.eq(0).all() or not decision.dt.second.eq(0).all():
        raise RuntimeError(f"HVCAV-8 {action} action is off exact 8h grid")
    if not decision.dt.hour.isin((0, 8, 16)).all():
        raise RuntimeError(f"HVCAV-8 {action} action is off exact 8h grid")
    return ordered


def load_eligibility_state() -> pd.DataFrame:
    """Load only the frozen HVTCCR source-valid concentration state."""
    artifact = prereg.ELIGIBILITY_ARTIFACTS[prereg.ELIGIBILITY_ID]["state_panel"]
    frame = pd.read_csv(artifact["path"], usecols=list(STATE_COLUMNS))
    frame["decision_time"] = pd.to_datetime(
        frame["decision_time"], utc=True, errors="raise"
    )
    normalized = frame["source_valid"].astype(str).str.lower()
    if not normalized.isin(("true", "false")).all():
        raise RuntimeError("HVCAV-8 HVTCCR-8 invalid source_valid")
    frame["source_valid"] = normalized.eq("true")
    frame["concentration_rank"] = pd.to_numeric(
        frame["concentration_rank"], errors="coerce"
    )
    return _validate_state_grid(frame)


def _validate_state_grid(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != list(STATE_COLUMNS):
        raise RuntimeError("HVCAV-8 HVTCCR-8 state schema drift")
    if frame.empty or frame["decision_time"].duplicated().any():
        raise RuntimeError("HVCAV-8 HVTCCR-8 invalid state decision grid")
    ordered = frame.sort_values("decision_time", kind="stable").reset_index(drop=True)
    decision = ordered["decision_time"]
    if not decision.dt.minute.eq(0).all() or not decision.dt.second.eq(0).all():
        raise RuntimeError("HVCAV-8 HVTCCR-8 decision is off exact 8h grid")
    if not decision.dt.hour.isin((0, 8, 16)).all():
        raise RuntimeError("HVCAV-8 HVTCCR-8 decision is off exact 8h grid")
    if len(ordered) > 1 and not decision.diff().iloc[1:].eq(HOLD).all():
        raise RuntimeError("HVCAV-8 HVTCCR-8 state grid is not contiguous exact 8h")
    return ordered


def build_action_vote_clock(
    action_clocks: Mapping[str, pd.DataFrame], eligibility_state: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Vote active exact-decision sides at each source-valid HVTCCR decision."""
    if set(action_clocks) != set(prereg.ACTION_ORDER):
        raise RuntimeError("HVCAV-8 requires all and only five frozen action clocks")
    state = _validate_state_grid(eligibility_state.copy())
    state = state.loc[
        state["source_valid"] & state["concentration_rank"].ge(0.80),
        ["decision_time"],
    ].copy()
    state["split"] = state["decision_time"].map(_stage_for_decision)
    state = state.loc[state["split"].notna()].reset_index(drop=True)

    by_action: dict[str, pd.DataFrame] = {}
    for action in prereg.ACTION_ORDER:
        clock = _validate_action_clock(action_clocks[action], action)
        by_action[action] = clock.set_index("decision_time", verify_integrity=True)

    rows: list[dict[str, Any]] = []
    action_vote_counts = {action: 0 for action in prereg.ACTION_ORDER}
    active_count_distribution = {str(count): 0 for count in range(6)}
    no_quorum_decisions = 0
    tie_decisions = 0
    for decision, split in state.itertuples(index=False, name=None):
        active = [
            (action, by_action[action].loc[decision])
            for action in prereg.ACTION_ORDER
            if decision in by_action[action].index
        ]
        active_count_distribution[str(len(active))] += 1
        for action, _ in active:
            action_vote_counts[action] += 1
        if len(active) < 2:
            no_quorum_decisions += 1
            continue

        sides = [int(source["side"]) for _, source in active]
        side = prereg.action_vote_side(sides)
        if side == 0:
            tie_decisions += 1
            continue
        if any(source["split"] != split for _, source in active):
            raise RuntimeError("HVCAV-8 active action split drift")
        entry_time = decision + ENTRY_DELAY
        exit_time = entry_time + HOLD
        if any(source["entry_time"] != entry_time for _, source in active):
            raise RuntimeError("HVCAV-8 active action entry drift")
        if any(source["exit_time"] != exit_time for _, source in active):
            raise RuntimeError("HVCAV-8 active action hold drift")
        rows.append(
            {
                "candidate": prereg.POLICY_ID,
                "control": "primary",
                "split": split,
                "eligibility_id": prereg.ELIGIBILITY_ID,
                "decision_time": decision,
                "feature_available_time": max(
                    source["feature_available_time"] for _, source in active
                ),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": side,
                "active_action_count": len(active),
                "long_vote_count": sum(value == 1 for value in sides),
                "short_vote_count": sum(value == -1 for value in sides),
            }
        )

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if not output.empty:
        output = output.sort_values("decision_time", kind="stable").reset_index(
            drop=True
        )
        if output["decision_time"].duplicated().any():
            raise RuntimeError("HVCAV-8 duplicate voted decision")
        if not output["entry_time"].eq(output["decision_time"] + ENTRY_DELAY).all():
            raise RuntimeError("HVCAV-8 voted entry drift")
        if not output["exit_time"].eq(output["entry_time"] + HOLD).all():
            raise RuntimeError("HVCAV-8 voted hold drift")
        if not output["feature_available_time"].le(output["entry_time"]).all():
            raise RuntimeError("HVCAV-8 voted action unavailable at entry")
        if not output["active_action_count"].ge(2).all():
            raise RuntimeError("HVCAV-8 voted without quorum")
        if not (
            output["long_vote_count"] + output["short_vote_count"]
        ).eq(output["active_action_count"]).all():
            raise RuntimeError("HVCAV-8 action vote accounting drift")
        if len(output) > 1 and not output["entry_time"].iloc[1:].reset_index(
            drop=True
        ).ge(output["exit_time"].iloc[:-1].reset_index(drop=True)).all():
            raise RuntimeError("HVCAV-8 voted clock overlap")

    accounting = {
        "eligible_state_decisions": len(state),
        "majority_action_decisions": len(output),
        "no_quorum_decisions": no_quorum_decisions,
        "tie_decisions": tie_decisions,
        "action_vote_counts": action_vote_counts,
        "active_action_count_distribution": active_count_distribution,
        "long_majority_decisions": int(output["side"].eq(1).sum()),
        "short_majority_decisions": int(output["side"].eq(-1).sum()),
    }
    if (
        accounting["majority_action_decisions"]
        + accounting["no_quorum_decisions"]
        + accounting["tie_decisions"]
        != accounting["eligible_state_decisions"]
    ):
        raise RuntimeError("HVCAV-8 decision accounting drift")
    if sum(active_count_distribution.values()) != len(state):
        raise RuntimeError("HVCAV-8 active-count accounting drift")
    return output, accounting


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    return hvsof.support_stats(clock, split)


def _support_checks(stats: Mapping[str, Mapping[str, float | int]]) -> dict[str, bool]:
    gates = prereg.build()["source_support_gates"]
    checks: dict[str, bool] = {}
    for split, values in stats.items():
        checks[f"{split}_minimum_events"] = (
            values["events"] >= gates["minimum_events"][split]
        )
        checks[f"{split}_side_balance"] = (
            values["minority_side_share"] >= gates["minority_side_share_min"]
        )
        checks[f"{split}_month_concentration"] = (
            values["max_month_share"] <= gates["max_month_share"]
        )
    return checks


def run(clock_path: Path = CLOCK, result_path: Path = RESULT) -> dict[str, Any]:
    verified = verify_frozen_inputs()
    actions = {action: load_action_clock(action) for action in prereg.ACTION_ORDER}
    state = load_eligibility_state()
    clock, accounting = build_action_vote_clock(actions, state)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(clock, clock_path)

    stats = {
        split: support_stats(clock, split) for split in prereg.build()["stages"]
    }
    checks = _support_checks(stats)
    passed = all(checks.values())
    registration = _read_registration()
    core = {
        "protocol_version": "hvcav_8_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "verified_component_artifacts": verified,
        "action_primary_clock_fields_opened": list(ACTION_COLUMNS),
        "eligibility_source_state_fields_opened": list(STATE_COLUMNS),
        "action_vote_incidence_opened": True,
        "action_vote_postentry_returns_or_pnl_opened": False,
        "entry_exit_prices_opened": False,
        "returns_opened": False,
        "funding_opened": False,
        "pnl_opened": False,
        "gross9_comparator_rows_opened": False,
        "additional_reservation_applied": False,
        "cash_materialized_as_position": False,
        "clock": {
            "path": str(clock_path),
            "sha256": sha256_file(clock_path),
            "rows": len(clock),
        },
        "vote_accounting": accounting,
        "support": stats,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_gross9_novelty"
        if passed
        else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clock", type=Path, default=CLOCK)
    parser.add_argument("--result", type=Path, default=RESULT)
    args = parser.parse_args()
    report = run(args.clock, args.result)
    print(
        json.dumps(
            {
                "events": {
                    split: values["events"]
                    for split, values in report["support"].items()
                },
                "vote_accounting": report["vote_accounting"],
                "support_passed": report["support_passed"],
            },
            indent=2,
        )
    )
