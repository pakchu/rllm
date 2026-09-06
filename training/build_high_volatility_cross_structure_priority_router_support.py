"""Materialize deterministic, outcome-blind source support for frozen HVCSPR-8."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from training import build_high_volatility_state_ordered_filter_support as hvsof
from training import (
    preregister_high_volatility_cross_structure_priority_router as prereg,
)
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PREREG_SHA = "d31652fd7bf80b9a2da73a15f4c4235d08b2f40ae583a0f90fd3b32dfd775bc0"
CLOCK_DIR = Path(
    "data/high_volatility_cross_structure_priority_router_clocks_2023_2026"
)
RESULT = Path(
    "results/high_volatility_cross_structure_priority_router_support_2026-08-14.json"
)
HOLD = hvsof.HOLD
ENTRY_DELAY = hvsof.ENTRY_DELAY
ACTION_COLUMNS = hvsof.ACTION_COLUMNS
FILTER_COLUMNS = hvsof.FILTER_COLUMNS
OUTPUT_COLUMNS = (
    "candidate",
    "control",
    "split",
    "action_id",
    "eligibility_id",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
)


sha256_file = hvsof.sha256_file
canonical_hash = hvsof.canonical_hash


def _read_registration() -> Mapping[str, Any]:
    if sha256_file(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVCSPR-8 preregistration artifact drift")
    value = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    if not isinstance(value, dict):
        raise TypeError("HVCSPR-8 expected preregistration JSON object")
    prereg.validate(value)
    return value


def verify_frozen_inputs() -> dict[str, Any]:
    """Verify every frozen component without opening Gross9 comparator rows."""
    _read_registration()
    verified: dict[str, Any] = {}
    bindings = {**prereg.ACTION_ARTIFACTS, **prereg.ELIGIBILITY_ARTIFACTS}
    for component, artifacts in bindings.items():
        component_result: dict[str, Any] = {}
        for artifact_type, artifact in artifacts.items():
            actual = sha256_file(artifact["path"])
            if actual != artifact["sha256"]:
                raise RuntimeError(
                    f"HVCSPR-8 {component} {artifact_type} artifact drift"
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
            raise RuntimeError(
                f"HVCSPR-8 {component} frozen source support did not pass"
            )
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
            raise RuntimeError(f"HVCSPR-8 {component} frozen Gross9 did not pass")
        component_result["source_support_passed"] = True
        component_result["gross9_passed"] = True
        verified[component] = component_result
    return verified


def load_action_clock(action: str) -> pd.DataFrame:
    """Load only the selected-action fields from its frozen primary clock."""
    if action not in prereg.ACTION_ORDER:
        raise ValueError(f"unknown HVCSPR-8 action: {action}")
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
    return hvsof._validate_reserved_action_clock(frame, action)


def _validate_exact_eight_hour_grid(
    frame: pd.DataFrame, eligibility: str
) -> pd.DataFrame:
    expected = list(FILTER_COLUMNS[eligibility])
    if frame.columns.tolist() != expected:
        raise RuntimeError(f"HVCSPR-8 {eligibility} state schema drift")
    if frame.empty or frame["decision_time"].duplicated().any():
        raise RuntimeError(f"HVCSPR-8 {eligibility} invalid state decision grid")
    ordered = frame.sort_values("decision_time", kind="stable").reset_index(drop=True)
    decision = ordered["decision_time"]
    if not decision.dt.minute.eq(0).all() or not decision.dt.second.eq(0).all():
        raise RuntimeError(f"HVCSPR-8 {eligibility} decision is off exact 8h grid")
    if not decision.dt.hour.isin((0, 8, 16)).all():
        raise RuntimeError(f"HVCSPR-8 {eligibility} decision is off exact 8h grid")
    if len(ordered) > 1 and not decision.diff().iloc[1:].eq(HOLD).all():
        raise RuntimeError(
            f"HVCSPR-8 {eligibility} state grid is not contiguous exact 8h"
        )
    return ordered


def load_eligibility_state(eligibility: str) -> pd.DataFrame:
    """Load only the frozen HVSOF eligibility fields and validate its 8h grid."""
    if eligibility not in prereg.ELIGIBILITY_ORDER:
        raise ValueError(f"unknown HVCSPR-8 eligibility: {eligibility}")
    return _validate_exact_eight_hour_grid(
        hvsof.load_filter_state(eligibility), eligibility
    )


def _state_true(frame: pd.DataFrame, eligibility: str) -> pd.Series:
    rank = FILTER_COLUMNS[eligibility][-1]
    if eligibility == "HVTCCR-8":
        return frame["source_valid"] & frame[rank].ge(0.80)
    if eligibility == "HVLZC-8":
        return frame["source_valid"] & frame[rank].le(0.25)
    raise ValueError(f"unknown HVCSPR-8 eligibility: {eligibility}")


def _stage_for_decision(decision_time: pd.Timestamp) -> str | None:
    entry_time = decision_time + ENTRY_DELAY
    exit_time = entry_time + HOLD
    for stage, bounds in prereg.build()["stages"].items():
        start, end = (pd.Timestamp(value) for value in bounds)
        if entry_time >= start and exit_time <= end:
            return stage
    return None


def _validate_action_grid(action_clock: pd.DataFrame, action: str) -> pd.DataFrame:
    actions = hvsof._validate_reserved_action_clock(action_clock.copy(), action)
    decision = actions["decision_time"]
    if not decision.dt.minute.eq(0).all() or not decision.dt.second.eq(0).all():
        raise RuntimeError(f"HVCSPR-8 {action} action is off exact 8h grid")
    if not decision.dt.hour.isin((0, 8, 16)).all():
        raise RuntimeError(f"HVCSPR-8 {action} action is off exact 8h grid")
    return actions


def route_priority_clock(
    priority_order: Sequence[str],
    eligibility: str,
    action_clocks: Mapping[str, pd.DataFrame],
    eligibility_state: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Route each true state to its first exact-decision active action; otherwise cash."""
    priority = tuple(priority_order)
    if priority not in prereg.PRIORITY_ORDERS:
        raise ValueError("HVCSPR-8 priority order must be one of the frozen orders")
    candidate = prereg.candidate_id(priority, eligibility)
    if candidate not in prereg.CANDIDATE_FAMILY:
        raise ValueError("HVCSPR-8 candidate must use the exact frozen family")
    if set(action_clocks) != set(prereg.ACTION_ORDER):
        raise RuntimeError("HVCSPR-8 requires all and only frozen action clocks")

    state = _validate_exact_eight_hour_grid(eligibility_state.copy(), eligibility)
    state = state.loc[_state_true(state, eligibility), ["decision_time"]].copy()
    state["split"] = state["decision_time"].map(_stage_for_decision)
    state = state.loc[state["split"].notna()].reset_index(drop=True)

    by_action: dict[str, pd.DataFrame] = {}
    for action in prereg.ACTION_ORDER:
        clock = _validate_action_grid(action_clocks[action], action)
        by_action[action] = clock.set_index("decision_time", verify_integrity=True)

    rows: list[dict[str, Any]] = []
    selected_counts = {action: 0 for action in priority}
    cash_decisions = 0
    for decision, split in state.itertuples(index=False, name=None):
        selected_action = next(
            (action for action in priority if decision in by_action[action].index), None
        )
        if selected_action is None:
            cash_decisions += 1
            continue
        source = by_action[selected_action].loc[decision]
        if source["split"] != split:
            raise RuntimeError(f"HVCSPR-8 {candidate} chosen action split drift")
        selected_counts[selected_action] += 1
        rows.append(
            {
                "candidate": candidate,
                "control": source["control"],
                "split": source["split"],
                "action_id": selected_action,
                "eligibility_id": eligibility,
                "decision_time": decision,
                "feature_available_time": source["feature_available_time"],
                "entry_time": source["entry_time"],
                "exit_time": source["exit_time"],
                "side": int(source["side"]),
            }
        )
    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if not output.empty:
        output = output.sort_values("decision_time", kind="stable").reset_index(
            drop=True
        )
        if output["decision_time"].duplicated().any():
            raise RuntimeError(f"HVCSPR-8 {candidate} selected more than one action")
        if not output["feature_available_time"].le(output["entry_time"]).all():
            raise RuntimeError(f"HVCSPR-8 {candidate} selected unavailable action")
        if not output["exit_time"].eq(output["entry_time"] + HOLD).all():
            raise RuntimeError(f"HVCSPR-8 {candidate} selected action hold drift")
        if (
            len(output) > 1
            and not output["entry_time"]
            .iloc[1:]
            .reset_index(drop=True)
            .ge(output["exit_time"].iloc[:-1].reset_index(drop=True))
            .all()
        ):
            raise RuntimeError(f"HVCSPR-8 {candidate} exact 8h routing overlap")
    routing = {
        "eligible_state_decisions": len(state),
        "routed_action_decisions": len(output),
        "cash_decisions": cash_decisions,
        "selected_action_counts": selected_counts,
    }
    if (
        routing["routed_action_decisions"] + routing["cash_decisions"]
        != routing["eligible_state_decisions"]
    ):
        raise RuntimeError(f"HVCSPR-8 {candidate} routing accounting drift")
    return output, routing


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


def run(clock_dir: Path = CLOCK_DIR, result_path: Path = RESULT) -> dict[str, Any]:
    verified = verify_frozen_inputs()
    actions = {action: load_action_clock(action) for action in prereg.ACTION_ORDER}
    states = {
        eligibility: load_eligibility_state(eligibility)
        for eligibility in prereg.ELIGIBILITY_ORDER
    }
    clock_dir.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, Any] = {}
    eligible_routers: list[str] = []
    for priority in prereg.PRIORITY_ORDERS:
        for eligibility in prereg.ELIGIBILITY_ORDER:
            candidate = prereg.candidate_id(priority, eligibility)
            clock, routing = route_priority_clock(
                priority, eligibility, actions, states[eligibility]
            )
            path = clock_dir / f"{candidate}.csv.gz"
            _write_gzip_csv(clock, path)
            stats = {
                split: support_stats(clock, split) for split in prereg.build()["stages"]
            }
            checks = _support_checks(stats)
            passed = all(checks.values())
            if passed:
                eligible_routers.append(candidate)
            candidates[candidate] = {
                "priority_order": list(priority),
                "eligibility": eligibility,
                "clock": {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "rows": len(clock),
                },
                "routing": routing,
                "support": stats,
                "support_checks": checks,
                "support_passed": passed,
                "advance_to_combination_gross9": passed,
                "advance_to_economic_outcomes": False,
                "decision": "pass_to_combination_gross9"
                if passed
                else "terminal_source_support_reject",
            }

    registration = _read_registration()
    core = {
        "protocol_version": "hvcspr_8_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "verified_component_artifacts": verified,
        "action_primary_clock_fields_opened": list(ACTION_COLUMNS),
        "eligibility_source_state_fields_opened": {
            key: list(value) for key, value in FILTER_COLUMNS.items()
        },
        "router_incidence_opened": True,
        "router_postentry_returns_or_pnl_opened": False,
        "entry_exit_prices_opened": False,
        "returns_opened": False,
        "funding_opened": False,
        "pnl_opened": False,
        "gross9_comparator_rows_opened": False,
        "additional_reservation_applied": False,
        "cash_materialized_as_position": False,
        "candidates": candidates,
        "eligible_routers_for_combination_gross9": eligible_routers,
        "eligible_router_count": len(eligible_routers),
        "advance_to_combination_gross9": bool(eligible_routers),
        "advance_to_economic_outcomes": False,
        "decision": "eligible_routers_to_combination_gross9"
        if eligible_routers
        else "terminal_no_source_supported_routers",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clock-dir", type=Path, default=CLOCK_DIR)
    parser.add_argument("--result", type=Path, default=RESULT)
    args = parser.parse_args()
    report = run(args.clock_dir, args.result)
    print(
        json.dumps(
            {
                "eligible_routers": report["eligible_routers_for_combination_gross9"],
                "eligible_router_count": report["eligible_router_count"],
            },
            indent=2,
        )
    )
