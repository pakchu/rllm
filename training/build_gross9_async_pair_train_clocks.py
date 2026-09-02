"""Materialize train-only asynchronous Gross9-like component pair clocks.

This source-support pass is deliberately outcome-blind: it authenticates bound
component artifacts, opens only primary clock fields, keeps only the frozen train
prefix, and writes one deterministic eight-hour hold clock for every unordered
component pair.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from training import preregister_gross9_async_pair_search as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


POLICY_ID = prereg.POLICY_ID
PROTOCOL_VERSION = "gross9_async_pair_train_clocks_v1"
TRAIN_START = pd.Timestamp("2023-07-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2024-01-01T00:00:00Z")
CONFIRMATION_WINDOW = pd.Timedelta("6h")
HOLD = pd.Timedelta("8h")
CLOCK_DIR = Path("data/gross9_async_pair_train_clocks_2026-09-02")
RESULT = Path("results/gross9_async_pair_train_clock_source_support_2026-09-02.json")
COMMON_CLOCK_FIELDS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
)
OUTPUT_COLUMNS = (
    "candidate",
    "control",
    "split",
    "left_component_id",
    "right_component_id",
    "trigger_component_id",
    "confirming_component_id",
    "trigger_entry_time",
    "confirming_entry_time",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
)
SOURCE_GATES = {
    "minimum_events": 8,
    "minority_side_share_min": 0.20,
    "max_month_share": 0.45,
}

# Nine immutable source/Gross9-passing component clocks.  The order defines the
# unordered 9C2 family and must not be changed after pair incidence is opened.
COMPONENT_ORDER = prereg.COMPONENT_ORDER
COMPONENT_ARTIFACTS = prereg.COMPONENT_ARTIFACTS


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pair_id(left: str, right: str) -> str:
    return f"{left}__ASYNC_SAME_SIDE_6H__{right}"


def candidate_family(component_order: Sequence[str] = COMPONENT_ORDER) -> tuple[str, ...]:
    family = tuple(
        pair_id(left, right)
        for index, left in enumerate(component_order)
        for right in component_order[index + 1 :]
    )
    if tuple(component_order) == COMPONENT_ORDER and family != prereg.CANDIDATE_FAMILY:
        raise RuntimeError(f"{POLICY_ID} candidate family differs from preregistration")
    return family


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} expected JSON object: {path}")
    return value


def _top_level_scalars(path: str | Path, keys: Sequence[str]) -> dict[str, Any]:
    value = _read_json_object(path)
    return {key: value.get(key) for key in keys}


def verify_bound_component_artifacts(
    component_artifacts: Mapping[str, Mapping[str, Mapping[str, str]]] = COMPONENT_ARTIFACTS,
    component_order: Sequence[str] = COMPONENT_ORDER,
) -> dict[str, Any]:
    """Hash-check all bound component artifacts without loading their clock rows."""
    if len(component_order) != 9 or len(set(component_order)) != 9:
        raise RuntimeError(f"{POLICY_ID} requires exactly nine unique bound components")
    if set(component_artifacts) != set(component_order):
        raise RuntimeError(f"{POLICY_ID} component artifact roster drift")

    verified: dict[str, Any] = {}
    for component in component_order:
        artifacts = component_artifacts[component]
        component_record: dict[str, Any] = {}
        for kind in (
            "train_economics",
            "preregistration",
            "source_support",
            "gross9",
            "clock",
        ):
            artifact = artifacts.get(kind)
            if not artifact or "path" not in artifact or "sha256" not in artifact:
                raise RuntimeError(f"{POLICY_ID} missing {component} {kind} binding")
            observed = sha256_file(artifact["path"])
            if observed != artifact["sha256"]:
                raise RuntimeError(f"{POLICY_ID} {component} {kind} artifact drift")
            component_record[kind] = {
                "path": artifact["path"],
                "sha256": observed,
                "verified": True,
            }

        train_economics = _top_level_scalars(
            artifacts["train_economics"]["path"],
            (
                "policy_id",
                "stage",
                "passed",
                "decision",
                "later_stage_outcomes_opened",
            ),
        )
        support = _top_level_scalars(
            artifacts["source_support"]["path"],
            ("policy_id", "support_passed"),
        )
        gross9 = _top_level_scalars(
            artifacts["gross9"]["path"],
            (
                "policy_id",
                "source_support_passed",
                "every_gross9_sleeve_passed",
                "gross9_novelty_status",
            ),
        )
        if train_economics != {
            "policy_id": component,
            "stage": "train",
            "passed": True,
            "decision": "pass",
            "later_stage_outcomes_opened": False,
        }:
            raise RuntimeError(f"{POLICY_ID} {component} train economics is not bound-pass")
        if support != {"policy_id": component, "support_passed": True}:
            raise RuntimeError(f"{POLICY_ID} {component} source support is not bound-pass")
        if gross9 != {
            "policy_id": component,
            "source_support_passed": True,
            "every_gross9_sleeve_passed": True,
            "gross9_novelty_status": "passed",
        }:
            raise RuntimeError(f"{POLICY_ID} {component} Gross9 novelty is not bound-pass")
        component_record["source_support_passed"] = True
        component_record["gross9_passed"] = True
        component_record["train_economics_passed"] = True
        verified[component] = component_record
    return verified


def _parse_timestamp(raw: str, column: str) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        raise RuntimeError(f"{POLICY_ID} {column} must be timezone-aware")
    return ts.tz_convert("UTC") if hasattr(ts, "tz_convert") else ts


def _row_before_train_end(row: Mapping[str, str]) -> bool:
    # Component clock timestamps use an ISO-like year-first representation, so
    # this check avoids parsing OOS rows after the frozen prefix boundary.
    return row["entry_time"] < "2024-01-01"


def load_train_prefix_clock(
    component: str,
    component_artifacts: Mapping[str, Mapping[str, Mapping[str, str]]] = COMPONENT_ARTIFACTS,
) -> pd.DataFrame:
    """Load only common primary-clock fields for [2023-07-01, 2024-01-01)."""
    if component not in component_artifacts:
        raise ValueError(f"unknown {POLICY_ID} component: {component}")
    path = Path(component_artifacts[component]["clock"]["path"])
    rows: list[dict[str, Any]] = []
    raw_rows_seen = 0
    stopped_at_train_end = False
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(field not in reader.fieldnames for field in COMMON_CLOCK_FIELDS):
            raise RuntimeError(f"{POLICY_ID} {component} clock schema drift")
        for row in reader:
            raw_rows_seen += 1
            if not _row_before_train_end(row):
                stopped_at_train_end = True
                break
            entry = _parse_timestamp(row["entry_time"], "entry_time")
            exit_time = _parse_timestamp(row["exit_time"], "exit_time")
            if entry < TRAIN_START or entry >= TRAIN_END or exit_time > TRAIN_END:
                continue
            parsed = {field: row[field] for field in COMMON_CLOCK_FIELDS}
            for column in ("decision_time", "feature_available_time", "entry_time", "exit_time"):
                parsed[column] = _parse_timestamp(parsed[column], column)
            parsed["side"] = int(parsed["side"])
            rows.append(parsed)
    frame = pd.DataFrame(rows, columns=COMMON_CLOCK_FIELDS)
    if frame.empty:
        raise RuntimeError(f"{POLICY_ID} {component} has no train-prefix primary rows")
    if not frame["candidate"].eq(component).all() or not frame["control"].eq("primary").all():
        raise RuntimeError(f"{POLICY_ID} {component} identity/control drift")
    if not frame["split"].eq("train").all():
        raise RuntimeError(f"{POLICY_ID} {component} non-train split opened")
    if not frame["side"].isin((-1, 1)).all():
        raise RuntimeError(f"{POLICY_ID} {component} non-strict side")
    if frame["entry_time"].duplicated().any():
        raise RuntimeError(f"{POLICY_ID} {component} duplicate entry_time")
    if not frame["decision_time"].le(frame["entry_time"]).all():
        raise RuntimeError(f"{POLICY_ID} {component} decision after entry")
    if not frame["feature_available_time"].le(frame["entry_time"]).all():
        raise RuntimeError(f"{POLICY_ID} {component} feature unavailable at entry")
    if not frame["entry_time"].ge(TRAIN_START).all() or not frame["entry_time"].lt(TRAIN_END).all():
        raise RuntimeError(f"{POLICY_ID} {component} loaded OOS entry")
    frame.attrs["raw_rows_seen_until_stop"] = raw_rows_seen
    frame.attrs["stopped_at_train_end"] = stopped_at_train_end
    return frame.sort_values("entry_time", kind="stable").reset_index(drop=True)


def reserve_half_open(clock: pd.DataFrame) -> pd.DataFrame:
    ordered = clock.sort_values(["entry_time", "candidate"], kind="stable")
    keep: list[int] = []
    next_available: pd.Timestamp | None = None
    for index, row in ordered.iterrows():
        if next_available is not None and row["entry_time"] < next_available:
            continue
        keep.append(index)
        next_available = row["exit_time"]
    return ordered.loc[keep].reset_index(drop=True)


def build_async_pair_clock(left: str, right: str, left_clock: pd.DataFrame, right_clock: pd.DataFrame) -> pd.DataFrame:
    """Build symmetric 6h same-side confirmations for one unordered pair."""
    for component, clock in ((left, left_clock), (right, right_clock)):
        if clock.empty or not clock["candidate"].eq(component).all():
            raise RuntimeError(f"{POLICY_ID} invalid component clock for {component}")

    by_component = {left: left_clock, right: right_clock}
    rows: list[dict[str, Any]] = []
    seen_simultaneous: set[tuple[pd.Timestamp, int]] = set()
    events = pd.concat(
        [left_clock.assign(_component=left), right_clock.assign(_component=right)],
        ignore_index=True,
    ).sort_values(["entry_time", "_component"], kind="stable")

    for _, event in events.iterrows():
        trigger = str(event["_component"])
        other = right if trigger == left else left
        other_clock = by_component[other]
        candidates = other_clock.loc[
            other_clock["side"].eq(event["side"])
            & other_clock["entry_time"].le(event["entry_time"])
            & other_clock["entry_time"].ge(event["entry_time"] - CONFIRMATION_WINDOW)
        ]
        if candidates.empty:
            continue
        confirmer = candidates.iloc[-1]
        simultaneous_key = (event["entry_time"], int(event["side"]))
        if confirmer["entry_time"] == event["entry_time"]:
            if simultaneous_key in seen_simultaneous:
                continue
            seen_simultaneous.add(simultaneous_key)
            trigger_component = left
            confirming_component = right
            trigger_row = left_clock[left_clock["entry_time"].eq(event["entry_time"])].iloc[0]
            confirming_row = right_clock[right_clock["entry_time"].eq(event["entry_time"])].iloc[0]
        else:
            trigger_component = trigger
            confirming_component = other
            trigger_row = event.drop(labels=["_component"])
            confirming_row = confirmer

        entry_time = pd.Timestamp(event["entry_time"])
        rows.append(
            {
                "candidate": pair_id(left, right),
                "control": "primary",
                "split": "train",
                "left_component_id": left,
                "right_component_id": right,
                "trigger_component_id": trigger_component,
                "confirming_component_id": confirming_component,
                "trigger_entry_time": trigger_row["entry_time"],
                "confirming_entry_time": confirming_row["entry_time"],
                "decision_time": max(trigger_row["decision_time"], confirming_row["decision_time"]),
                "feature_available_time": max(
                    trigger_row["feature_available_time"],
                    confirming_row["feature_available_time"],
                ),
                "entry_time": entry_time,
                "exit_time": entry_time + HOLD,
                "side": int(event["side"]),
            }
        )

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if output.empty:
        return output
    output = reserve_half_open(output)
    if not output["entry_time"].ge(TRAIN_START).all() or not output["exit_time"].le(TRAIN_END).all():
        raise RuntimeError(f"{POLICY_ID} pair clock escaped train prefix")
    if not output["decision_time"].le(output["entry_time"]).all():
        raise RuntimeError(f"{POLICY_ID} pair decision after entry")
    if not output["feature_available_time"].le(output["entry_time"]).all():
        raise RuntimeError(f"{POLICY_ID} pair feature after entry")
    if not output["side"].isin((-1, 1)).all():
        raise RuntimeError(f"{POLICY_ID} pair non-strict side")
    lag = output["entry_time"] - output["confirming_entry_time"]
    if not (lag.ge(pd.Timedelta(0)).all() and lag.le(CONFIRMATION_WINDOW).all()):
        raise RuntimeError(f"{POLICY_ID} pair confirmation lag drift")
    if len(output) > 1 and not output["entry_time"].iloc[1:].reset_index(drop=True).ge(
        output["exit_time"].iloc[:-1].reset_index(drop=True)
    ).all():
        raise RuntimeError(f"{POLICY_ID} pair reservation overlap")
    return output


def support_stats(clock: pd.DataFrame) -> dict[str, float | int]:
    if clock.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(clock["side"].eq(1).sum())
    shorts = int(clock["side"].eq(-1).sum())
    events = len(clock)
    month_counts = clock["entry_time"].dt.strftime("%Y-%m").value_counts()
    return {
        "events": events,
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / events,
        "max_month_share": int(month_counts.max()) / events,
    }


def support_checks(stats: Mapping[str, float | int]) -> dict[str, bool]:
    return {
        "minimum_events": stats["events"] >= SOURCE_GATES["minimum_events"],
        "side_balance": stats["minority_side_share"] >= SOURCE_GATES["minority_side_share_min"],
        "month_concentration": stats["max_month_share"] <= SOURCE_GATES["max_month_share"],
    }


def run(clock_dir: Path = CLOCK_DIR, result_path: Path = RESULT) -> dict[str, Any]:
    if not prereg.DEFAULT_OUTPUT.is_file():
        raise RuntimeError(f"{POLICY_ID} missing committed preregistration artifact")
    registration = _read_json_object(prereg.DEFAULT_OUTPUT)
    prereg.validate(registration)
    if registration != prereg.build():
        raise RuntimeError(f"{POLICY_ID} preregistration artifact differs from code")
    verified = verify_bound_component_artifacts()
    clocks = {component: load_train_prefix_clock(component) for component in COMPONENT_ORDER}
    clock_dir.mkdir(parents=True, exist_ok=True)

    pairs: dict[str, Any] = {}
    passed_pairs: list[str] = []
    for index, left in enumerate(COMPONENT_ORDER):
        for right in COMPONENT_ORDER[index + 1 :]:
            candidate = pair_id(left, right)
            clock = build_async_pair_clock(left, right, clocks[left], clocks[right])
            path = clock_dir / f"{candidate}.csv.gz"
            _write_gzip_csv(clock, path)
            stats = support_stats(clock)
            checks = support_checks(stats)
            passed = all(checks.values())
            if passed:
                passed_pairs.append(candidate)
            pairs[candidate] = {
                "components": [left, right],
                "operator": "symmetric_async_same_side_latest_other_within_6h",
                "clock": {"path": str(path), "sha256": sha256_file(path), "rows": len(clock)},
                "support": stats,
                "support_checks": checks,
                "support_passed": passed,
                "advance_to_gross9_novelty": passed,
                "advance_to_economic_outcomes": False,
                "decision": "pass_to_gross9_novelty" if passed else "terminal_source_support_reject",
            }

    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": "2026-09-02",
        "component_order": list(COMPONENT_ORDER),
        "candidate_family": list(candidate_family()),
        "candidate_family_size": 36,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha256_file(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "verified_component_artifacts": verified,
        "train_prefix": {
            "start_inclusive": TRAIN_START.isoformat().replace("+00:00", "Z"),
            "end_exclusive": TRAIN_END.isoformat().replace("+00:00", "Z"),
            "component_rows_loaded": {component: len(clock) for component, clock in clocks.items()},
        },
        "construction": {
            "operator": "for each unordered pair, emit at later same-side component entry t when the other component's latest same-side entry is in [t-6h,t]",
            "simultaneous_dedupe": True,
            "decision_time": "max(trigger decision_time, confirming decision_time)",
            "feature_available_time": "max(trigger feature_available_time, confirming feature_available_time)",
            "entry_time": "later component entry_time t",
            "exit_time": "entry_time + 8 elapsed hours",
            "reservation": "global chronological half-open reservation per pair after candidate materialization",
        },
        "source_support_gates": SOURCE_GATES,
        "pairs": pairs,
        "passed_pairs": passed_pairs,
        "support_passed_any_pair": bool(passed_pairs),
        "evidence_boundary": {
            "component_clock_fields_opened": list(COMMON_CLOCK_FIELDS),
            "component_clock_rows_materialized_train_prefix_only": True,
            "first_boundary_row_inspected_only": True,
            "gross9_rows_opened": False,
            "market_rows_opened": False,
            "entry_exit_prices_opened": False,
            "funding_opened": False,
            "pair_combination_returns_or_pnl_opened": False,
            "pair_combination_economic_outcomes_opened": False,
            "bound_component_train_economics_artifact_opened_for_pass_authentication": True,
            "component_metric_fields_used_for_pair_selection": [],
            "oos_component_rows_materialized": 0,
        },
        "decision": "pass_supported_pairs_to_gross9_novelty" if passed_pairs else "terminal_no_source_supported_pairs",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clock-dir", type=Path, default=CLOCK_DIR)
    parser.add_argument("--result", type=Path, default=RESULT)
    args = parser.parse_args()
    report = run(args.clock_dir, args.result)
    print(json.dumps({"passed_pairs": report["passed_pairs"], "candidate_family_size": report["candidate_family_size"]}, indent=2))
