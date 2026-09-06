"""Materialize deterministic, outcome-blind source support for HVEIV-24."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from training import build_high_volatility_state_ordered_filter_support as hvsof
from training import preregister_high_volatility_ehlers_imi_veto as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PREREG_SHA = "09992a113faa45c9ef420ba40f1ff9363e2ee3f2f11548315056b1e1730bb60d"
CLOCK = Path("data/high_volatility_ehlers_imi_veto_clocks_2023_2026.csv.gz")
RESULT = Path("results/high_volatility_ehlers_imi_veto_support_2026-08-16.json")
ENTRY_DELAY = pd.Timedelta("5m")
HOLD = pd.Timedelta("24h")
FIELDS = (
    "candidate", "control", "split", "feature_available_time",
    "entry_time", "exit_time", "side",
)


def _registration() -> dict[str, Any]:
    if prereg.sha256_file(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVEIV-24 preregistration hash drift")
    value = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(value)
    return value


def verify_inputs() -> dict[str, Any]:
    _registration()
    verified = {}
    for component, artifacts in prereg.COMPONENT_ARTIFACTS.items():
        for kind, artifact in artifacts.items():
            if prereg.sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(f"HVEIV-24 {component} {kind} drift")
        support = hvsof._read_top_level_scalars(Path(artifacts["support"]["path"]), ("policy_id", "support_passed"))
        gross9 = hvsof._read_top_level_scalars(Path(artifacts["gross9"]["path"]), ("policy_id", "source_support_passed", "every_gross9_sleeve_passed", "gross9_novelty_status"))
        if support != {"policy_id": component, "support_passed": True}:
            raise RuntimeError(f"HVEIV-24 {component} source gate drift")
        if gross9 != {"policy_id": component, "source_support_passed": True, "every_gross9_sleeve_passed": True, "gross9_novelty_status": "passed"}:
            raise RuntimeError(f"HVEIV-24 {component} Gross9 gate drift")
        verified[component] = {kind: {**artifact, "verified": True} for kind, artifact in artifacts.items()}
    return verified


def load_clock(component: str) -> pd.DataFrame:
    frame = pd.read_csv(prereg.COMPONENT_ARTIFACTS[component]["clock"]["path"], usecols=list(FIELDS))
    for column in ("feature_available_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["side"] = pd.to_numeric(frame["side"], errors="raise").astype(int)
    if frame["feature_available_time"].duplicated().any() or not frame["side"].isin((-1, 1)).all():
        raise RuntimeError(f"HVEIV-24 invalid {component} clock")
    if not frame["feature_available_time"].dt.hour.isin((0, 4, 8, 12, 16, 20)).all() or not frame["feature_available_time"].dt.minute.eq(0).all():
        raise RuntimeError(f"HVEIV-24 off-grid {component} clock")
    if not frame["entry_time"].eq(frame["feature_available_time"] + ENTRY_DELAY).all() or not frame["exit_time"].eq(frame["entry_time"] + HOLD).all():
        raise RuntimeError(f"HVEIV-24 {component} timing drift")
    if not frame["feature_available_time"].le(frame["entry_time"]).all():
        raise RuntimeError(f"HVEIV-24 unavailable {component} feature")
    return frame.sort_values("feature_available_time", kind="stable").reset_index(drop=True)


def _stage(decision: pd.Timestamp) -> str | None:
    entry, exit_ = decision + ENTRY_DELAY, decision + ENTRY_DELAY + HOLD
    for stage, (start, end) in prereg.build()["stages"].items():
        if pd.Timestamp(start) <= entry and exit_ <= pd.Timestamp(end):
            return stage
    return None


def build_clock(clocks: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if set(clocks) != set(prereg.COMPONENT_IDS):
        raise RuntimeError("HVEIV-24 component roster drift")
    indexed = {key: {row.feature_available_time: row for row in value.itertuples(index=False)} for key, value in clocks.items()}
    primary_id, veto_id = prereg.ACTION_IDS
    decisions = sorted(indexed[primary_id])
    rows = []
    counts: dict[str, Any] = {
        "primary_decisions": 0, "emitted_decisions": 0,
        "veto_absent": 0, "veto_same_side": 0, "veto_opposite_side": 0,
        "longs": 0, "shorts": 0,
        "active_count_distribution": {"1": 0, "2": 0},
        "action_input_counts": {action: len(indexed[action]) for action in prereg.ACTION_IDS},
    }
    for decision in decisions:
        split = _stage(decision)
        if split is None:
            continue
        counts["primary_decisions"] += 1
        active = [indexed[action][decision] for action in prereg.ACTION_IDS if decision in indexed[action]]
        counts["active_count_distribution"][str(len(active))] += 1
        primary = indexed[primary_id][decision]
        veto = indexed[veto_id].get(decision)
        entry, exit_ = decision + ENTRY_DELAY, decision + ENTRY_DELAY + HOLD
        if primary.split != split or primary.feature_available_time > entry:
            raise RuntimeError("HVEIV-24 primary stage or availability drift")
        if veto is not None and (veto.split != split or veto.feature_available_time > entry):
            raise RuntimeError("HVEIV-24 veto stage or availability drift")
        indicator_veto_side = 0 if veto is None else veto.side
        side = prereg.indicator_veto_side(primary.side, indicator_veto_side)
        if veto is None:
            counts["veto_absent"] += 1
        elif veto.side == primary.side:
            counts["veto_same_side"] += 1
        else:
            counts["veto_opposite_side"] += 1
        if side == 0:
            continue
        rows.append({"candidate": prereg.POLICY_ID, "control": "primary", "split": split, "selected_action": primary_id, "decision_time": decision, "feature_available_time": primary.feature_available_time, "entry_time": entry, "exit_time": exit_, "side": side, "active_action_count": len(active)})
        counts["longs" if side == 1 else "shorts"] += 1
    columns = ("candidate", "control", "split", "selected_action", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "active_action_count")
    output = pd.DataFrame(rows, columns=columns).sort_values("decision_time", kind="stable").reset_index(drop=True)
    counts["emitted_decisions"] = len(output)
    if counts["emitted_decisions"] + counts["veto_opposite_side"] != counts["primary_decisions"]:
        raise RuntimeError("HVEIV-24 incidence accounting drift")
    if len(output) > 1 and not output["entry_time"].iloc[1:].reset_index(drop=True).ge(output["exit_time"].iloc[:-1].reset_index(drop=True)).all():
        raise RuntimeError("HVEIV-24 output overlap")
    return output, counts


def run(clock_path: Path = CLOCK, result_path: Path = RESULT) -> dict[str, Any]:
    verified = verify_inputs()
    clocks = {component: load_clock(component) for component in prereg.COMPONENT_IDS}
    clock, accounting = build_clock(clocks)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(clock, clock_path)
    stats = {stage: hvsof.support_stats(clock, stage) for stage in prereg.build()["stages"]}
    gates = prereg.build()["source_support_gates"]
    checks = {}
    for stage, values in stats.items():
        checks[f"{stage}_minimum_events"] = values["events"] >= gates["minimum_events"][stage]
        checks[f"{stage}_side_balance"] = values["minority_side_share"] >= gates["minority_side_share_min"]
        checks[f"{stage}_month_concentration"] = values["max_month_share"] <= gates["max_month_share"]
    passed = all(checks.values())
    registration = _registration()
    core = {"protocol_version": "hveiv_8_source_support_v1", "policy_id": prereg.POLICY_ID, "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "verified_component_artifacts": verified, "component_clock_fields_opened": list(FIELDS), "combined_incidence_opened": True, "combined_postentry_returns_or_pnl_opened": False, "entry_exit_prices_opened": False, "returns_opened": False, "funding_opened": False, "pnl_opened": False, "gross9_comparator_rows_opened": False, "clock": {"path": str(clock_path), "sha256": prereg.sha256_file(clock_path), "rows": len(clock)}, "indicator_veto_accounting": accounting, "support": stats, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_gross9_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clock", type=Path, default=CLOCK)
    parser.add_argument("--result", type=Path, default=RESULT)
    args = parser.parse_args()
    report = run(args.clock, args.result)
    print(json.dumps({"events": {k: v["events"] for k, v in report["support"].items()}, "accounting": report["indicator_veto_accounting"], "support_passed": report["support_passed"]}, indent=2))
