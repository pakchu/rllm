"""Materialize deterministic, outcome-blind source support for frozen HVSOF-8."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from training import preregister_high_volatility_state_ordered_filter as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


PREREG_SHA = "53e78737a357eb183c8247bbca847a8319334400be5390e4bf01861a368e3484"
CLOCK_DIR = Path("data/high_volatility_state_ordered_filter_clocks_2023_2026")
RESULT = Path("results/high_volatility_state_ordered_filter_support_2026-08-14.json")
HOLD = pd.Timedelta("8h")
ENTRY_DELAY = pd.Timedelta("5m")
ACTION_COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
)
FILTER_COLUMNS = {
    "HVTCCR-8": ("decision_time", "source_valid", "concentration_rank"),
    "HVLZC-8": ("decision_time", "source_valid", "complexity_rank"),
}
OUTPUT_COLUMNS = (
    "candidate",
    "control",
    "split",
    "action_id",
    "filter_id",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
)
_TOP_LEVEL_SCALAR = re.compile(r'^  "(?P<key>[^"]+)": (?P<value>true|false|"[^"]*"),?$')


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"HVSOF-8 expected JSON object: {path}")
    return value


def _read_top_level_scalars(path: Path, required: tuple[str, ...]) -> dict[str, Any]:
    """Read selected top-level pass fields without deserializing sealed nested rows."""
    found: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = _TOP_LEVEL_SCALAR.match(line.rstrip("\n"))
            if match is None or match.group("key") not in required:
                continue
            key = match.group("key")
            if key in found:
                raise RuntimeError(f"HVSOF-8 duplicate top-level field {key}: {path}")
            raw = match.group("value")
            found[key] = raw == "true" if raw in ("true", "false") else raw[1:-1]
    if set(found) != set(required):
        missing = sorted(set(required) - set(found))
        raise RuntimeError(f"HVSOF-8 missing top-level fields {missing}: {path}")
    return found


def verify_frozen_inputs() -> dict[str, Any]:
    """Verify all frozen hashes and component source/Gross9 pass states."""
    if sha256_file(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVSOF-8 preregistration artifact drift")
    registration = _read_json_object(prereg.DEFAULT_OUTPUT)
    prereg.validate(registration)

    verified: dict[str, Any] = {}
    bindings = {**prereg.ACTION_ARTIFACTS, **prereg.ELIGIBILITY_ARTIFACTS}
    for component, artifacts in bindings.items():
        component_result: dict[str, Any] = {}
        for artifact_type, artifact in artifacts.items():
            actual = sha256_file(artifact["path"])
            if actual != artifact["sha256"]:
                raise RuntimeError(
                    f"HVSOF-8 {component} {artifact_type} artifact drift"
                )
            component_result[artifact_type] = {
                "path": artifact["path"],
                "sha256": actual,
                "verified": True,
            }

        support = _read_top_level_scalars(
            Path(artifacts["support"]["path"]), ("policy_id", "support_passed")
        )
        if support != {"policy_id": component, "support_passed": True}:
            raise RuntimeError(f"HVSOF-8 {component} frozen source support did not pass")
        gross9 = _read_top_level_scalars(
            Path(artifacts["gross9"]["path"]),
            (
                "policy_id",
                "source_support_passed",
                "every_gross9_sleeve_passed",
                "gross9_novelty_status",
            ),
        )
        expected_gross9 = {
            "policy_id": component,
            "source_support_passed": True,
            "every_gross9_sleeve_passed": True,
            "gross9_novelty_status": "passed",
        }
        if gross9 != expected_gross9:
            raise RuntimeError(f"HVSOF-8 {component} frozen Gross9 artifact did not pass")
        component_result["source_support_passed"] = True
        component_result["gross9_passed"] = True
        verified[component] = component_result
    return verified


def _stage_for(entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> str | None:
    for stage, bounds in prereg.build()["stages"].items():
        start, end = (pd.Timestamp(value) for value in bounds)
        if entry_time >= start and exit_time <= end:
            return stage
    return None


def _validate_reserved_action_clock(frame: pd.DataFrame, action: str) -> pd.DataFrame:
    if frame.columns.tolist() != list(ACTION_COLUMNS):
        raise RuntimeError(f"HVSOF-8 {action} action clock schema drift")
    if frame.empty:
        raise RuntimeError(f"HVSOF-8 {action} action clock is empty")
    if not frame["candidate"].eq(action).all() or not frame["control"].eq("primary").all():
        raise RuntimeError(f"HVSOF-8 {action} action clock identity drift")
    if frame["decision_time"].duplicated().any() or frame["entry_time"].duplicated().any():
        raise RuntimeError(f"HVSOF-8 {action} duplicate action time")
    if not frame["side"].isin((-1, 1)).all():
        raise RuntimeError(f"HVSOF-8 {action} action clock has non-strict side")
    if not frame["entry_time"].eq(frame["decision_time"] + ENTRY_DELAY).all():
        raise RuntimeError(f"HVSOF-8 {action} action entry drift")
    if not frame["exit_time"].eq(frame["entry_time"] + HOLD).all():
        raise RuntimeError(f"HVSOF-8 {action} action hold drift")
    if not frame["feature_available_time"].le(frame["entry_time"]).all():
        raise RuntimeError(f"HVSOF-8 {action} action feature unavailable at entry")
    expected_splits = [
        _stage_for(entry, exit_time)
        for entry, exit_time in zip(frame["entry_time"], frame["exit_time"])
    ]
    if any(stage is None for stage in expected_splits) or frame["split"].tolist() != expected_splits:
        raise RuntimeError(f"HVSOF-8 {action} action split drift")
    ordered = frame.sort_values("entry_time", kind="stable").reset_index(drop=True)
    if len(ordered) > 1 and not ordered["entry_time"].iloc[1:].reset_index(drop=True).ge(
        ordered["exit_time"].iloc[:-1].reset_index(drop=True)
    ).all():
        raise RuntimeError(f"HVSOF-8 {action} action reservation overlap")
    return ordered


def load_action_clock(action: str) -> pd.DataFrame:
    """Load only the frozen primary action-clock fields."""
    if action not in prereg.ACTION_ARTIFACTS:
        raise ValueError(f"unknown HVSOF-8 action: {action}")
    frame = pd.read_csv(
        prereg.ACTION_ARTIFACTS[action]["clock"]["path"], usecols=list(ACTION_COLUMNS)
    )
    for column in ("decision_time", "feature_available_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["side"] = pd.to_numeric(frame["side"], errors="raise").astype(int)
    return _validate_reserved_action_clock(frame, action)


def load_filter_state(state_filter: str) -> pd.DataFrame:
    """Load only decision_time, source_valid, and the frozen eligibility rank."""
    if state_filter not in prereg.ELIGIBILITY_ARTIFACTS:
        raise ValueError(f"unknown HVSOF-8 filter: {state_filter}")
    columns = FILTER_COLUMNS[state_filter]
    frame = pd.read_csv(
        prereg.ELIGIBILITY_ARTIFACTS[state_filter]["state_panel"]["path"],
        usecols=list(columns),
    )
    if frame.columns.tolist() != list(columns):
        raise RuntimeError(f"HVSOF-8 {state_filter} state schema drift")
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="raise")
    if frame["decision_time"].duplicated().any():
        raise RuntimeError(f"HVSOF-8 {state_filter} duplicate state decision_time")
    normalized = frame["source_valid"].astype(str).str.lower()
    if not normalized.isin(("true", "false")).all():
        raise RuntimeError(f"HVSOF-8 {state_filter} invalid source_valid")
    frame["source_valid"] = normalized.eq("true")
    rank = columns[-1]
    frame[rank] = pd.to_numeric(frame[rank], errors="coerce")
    return frame.sort_values("decision_time", kind="stable").reset_index(drop=True)


def filter_action_clock(
    action: str,
    state_filter: str,
    action_clock: pd.DataFrame,
    filter_state: pd.DataFrame,
) -> pd.DataFrame:
    """Retain reserved actions whose exactly matched frozen filter state is true."""
    candidate = f"{action}__FILTERED_BY__{state_filter}"
    if candidate not in prereg.CANDIDATE_FAMILY:
        raise ValueError("HVSOF-8 candidate must use the exact frozen ordered family")
    actions = _validate_reserved_action_clock(action_clock.copy(), action)
    expected_filter_columns = list(FILTER_COLUMNS[state_filter])
    if filter_state.columns.tolist() != expected_filter_columns:
        raise RuntimeError(f"HVSOF-8 {state_filter} state schema drift")
    joined = actions.merge(
        filter_state,
        on="decision_time",
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError(f"HVSOF-8 {candidate} missing exact filter decision state")
    if not joined["decision_time"].le(joined["entry_time"]).all():
        raise RuntimeError(f"HVSOF-8 {candidate} filter state unavailable at entry")
    rank = FILTER_COLUMNS[state_filter][-1]
    if state_filter == "HVTCCR-8":
        keep = joined["source_valid"] & joined[rank].ge(0.80)
    else:
        keep = joined["source_valid"] & joined[rank].le(0.25)
    retained = joined.loc[keep, list(ACTION_COLUMNS)].copy()
    output = pd.DataFrame(
        {
            "candidate": candidate,
            "control": retained["control"],
            "split": retained["split"],
            "action_id": action,
            "filter_id": state_filter,
            "decision_time": retained["decision_time"],
            "feature_available_time": retained["feature_available_time"],
            "entry_time": retained["entry_time"],
            "exit_time": retained["exit_time"],
            "side": retained["side"],
        },
        columns=OUTPUT_COLUMNS,
    ).reset_index(drop=True)
    # Filtering an already-reserved action clock must preserve, not recompute, reservation.
    if len(output) > 1 and not output["entry_time"].iloc[1:].reset_index(drop=True).ge(
        output["exit_time"].iloc[:-1].reset_index(drop=True)
    ).all():
        raise RuntimeError(f"HVSOF-8 {candidate} retained action reservation overlap")
    return output


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock["split"].eq(split)]
    if selected.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
    count = len(selected)
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    return {
        "events": count,
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / count,
        "max_month_share": int(
            selected["entry_time"].dt.strftime("%Y-%m").value_counts().max()
        )
        / count,
    }


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
        state_filter: load_filter_state(state_filter)
        for state_filter in prereg.FILTER_ORDER
    }
    clock_dir.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, Any] = {}
    eligible: list[str] = []
    for action in prereg.ACTION_ORDER:
        for state_filter in prereg.FILTER_ORDER:
            candidate = f"{action}__FILTERED_BY__{state_filter}"
            clock = filter_action_clock(action, state_filter, actions[action], states[state_filter])
            path = clock_dir / f"{candidate}.csv.gz"
            _write_gzip_csv(clock, path)
            stats = {
                split: support_stats(clock, split) for split in prereg.build()["stages"]
            }
            checks = _support_checks(stats)
            passed = all(checks.values())
            if passed:
                eligible.append(candidate)
            candidates[candidate] = {
                "action": action,
                "filter": state_filter,
                "clock": {"path": str(path), "sha256": sha256_file(path), "rows": len(clock)},
                "support": stats,
                "support_checks": checks,
                "support_passed": passed,
                "advance_to_combination_gross9": passed,
                "advance_to_economic_outcomes": False,
                "decision": "pass_to_combination_gross9" if passed else "terminal_source_support_reject",
            }

    registration = _read_json_object(prereg.DEFAULT_OUTPUT)
    core = {
        "protocol_version": "hvsof_8_source_support_v1",
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
        "filter_incidence_opened": True,
        "filter_postentry_returns_or_pnl_opened": False,
        "entry_exit_prices_opened": False,
        "returns_opened": False,
        "funding_opened": False,
        "pnl_opened": False,
        "gross9_comparator_rows_opened": False,
        "additional_reservation_applied": False,
        "candidates": candidates,
        "eligible_candidates_for_combination_gross9": eligible,
        "advance_to_combination_gross9": bool(eligible),
        "advance_to_economic_outcomes": False,
        "decision": "eligible_candidates_to_combination_gross9" if eligible else "terminal_no_source_supported_candidates",
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
            {"eligible_candidates": report["eligible_candidates_for_combination_gross9"]},
            indent=2,
        )
    )
