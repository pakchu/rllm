"""Materialize train-only Gross9 asynchronous active-opposite-veto clocks.

Outcome-blind source-support builder for G9ASYNCACTIVEVETO-8.  It authenticates
hash-bound Gross9 component clocks and prior Gross9-like clock families, opens
only train-prefix schedule fields, writes all 72 ordered active-veto clocks, and
applies source/duplicate gates without market, funding, price, return, or PnL
access.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import permutations
from pathlib import Path
from typing import Any

import pandas as pd

from training import build_gross9_async_opposition_handoff_train_clocks as handoff
from training import build_gross9_async_pair_train_clocks as same_side
from training import build_gross9_async_three_way_consensus_train_clocks as three_way
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

POLICY_ID = "G9ASYNCACTIVEVETO-8"
PROTOCOL_VERSION = "gross9_async_active_opposite_veto_train_clocks_v1"
AS_OF_DATE = "2026-09-02"
TRAIN_START = same_side.TRAIN_START
TRAIN_END = same_side.TRAIN_END
CONFIRMATION_WINDOW = same_side.CONFIRMATION_WINDOW
HOLD = same_side.HOLD
COMPONENT_ORDER = same_side.COMPONENT_ORDER
COMPONENT_ARTIFACTS = same_side.COMPONENT_ARTIFACTS
COMMON_CLOCK_FIELDS = same_side.COMMON_CLOCK_FIELDS
CLOCK_DIR = Path("data/gross9_async_active_veto_train_clocks_2026-09-02")
BASE_CONTROL_DIR = Path("data/gross9_async_active_veto_train_base_controls_2026-09-02")
RESULT = Path("results/gross9_async_active_veto_train_clock_source_support_2026-09-02.json")
PREREG_MODULE = "training.preregister_gross9_async_active_veto_search"
SOURCE_GATES = {
    "minimum_events": 10,
    "minority_side_share_min": 0.20,
    "max_month_share": 0.45,
    "distinct_iso_weeks_min": 10,
    "minimum_events_each_train_half": 1,
    "minimum_opposite_suppressions": 1,
}
OUTPUT_COLUMNS = (
    "candidate",
    "control",
    "split",
    "base_component_id",
    "veto_component_id",
    "base_entry_time",
    "veto_entry_time",
    "veto_side",
    "veto_relation",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
)
BASE_CONTROL_COLUMNS = (
    "candidate",
    "control",
    "split",
    "base_component_id",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
)
PRIOR_SOURCE_SUPPORT_ARTIFACTS = {
    "same_side": {
        "policy_id": same_side.POLICY_ID,
        "path": "results/gross9_async_pair_train_clock_source_support_2026-09-02.json",
        "sha256": "c6d3929f282ba1075c2ebc091e4bc62164b923a038bce94de32884aaf7ff0009",
        "manifest_hash": "b92d3afb7a3539cdd194eddc1ab09bc65068716135d0bca575db0531ac450011",
        "family_size": 36,
        "members_key": "pairs",
        "expected_family": same_side.candidate_family,
        "schedule_scope": "same-side36 source-support artifact and schedules bound for exact-duplicate gates and overlap disclosure",
    },
    "handoff": {
        "policy_id": handoff.POLICY_ID,
        "path": "results/gross9_async_opposition_handoff_train_clock_source_support_2026-09-02.json",
        "sha256": "a8982c1b6e155f65f76af4559ca2d01b2a7824cb5c58524a260b72beb997f754",
        "manifest_hash": "92501aa4c921bba20d05378b6f658f33d6c712e8b3adb9f095940dd44ac3f3b0",
        "family_size": 36,
        "members_key": "pairs",
        "expected_family": handoff.candidate_family,
        "schedule_scope": "handoff36 source-support artifact and schedules bound for exact-duplicate gates and overlap disclosure",
    },
    "three_way": {
        "policy_id": three_way.POLICY_ID,
        "path": "results/gross9_async_three_way_consensus_train_clock_source_support_2026-09-02.json",
        "sha256": "0b9c9366d0d0d214787e1fdb6f3fad9e604e2dbfad49fedd8f4b84fadbcb5265",
        "manifest_hash": "de32635d2e1853359cfe62ca4ef779442fec0dd29caf680433693f07ce6b6495",
        "family_size": 84,
        "members_key": "triples",
        "expected_family": three_way.candidate_family,
        "schedule_scope": "triple84 source-support artifact and schedules bound for exact-duplicate gates and overlap disclosure",
    },
}

PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT = {
    "commit": "1bfddd3c",
    "path": "results/gross9_async_active_veto_train_clock_source_support_2026-09-02.json",
    "sha256": "ce95d6373655ded0daba9d6f5635908106337827fbf1a98c978cf41d8231e6e3",
    "manifest_hash": "88ce540e6ce329e0d9f763c128b2f431949c191772d207b6a1b6b65ee4fb3e6d",
    "builder": {
        "path": "training/build_gross9_async_active_veto_train_clocks.py",
        "sha256": "bf8bfaf41d0ca761a2bc0f2db53de5ad05103fe596b880eb0fd8acbbbc6c90df",
    },
    "placeholder_preregistration_sha256": "b70dbeea6a6d1bde63ea60c854fcfa09688060bf56c0f5a08f3f21073a5f4cba",
    "placeholder_preregistration_manifest_hash": "8cc95042fe2e76c5193f3679f6d1f073e2e97bd0a69b658c57599b9fba06ba28",
    "placeholder_builder_value": "PENDING_G9ASYNCACTIVEVETO_BUILDER_FOLLOWUP",
    "passed_candidates": 14,
}


def sha256_file(path: str | Path) -> str:
    return same_side.sha256_file(path)


def canonical_hash(value: Any) -> str:
    return same_side.canonical_hash(value)


def candidate_id(base: str, veto: str) -> str:
    return f"{base}__ASYNC_ACTIVE_OPPOSITE_VETO_6H__{veto}"


def base_control_id(base: str) -> str:
    return f"{base}__NO_VETO_8H_CONTROL"


def candidate_family(component_order: Sequence[str] = COMPONENT_ORDER) -> tuple[str, ...]:
    family = tuple(candidate_id(base, veto) for base, veto in permutations(component_order, 2))
    if len(family) != 72 or len(set(family)) != 72:
        raise RuntimeError(f"{POLICY_ID} requires exactly 72 ordered candidates")
    return family


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} expected JSON object: {path}")
    return value


def _validate_manifest_hash(value: Mapping[str, Any], *, label: str) -> None:
    core = dict(value)
    manifest_hash = core.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} {label} manifest hash drift")


def _expected_prior_source_support_bindings() -> list[dict[str, str]]:
    return [
        {
            "policy_id": str(spec["policy_id"]),
            "path": str(spec["path"]),
            "sha256": str(spec["sha256"]),
            "manifest_hash": str(spec["manifest_hash"]),
            "schedule_scope": str(spec["schedule_scope"]),
        }
        for spec in PRIOR_SOURCE_SUPPORT_ARTIFACTS.values()
    ]


def _validate_preregistration_contract(registration: Mapping[str, Any]) -> None:
    if registration.get("policy_id") != POLICY_ID:
        raise RuntimeError(f"{POLICY_ID} preregistration policy drift")
    if tuple(registration.get("component_order", ())) != COMPONENT_ORDER:
        raise RuntimeError(f"{POLICY_ID} component order drift")
    if tuple(registration.get("candidate_family", ())) != candidate_family():
        raise RuntimeError(f"{POLICY_ID} candidate family drift")
    if registration.get("candidate_family_size") != 72:
        raise RuntimeError(f"{POLICY_ID} family size drift")

    gates = registration.get("source_support_gates", {})
    if (
        gates.get("minimum_events", {}).get("train") != SOURCE_GATES["minimum_events"]
        or gates.get("minority_side_share_min") != SOURCE_GATES["minority_side_share_min"]
        or gates.get("max_month_share") != SOURCE_GATES["max_month_share"]
        or gates.get("distinct_iso_weeks_min") != SOURCE_GATES["distinct_iso_weeks_min"]
        or gates.get("each_calendar_half_min_events") != SOURCE_GATES["minimum_events_each_train_half"]
        or gates.get("opposite_suppressions_min") != SOURCE_GATES["minimum_opposite_suppressions"]
    ):
        raise RuntimeError(f"{POLICY_ID} preregistration source gate drift")

    implementation = registration.get("implementation", {}).get("train_clock_builder", {})
    bound_sha = implementation.get("sha256")
    if not isinstance(bound_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", bound_sha):
        raise RuntimeError(f"{POLICY_ID} preregistration builder hash is missing or placeholder")
    if implementation.get("status") is not None:
        raise RuntimeError(f"{POLICY_ID} preregistration builder status placeholder must be removed")
    observed_builder_sha = sha256_file(__file__)
    if bound_sha != observed_builder_sha:
        raise RuntimeError(f"{POLICY_ID} preregistration builder hash mismatch")

    if registration.get("prior_source_support_artifacts") != _expected_prior_source_support_bindings():
        raise RuntimeError(f"{POLICY_ID} preregistration prior source-support binding drift")

    receipt = registration.get("preliminary_source_materialization_receipt", {})
    if not isinstance(receipt, Mapping):
        raise RuntimeError(f"{POLICY_ID} preregistration preliminary materialization receipt missing")
    expected_receipt_scalars = {
        "commit": PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["commit"],
        "path": PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["path"],
        "sha256": PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["sha256"],
        "manifest_hash": PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["manifest_hash"],
        "placeholder_builder_value": PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["placeholder_builder_value"],
    }
    for key, expected in expected_receipt_scalars.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"{POLICY_ID} preregistration preliminary materialization receipt drift: {key}")
    if receipt.get("builder") != PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["builder"]:
        raise RuntimeError(f"{POLICY_ID} preregistration preliminary builder receipt drift")
    placeholder = receipt.get("preregistration_artifact_with_placeholder_builder_binding", {})
    if not isinstance(placeholder, Mapping):
        raise RuntimeError(f"{POLICY_ID} preregistration placeholder artifact receipt missing")
    if (
        placeholder.get("sha256") != PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["placeholder_preregistration_sha256"]
        or placeholder.get("manifest_hash") != PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["placeholder_preregistration_manifest_hash"]
    ):
        raise RuntimeError(f"{POLICY_ID} preregistration placeholder artifact hash receipt drift")
    support = receipt.get("support_count_disclosure", {})
    if not isinstance(support, Mapping) or support.get("passed_candidates") != PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["passed_candidates"] or support.get("used_to_retune_family_operator_gates_thresholds_or_order") is not False:
        raise RuntimeError(f"{POLICY_ID} preregistration preliminary support-count disclosure drift")

    boundary = registration.get("research_boundary", {})
    required = {
        "design_family_operator_and_gates_fixed_before_preliminary_source_materialization": True,
        "source_incidence_and_support_counts_opened_before_committed_preregistration": True,
        "family_operator_gate_threshold_or_order_changed_after_preliminary_source_materialization": False,
        "preliminary_14_source_passes_used_to_retune": False,
        "gross9_market_funding_or_pnl_opened_by_preregistration": False,
        "active_veto_combination_outcomes_opened_by_preregistration": False,
        "market_or_funding_rows_opened_by_preregistration": False,
    }
    for key, expected in required.items():
        if boundary.get(key) is not expected:
            raise RuntimeError(f"{POLICY_ID} preregistration research-boundary disclosure drift: {key}")


def load_validated_preregistration() -> dict[str, Any]:
    """Hard-validate the committed preregistration before source materialization."""
    try:
        prereg = importlib.import_module(PREREG_MODULE)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"{POLICY_ID} missing preregistration module: {PREREG_MODULE}") from exc
    default_output = Path(getattr(prereg, "DEFAULT_OUTPUT"))
    if not default_output.is_file():
        raise RuntimeError(f"{POLICY_ID} missing committed preregistration artifact: {default_output}")
    registration = _read_json_object(default_output)
    prereg.validate(registration)
    if registration != prereg.build():
        raise RuntimeError(f"{POLICY_ID} preregistration artifact differs from code")
    _validate_preregistration_contract(registration)
    return {
        "available": True,
        "module": PREREG_MODULE,
        "path": str(default_output),
        "sha256": sha256_file(default_output),
        "manifest_hash": registration["manifest_hash"],
        "status": "validated_against_committed_preregistration",
        "prior_source_support_artifacts_cross_checked": True,
        "research_boundary_disclosure_cross_checked": True,
    }


def verify_bound_component_artifacts(
    component_artifacts: Mapping[str, Mapping[str, Mapping[str, str]]] = COMPONENT_ARTIFACTS,
    component_order: Sequence[str] = COMPONENT_ORDER,
) -> dict[str, Any]:
    return same_side.verify_bound_component_artifacts(component_artifacts, component_order)


def load_train_prefix_clock(
    component: str,
    component_artifacts: Mapping[str, Mapping[str, Mapping[str, str]]] = COMPONENT_ARTIFACTS,
) -> pd.DataFrame:
    return same_side.load_train_prefix_clock(component, component_artifacts)


def reserve_half_open(clock: pd.DataFrame) -> pd.DataFrame:
    ordered = clock.sort_values(["entry_time", "candidate", "side"], kind="stable")
    keep: list[int] = []
    next_available: pd.Timestamp | None = None
    for index, row in ordered.iterrows():
        if next_available is not None and row["entry_time"] < next_available:
            continue
        keep.append(index)
        next_available = row["exit_time"]
    return ordered.loc[keep].reset_index(drop=True)


def build_normalized_base_control(base: str, base_clock: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in base_clock.sort_values("entry_time", kind="stable").itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        rows.append(
            {
                "candidate": base_control_id(base),
                "control": "normalized_no_veto_8h_base",
                "split": "train",
                "base_component_id": base,
                "decision_time": pd.Timestamp(row.decision_time),
                "feature_available_time": pd.Timestamp(row.feature_available_time),
                "entry_time": entry,
                "exit_time": entry + HOLD,
                "side": int(row.side),
            }
        )
    control = pd.DataFrame(rows, columns=BASE_CONTROL_COLUMNS)
    return reserve_half_open(control) if not control.empty else control


def _latest_veto(veto_clock: pd.DataFrame, entry_time: pd.Timestamp) -> pd.Series | None:
    window = veto_clock.loc[
        veto_clock["entry_time"].gt(entry_time - CONFIRMATION_WINDOW)
        & veto_clock["entry_time"].le(entry_time)
    ].sort_values("entry_time", kind="stable")
    if window.empty:
        return None
    return window.iloc[-1]


def build_active_veto_pre_clock(base: str, veto: str, base_clock: pd.DataFrame, veto_clock: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    for component, clock in ((base, base_clock), (veto, veto_clock)):
        if clock.empty or not clock["candidate"].eq(component).all():
            raise RuntimeError(f"{POLICY_ID} invalid component clock for {component}")
    rows: list[dict[str, Any]] = []
    diagnostics = {
        "base_events_seen": int(len(base_clock)),
        "no_veto_window_keeps": 0,
        "same_side_latest_veto_keeps": 0,
        "opposite_latest_veto_suppressions": 0,
        "duplicate_rows_dropped": 0,
    }
    for event in base_clock.sort_values("entry_time", kind="stable").itertuples(index=False):
        entry = pd.Timestamp(event.entry_time)
        side = int(event.side)
        veto_row = _latest_veto(veto_clock, entry)
        veto_entry: pd.Timestamp | pd.NaT = pd.NaT
        veto_side: int | pd.NA = pd.NA
        relation = "none"
        decision_time = pd.Timestamp(event.decision_time)
        feature_available_time = pd.Timestamp(event.feature_available_time)
        if veto_row is None:
            diagnostics["no_veto_window_keeps"] += 1
        else:
            veto_entry = pd.Timestamp(veto_row["entry_time"])
            veto_side = int(veto_row["side"])
            decision_time = max(decision_time, pd.Timestamp(veto_row["decision_time"]))
            feature_available_time = max(feature_available_time, pd.Timestamp(veto_row["feature_available_time"]))
            if veto_side == side:
                relation = "same_side_latest_keep"
                diagnostics["same_side_latest_veto_keeps"] += 1
            else:
                diagnostics["opposite_latest_veto_suppressions"] += 1
                continue
        rows.append(
            {
                "candidate": candidate_id(base, veto),
                "control": "primary",
                "split": "train",
                "base_component_id": base,
                "veto_component_id": veto,
                "base_entry_time": entry,
                "veto_entry_time": veto_entry,
                "veto_side": veto_side,
                "veto_relation": relation,
                "decision_time": decision_time,
                "feature_available_time": feature_available_time,
                "entry_time": entry,
                "exit_time": entry + HOLD,
                "side": side,
            }
        )
    pre = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if not pre.empty:
        before = len(pre)
        pre = pre.drop_duplicates(["candidate", "entry_time", "side"], keep="first").reset_index(drop=True)
        diagnostics["duplicate_rows_dropped"] = before - len(pre)
    _validate_clock(pre, require_reservation=False)
    return pre, diagnostics


def build_active_veto_clock(base: str, veto: str, base_clock: pd.DataFrame, veto_clock: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    pre, diagnostics = build_active_veto_pre_clock(base, veto, base_clock, veto_clock)
    clock = reserve_half_open(pre) if not pre.empty else pre
    _validate_clock(clock)
    diagnostics = {**diagnostics, "pre_reservation_rows": len(pre), "post_reservation_rows": len(clock), "reservation_dropped_rows": len(pre) - len(clock)}
    return clock, diagnostics


def _validate_clock(clock: pd.DataFrame, *, require_reservation: bool = True) -> None:
    if clock.empty:
        return
    if not clock["entry_time"].ge(TRAIN_START).all() or not clock["entry_time"].lt(TRAIN_END).all() or not clock["exit_time"].le(TRAIN_END).all():
        raise RuntimeError(f"{POLICY_ID} clock escaped train prefix")
    if not clock["decision_time"].le(clock["entry_time"]).all():
        raise RuntimeError(f"{POLICY_ID} decision after entry")
    if not clock["feature_available_time"].le(clock["entry_time"]).all():
        raise RuntimeError(f"{POLICY_ID} feature after entry")
    if not clock["side"].isin((-1, 1)).all():
        raise RuntimeError(f"{POLICY_ID} non-strict side")
    if "veto_entry_time" in clock.columns:
        used = clock.loc[clock["veto_entry_time"].notna()]
        if not used.empty:
            lag = used["entry_time"] - used["veto_entry_time"]
            if not (lag.ge(pd.Timedelta(0)).all() and lag.lt(CONFIRMATION_WINDOW).all()):
                raise RuntimeError(f"{POLICY_ID} veto lag drift")
    if require_reservation and len(clock) > 1 and not clock["entry_time"].iloc[1:].reset_index(drop=True).ge(
        clock["exit_time"].iloc[:-1].reset_index(drop=True)
    ).all():
        raise RuntimeError(f"{POLICY_ID} reservation overlap")


def support_stats(clock: pd.DataFrame, *, opposite_suppressions: int) -> dict[str, float | int]:
    if clock.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0, "distinct_iso_weeks": 0, "first_half_events": 0, "second_half_events": 0, "opposite_suppressions": opposite_suppressions}
    longs = int(clock["side"].eq(1).sum())
    shorts = int(clock["side"].eq(-1).sum())
    events = len(clock)
    month_counts = clock["entry_time"].dt.strftime("%Y-%m").value_counts()
    iso = clock["entry_time"].dt.isocalendar()
    weeks = set(zip(iso["year"].astype(int), iso["week"].astype(int), strict=True))
    midpoint = TRAIN_START + (TRAIN_END - TRAIN_START) / 2
    first_half = int(clock["entry_time"].lt(midpoint).sum())
    return {"events": events, "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / events, "max_month_share": int(month_counts.max()) / events, "distinct_iso_weeks": len(weeks), "first_half_events": first_half, "second_half_events": events - first_half, "opposite_suppressions": opposite_suppressions}


def support_checks(stats: Mapping[str, float | int]) -> dict[str, bool]:
    return {
        "minimum_events": stats["events"] >= SOURCE_GATES["minimum_events"],
        "side_balance": stats["minority_side_share"] >= SOURCE_GATES["minority_side_share_min"],
        "month_concentration": stats["max_month_share"] <= SOURCE_GATES["max_month_share"],
        "distinct_iso_weeks": stats["distinct_iso_weeks"] >= SOURCE_GATES["distinct_iso_weeks_min"],
        "both_train_halves": stats["first_half_events"] >= SOURCE_GATES["minimum_events_each_train_half"] and stats["second_half_events"] >= SOURCE_GATES["minimum_events_each_train_half"],
        "opposite_suppression_presence": stats["opposite_suppressions"] >= SOURCE_GATES["minimum_opposite_suppressions"],
    }


def _clock_key_set(clock: pd.DataFrame) -> set[tuple[pd.Timestamp, int]]:
    if clock.empty:
        return set()
    return set(zip(pd.to_datetime(clock["entry_time"], utc=True), clock["side"].astype(int), strict=True))


def _schedule_signature(clock: pd.DataFrame) -> tuple[tuple[str, str, int], ...]:
    if clock.empty:
        return tuple()
    ordered = clock.sort_values(["entry_time", "exit_time", "side"], kind="stable")
    return tuple((pd.Timestamp(row.entry_time).isoformat(), pd.Timestamp(row.exit_time).isoformat(), int(row.side)) for row in ordered.itertuples(index=False))


def _matched_share(candidate_keys: set[tuple[pd.Timestamp, int]], comparator_keys: set[tuple[pd.Timestamp, int]]) -> float:
    if not candidate_keys:
        return 0.0
    remaining = sorted(comparator_keys)
    matched = 0
    for entry_time, side in sorted(candidate_keys):
        for index, (other_time, other_side) in enumerate(remaining):
            if side == other_side and abs(entry_time - other_time) <= CONFIRMATION_WINDOW:
                matched += 1
                remaining.pop(index)
                break
    return matched / len(candidate_keys)


def _overlap(candidate_keys: set[tuple[pd.Timestamp, int]], comparator_keys: set[tuple[pd.Timestamp, int]]) -> dict[str, float | int]:
    intersection = candidate_keys & comparator_keys
    union = candidate_keys | comparator_keys
    return {
        "candidate_keys": len(candidate_keys),
        "comparator_keys": len(comparator_keys),
        "exact_entry_side_intersection": len(intersection),
        "exact_entry_side_jaccard": len(intersection) / len(union) if union else 1.0,
        "same_side_matched_share_plus_minus_6h": _matched_share(candidate_keys, comparator_keys),
    }


def _load_prior_clock(path: Path, *, expected_sha256: str, expected_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"{POLICY_ID} missing prior clock schedule: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"{POLICY_ID} prior clock schedule SHA drift: {path}")
    frame = pd.read_csv(path, usecols=["entry_time", "exit_time", "side"])
    if frame.empty:
        if expected_rows != 0:
            raise RuntimeError(f"{POLICY_ID} prior clock row-count drift: {path}")
        return pd.DataFrame(columns=["entry_time", "exit_time", "side"])
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    frame["side"] = frame["side"].astype(int)
    if len(frame) != expected_rows:
        raise RuntimeError(f"{POLICY_ID} prior clock row-count drift: {path}")
    if not frame["entry_time"].ge(TRAIN_START).all() or not frame["entry_time"].lt(TRAIN_END).all() or not frame["exit_time"].le(TRAIN_END).all():
        raise RuntimeError(f"{POLICY_ID} prior clock escaped train containment: {path}")
    return frame.sort_values(["entry_time", "side"], kind="stable").reset_index(drop=True)


def load_validated_prior_source_artifacts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, spec in PRIOR_SOURCE_SUPPORT_ARTIFACTS.items():
        path = Path(str(spec["path"]))
        observed_sha = sha256_file(path)
        if observed_sha != spec["sha256"]:
            raise RuntimeError(f"{POLICY_ID} {label} prior source-support SHA drift")
        value = _read_json_object(path)
        _validate_manifest_hash(value, label=label)
        if value.get("policy_id") != spec["policy_id"] or value.get("manifest_hash") != spec["manifest_hash"]:
            raise RuntimeError(f"{POLICY_ID} {label} prior source-support identity drift")
        if tuple(value.get("candidate_family", ())) != tuple(spec["expected_family"]()):
            raise RuntimeError(f"{POLICY_ID} {label} prior candidate-family drift")
        members = value.get(str(spec["members_key"]), {})
        if not isinstance(members, dict) or len(members) != spec["family_size"]:
            raise RuntimeError(f"{POLICY_ID} {label} prior member-count drift")
        out[label] = {"path": str(path), "sha256": observed_sha, "manifest_hash": value["manifest_hash"], "policy_id": value["policy_id"], "family_size": spec["family_size"], "members_key": spec["members_key"], "members": members}
    return out


def load_prior_clock_schedules(prior_artifacts: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    if prior_artifacts is None:
        prior_artifacts = load_validated_prior_source_artifacts()
    schedules: dict[str, dict[str, Any]] = {}
    for label, artifact in prior_artifacts.items():
        for member_id, member in artifact["members"].items():
            clock_binding = member.get("clock", {})
            path = Path(clock_binding.get("path", ""))
            clock = _load_prior_clock(path, expected_sha256=str(clock_binding.get("sha256", "")), expected_rows=int(clock_binding.get("rows", -1)))
            schedules[member_id] = {"family": label, "path": str(path), "sha256": clock_binding.get("sha256"), "rows": len(clock), "keys": _clock_key_set(clock), "signature": _schedule_signature(clock)}
    return schedules


def duplicate_groups(signatures: Mapping[str, tuple[tuple[str, str, int], ...]]) -> dict[tuple[tuple[str, str, int], ...], list[str]]:
    groups: dict[tuple[tuple[str, str, int], ...], list[str]] = defaultdict(list)
    for name, signature in signatures.items():
        if signature:
            groups[signature].append(name)
    return {sig: names for sig, names in groups.items() if len(names) > 1}


def duplicate_gate_report(
    candidates: Mapping[str, pd.DataFrame],
    base_controls: Mapping[str, pd.DataFrame],
    prior_schedules: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_sigs = {name: _schedule_signature(clock) for name, clock in candidates.items()}
    base_sigs = {name: _schedule_signature(clock) for name, clock in base_controls.items()}
    prior_sigs = {name: prior["signature"] for name, prior in prior_schedules.items()}
    current_groups = duplicate_groups(candidate_sigs)
    rejects: dict[str, list[str]] = {name: [] for name in candidates}
    base_duplicate_of: dict[str, list[str]] = defaultdict(list)
    prior_duplicate_of: dict[str, list[str]] = defaultdict(list)
    for candidate, sig in candidate_sigs.items():
        if not sig:
            continue
        for base_name, base_sig in base_sigs.items():
            if base_sig and sig == base_sig:
                base_duplicate_of[candidate].append(base_name)
                rejects[candidate].append(f"base:{base_name}")
        for prior_name, prior_sig in prior_sigs.items():
            if prior_sig and sig == prior_sig:
                prior_duplicate_of[candidate].append(prior_name)
                rejects[candidate].append(f"prior:{prior_name}")
    for names in current_groups.values():
        for candidate in names:
            rejects[candidate].extend(f"current:{other}" for other in names if other != candidate)
    return {
        "candidate_reject_reasons": {k: v for k, v in rejects.items() if v},
        "base_control_duplicates": dict(base_duplicate_of),
        "current_candidate_duplicate_groups": [names for names in current_groups.values()],
        "prior_schedule_duplicates": dict(prior_duplicate_of),
        "rejected_candidates": sorted(k for k, v in rejects.items() if v),
    }


def overlap_disclosure_for_candidate(
    candidate_clock: pd.DataFrame,
    base_control: pd.DataFrame,
    candidates: Mapping[str, pd.DataFrame],
    prior_schedules: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    keys = _clock_key_set(candidate_clock)
    current_union: set[tuple[pd.Timestamp, int]] = set()
    current_nonzero: dict[str, Any] = {}
    for name, clock in candidates.items():
        if clock is candidate_clock:
            continue
        other = _clock_key_set(clock)
        current_union |= other
        metrics = _overlap(keys, other)
        if metrics["exact_entry_side_intersection"] or metrics["same_side_matched_share_plus_minus_6h"]:
            current_nonzero[name] = metrics
    prior_unions: dict[str, set[tuple[pd.Timestamp, int]]] = defaultdict(set)
    prior_nonzero: dict[str, Any] = {}
    for name, prior in prior_schedules.items():
        prior_keys = set(prior["keys"])
        prior_unions[str(prior["family"])].update(prior_keys)
        metrics = _overlap(keys, prior_keys)
        if metrics["exact_entry_side_intersection"] or metrics["same_side_matched_share_plus_minus_6h"]:
            prior_nonzero[name] = {"family": prior["family"], **metrics}
    return {
        "base_control": _overlap(keys, _clock_key_set(base_control)),
        "current_family_union": _overlap(keys, current_union),
        "current_family_nonzero_overlaps": current_nonzero,
        "prior_family_unions": {family: _overlap(keys, union) for family, union in prior_unions.items()},
        "prior_family_nonzero_overlaps": prior_nonzero,
    }


def run(clock_dir: Path = CLOCK_DIR, result_path: Path = RESULT, base_control_dir: Path = BASE_CONTROL_DIR) -> dict[str, Any]:
    preregistration = load_validated_preregistration()
    verified = verify_bound_component_artifacts()
    clocks = {component: load_train_prefix_clock(component) for component in COMPONENT_ORDER}
    prior_artifacts = load_validated_prior_source_artifacts()
    prior_schedules = load_prior_clock_schedules(prior_artifacts)
    clock_dir.mkdir(parents=True, exist_ok=True)
    base_control_dir.mkdir(parents=True, exist_ok=True)

    base_controls: dict[str, pd.DataFrame] = {}
    base_control_records: dict[str, Any] = {}
    for component in COMPONENT_ORDER:
        control = build_normalized_base_control(component, clocks[component])
        path = base_control_dir / f"{base_control_id(component)}.csv.gz"
        _write_gzip_csv(control, path)
        name = base_control_id(component)
        base_controls[name] = control
        base_control_records[name] = {"base_component_id": component, "clock": {"path": str(path), "sha256": sha256_file(path), "rows": len(control)}}

    built: dict[str, pd.DataFrame] = {}
    diagnostics_by_candidate: dict[str, dict[str, int]] = {}
    for base, veto in permutations(COMPONENT_ORDER, 2):
        name = candidate_id(base, veto)
        clock, diagnostics = build_active_veto_clock(base, veto, clocks[base], clocks[veto])
        built[name] = clock
        diagnostics_by_candidate[name] = diagnostics

    duplicate_report = duplicate_gate_report(built, base_controls, prior_schedules)
    candidates: dict[str, Any] = {}
    passed: list[str] = []
    empty_source_fails: list[str] = []
    duplicate_rejects = set(duplicate_report["rejected_candidates"])
    for base, veto in permutations(COMPONENT_ORDER, 2):
        name = candidate_id(base, veto)
        clock = built[name]
        path = clock_dir / f"{name}.csv.gz"
        _write_gzip_csv(clock, path)
        diagnostics = diagnostics_by_candidate[name]
        stats = support_stats(clock, opposite_suppressions=diagnostics["opposite_latest_veto_suppressions"])
        checks = support_checks(stats)
        duplicate_reject = name in duplicate_rejects
        source_pass = all(checks.values()) and not duplicate_reject
        if clock.empty:
            empty_source_fails.append(name)
        if source_pass:
            passed.append(name)
        candidates[name] = {
            "components": {"base": base, "veto": veto},
            "operator": "ordered_async_active_opposite_veto_latest_veto_in_strict_lower_inclusive_upper_6h",
            "clock": {"path": str(path), "sha256": sha256_file(path), "rows": len(clock)},
            "construction_diagnostics": diagnostics,
            "support": stats,
            "support_checks": checks,
            "duplicate_gate": {"rejected": duplicate_reject, "reasons": duplicate_report["candidate_reject_reasons"].get(name, [])},
            "overlap_disclosure": overlap_disclosure_for_candidate(clock, base_controls[base_control_id(base)], built, prior_schedules),
            "support_passed": source_pass,
            "advance_to_gross9_novelty": source_pass,
            "advance_to_economic_outcomes": False,
            "decision": "pass_to_gross9_novelty" if source_pass else "terminal_source_support_reject",
        }

    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "component_order": list(COMPONENT_ORDER),
        "candidate_family": list(candidate_family()),
        "candidate_family_size": 72,
        "preregistration": preregistration,
        "implementation": {"builder": {"path": "training/build_gross9_async_active_veto_train_clocks.py", "sha256": sha256_file(__file__)}},
        "verified_component_artifacts": verified,
        "prior_source_support_artifacts": {family: {key: value for key, value in artifact.items() if key != "members"} for family, artifact in prior_artifacts.items()},
        "prior_clock_schedules_opened": {"same_side_family_size": 36, "handoff_family_size": 36, "three_way_family_size": 84, "total_prior_schedules": len(prior_schedules), "authenticated_via_prior_source_support_artifacts": True, "schedule_fields_opened": ["entry_time", "exit_time", "side"]},
        "train_prefix": {"start_inclusive": TRAIN_START.isoformat().replace("+00:00", "Z"), "end_exclusive": TRAIN_END.isoformat().replace("+00:00", "Z"), "component_rows_loaded": {component: len(clock) for component, clock in clocks.items()}},
        "construction": {"operator": "for each ordered pair A(base),B(veto), evaluate each base event t,s; use latest B event satisfying t-6h < v.entry <= t; no veto or same-side latest veto keeps base side; opposite latest veto suppresses to cash and never reverses", "same_time_veto_allowed": True, "strict_lower_window_boundary": "t-6h is excluded", "inclusive_upper_window_boundary": "t is included", "latest_veto_supersedes_older": True, "reservation": "candidate-local chronological half-open 8h reservation after active-veto materialization", "base_controls": "normalized no-veto 8h half-open reserved controls built for each base component", "duplicate_policy": "nonempty exact full schedule duplicates are rejected vs normalized bases, vs other current candidates with all duplicate group members rejected, and vs prior same-side/handoff/three-way schedules; empty schedules fail only source gates"},
        "source_support_gates": SOURCE_GATES,
        "base_controls": base_control_records,
        "duplicate_gate_summary": duplicate_report,
        "candidates": candidates,
        "passed_candidates": passed,
        "empty_source_failures": empty_source_fails,
        "support_passed_any_candidate": bool(passed),
        "preliminary_source_materialization_receipt": PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT,
        "research_boundary": {
            "design_family_operator_and_gates_fixed_before_preliminary_source_materialization": True,
            "source_incidence_and_support_counts_opened_before_committed_preregistration": True,
            "family_operator_gate_threshold_or_order_changed_after_preliminary_source_materialization": False,
            "preliminary_14_source_passes_used_to_retune": False,
            "gross9_market_funding_or_pnl_opened_by_preregistration": False,
            "active_veto_combination_outcomes_opened_by_preregistration": False,
            "market_or_funding_rows_opened_by_preregistration": False,
            "returns_or_pnl_opened": False,
            "economic_outcomes_opened": False,
        },
        "evidence_boundary": {"component_clock_fields_opened": list(COMMON_CLOCK_FIELDS), "component_clock_rows_materialized_train_prefix_only": True, "prior_clock_schedule_fields_opened": ["entry_time", "exit_time", "side"], "prior_clock_schedules_opened_for_authentication_duplicate_gates_and_disclosure_only": True, "preliminary_source_materialization_commit": "1bfddd3c", "source_incidence_and_support_counts_opened_before_committed_preregistration": True, "family_operator_gate_threshold_or_order_changed_after_preliminary_source_materialization": False, "preliminary_14_source_passes_used_to_retune": False, "gross9_rows_opened": False, "market_rows_opened": False, "entry_exit_prices_opened": False, "funding_opened": False, "returns_or_pnl_opened": False, "economic_outcomes_opened": False, "base_control_economic_outcomes_opened": False, "oos_component_rows_materialized": 0},
        "decision": "pass_supported_active_veto_candidates_to_gross9_novelty" if passed else "terminal_no_source_supported_active_veto_candidates",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clock-dir", type=Path, default=CLOCK_DIR)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--base-control-dir", type=Path, default=BASE_CONTROL_DIR)
    args = parser.parse_args()
    report = run(args.clock_dir, args.result, args.base_control_dir)
    print(json.dumps({"passed_candidates": report["passed_candidates"], "candidate_family_size": report["candidate_family_size"], "decision": report["decision"]}, indent=2))
