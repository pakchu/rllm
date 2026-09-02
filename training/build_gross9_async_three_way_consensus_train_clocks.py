"""Materialize train-only Gross9 asynchronous three-way consensus clocks.

This builder is outcome-blind.  It authenticates the frozen Gross9 component
clock sources, opens only primary clock timing/side fields, keeps only the train
prefix, and writes one deterministic eight-hour hold clock for every unordered
three-component candidate.  It may open prior same-side/handoff clock schedules
for overlap disclosure and exact-duplicate rejection, but opens no market,
funding, return, price, PnL, or OOS outcome rows.
"""
from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from training import build_gross9_async_opposition_handoff_train_clocks as handoff
from training import build_gross9_async_pair_train_clocks as same_side
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


POLICY_ID = "G9ASYNC3WAY-8"
PROTOCOL_VERSION = "gross9_async_three_way_consensus_train_clocks_v1"
TRAIN_START = same_side.TRAIN_START
TRAIN_END = same_side.TRAIN_END
CONFIRMATION_WINDOW = same_side.CONFIRMATION_WINDOW
HOLD = same_side.HOLD
COMPONENT_ORDER = same_side.COMPONENT_ORDER
COMPONENT_ARTIFACTS = same_side.COMPONENT_ARTIFACTS
COMMON_CLOCK_FIELDS = same_side.COMMON_CLOCK_FIELDS
CLOCK_DIR = Path("data/gross9_async_three_way_consensus_train_clocks_2026-09-02")
RESULT = Path("results/gross9_async_three_way_consensus_train_clock_source_support_2026-09-02.json")
SAME_SIDE_CLOCK_DIR = same_side.CLOCK_DIR
HANDOFF_CLOCK_DIR = handoff.CLOCK_DIR
PREREG_MODULE = "training.preregister_gross9_async_three_way_consensus_search"
SOURCE_GATES = {
    "minimum_events": 10,
    "minority_side_share_min": 0.20,
    "max_month_share": 0.45,
    "distinct_iso_weeks_min": 10,
    "minimum_events_each_train_half": 1,
}
OUTPUT_COLUMNS = (
    "candidate",
    "control",
    "split",
    "left_component_id",
    "middle_component_id",
    "right_component_id",
    "trigger_component_id",
    "trigger_component_ids",
    "selected_component_ids",
    "left_selected_entry_time",
    "middle_selected_entry_time",
    "right_selected_entry_time",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
)


def sha256_file(path: str | Path) -> str:
    return same_side.sha256_file(path)


def canonical_hash(value: Any) -> str:
    return same_side.canonical_hash(value)


def triple_id(left: str, middle: str, right: str) -> str:
    return f"{left}__ASYNC_SAME_SIDE_3WAY_6H__{middle}__{right}"


def candidate_family(component_order: Sequence[str] = COMPONENT_ORDER) -> tuple[str, ...]:
    family = tuple(triple_id(*triple) for triple in combinations(component_order, 3))
    if tuple(component_order) == COMPONENT_ORDER:
        try:
            prereg = importlib.import_module(PREREG_MODULE)
            registered = getattr(prereg, "CANDIDATE_FAMILY", None)
        except ModuleNotFoundError:
            registered = None
        if registered is not None and family != tuple(registered):
            raise RuntimeError(f"{POLICY_ID} candidate family differs from preregistration")
    return family


def load_train_prefix_clock(
    component: str,
    component_artifacts: Mapping[str, Mapping[str, Mapping[str, str]]] = COMPONENT_ARTIFACTS,
) -> pd.DataFrame:
    return same_side.load_train_prefix_clock(component, component_artifacts)


def verify_bound_component_artifacts(
    component_artifacts: Mapping[str, Mapping[str, Mapping[str, str]]] = COMPONENT_ARTIFACTS,
    component_order: Sequence[str] = COMPONENT_ORDER,
) -> dict[str, Any]:
    return same_side.verify_bound_component_artifacts(component_artifacts, component_order)


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} expected JSON object: {path}")
    return value


def load_validated_preregistration() -> tuple[Any, Mapping[str, Any]]:
    prereg = importlib.import_module(PREREG_MODULE)
    default_output = getattr(prereg, "DEFAULT_OUTPUT")
    if not Path(default_output).is_file():
        raise RuntimeError(f"{POLICY_ID} missing committed preregistration artifact")
    registration = _read_json_object(default_output)
    prereg.validate(registration)
    if registration != prereg.build():
        raise RuntimeError(f"{POLICY_ID} preregistration artifact differs from code")
    if registration.get("policy_id") != POLICY_ID:
        raise RuntimeError(f"{POLICY_ID} preregistration policy drift")
    if tuple(registration.get("component_order", ())) != COMPONENT_ORDER:
        raise RuntimeError(f"{POLICY_ID} component order drift")
    if tuple(registration.get("candidate_family", ())) != candidate_family():
        raise RuntimeError(f"{POLICY_ID} candidate family drift")
    if registration.get("candidate_family_size") != 84:
        raise RuntimeError(f"{POLICY_ID} family size drift")
    return prereg, registration


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


def _latest_same_side(clock: pd.DataFrame, side: int, entry_time: pd.Timestamp) -> pd.Series | None:
    window = clock.loc[
        clock["side"].eq(side)
        & clock["entry_time"].le(entry_time)
        & clock["entry_time"].ge(entry_time - CONFIRMATION_WINDOW)
    ].sort_values("entry_time", kind="stable")
    if window.empty:
        return None
    return window.iloc[-1]


def build_async_three_way_consensus_pre_clock(
    left: str,
    middle: str,
    right: str,
    left_clock: pd.DataFrame,
    middle_clock: pd.DataFrame,
    right_clock: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build inclusive 6h same-side consensus before triple-local reservation."""
    clocks = {left: left_clock, middle: middle_clock, right: right_clock}
    triple = (left, middle, right)
    for component in triple:
        clock = clocks[component]
        if clock.empty or not clock["candidate"].eq(component).all():
            raise RuntimeError(f"{POLICY_ID} invalid component clock for {component}")

    event_times = sorted(set(pd.concat([clock["entry_time"] for clock in clocks.values()], ignore_index=True)))
    rows: list[dict[str, Any]] = []
    missing_component_rejections = 0
    no_exact_trigger_rejections = 0
    availability_rejections = 0
    for entry_time in event_times:
        entry_time = pd.Timestamp(entry_time)
        for side in (1, -1):
            selected: dict[str, pd.Series] = {}
            for component in triple:
                row = _latest_same_side(clocks[component], side, entry_time)
                if row is None:
                    missing_component_rejections += 1
                    selected = {}
                    break
                selected[component] = row
            if not selected:
                continue
            exact_components = [component for component in triple if pd.Timestamp(selected[component]["entry_time"]) == entry_time]
            if not exact_components:
                no_exact_trigger_rejections += 1
                continue
            trigger_component = exact_components[0]
            decision_time = max(pd.Timestamp(row["decision_time"]) for row in selected.values())
            feature_available_time = max(pd.Timestamp(row["feature_available_time"]) for row in selected.values())
            if decision_time > entry_time or feature_available_time > entry_time:
                availability_rejections += 1
                continue
            rows.append(
                {
                    "candidate": triple_id(left, middle, right),
                    "control": "primary",
                    "split": "train",
                    "left_component_id": left,
                    "middle_component_id": middle,
                    "right_component_id": right,
                    "trigger_component_id": trigger_component,
                    "trigger_component_ids": "|".join(exact_components),
                    "selected_component_ids": "|".join(triple),
                    "left_selected_entry_time": selected[left]["entry_time"],
                    "middle_selected_entry_time": selected[middle]["entry_time"],
                    "right_selected_entry_time": selected[right]["entry_time"],
                    "decision_time": decision_time,
                    "feature_available_time": feature_available_time,
                    "entry_time": entry_time,
                    "exit_time": entry_time + HOLD,
                    "side": side,
                }
            )

    pre = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    duplicate_rows_dropped = 0
    ambiguous_rows_dropped = 0
    if not pre.empty:
        before = len(pre)
        pre = pre.drop_duplicates(["candidate", "entry_time", "side"], keep="first").reset_index(drop=True)
        duplicate_rows_dropped = before - len(pre)
        ambiguous_times = set(
            pre.groupby("entry_time")["side"].nunique().loc[lambda series: series > 1].index
        )
        if ambiguous_times:
            ambiguous_rows_dropped = int(pre["entry_time"].isin(ambiguous_times).sum())
            pre = pre.loc[~pre["entry_time"].isin(ambiguous_times)].reset_index(drop=True)
    _validate_clock(pre, require_reservation=False)
    return pre, {
        "pre_reservation_rows": len(pre),
        "duplicate_rows_dropped": duplicate_rows_dropped,
        "ambiguous_same_timestamp_both_sides_rows_dropped": ambiguous_rows_dropped,
        "missing_component_window_rejections": missing_component_rejections,
        "no_exact_trigger_rejections": no_exact_trigger_rejections,
        "availability_rejections": availability_rejections,
    }


def build_async_three_way_consensus_clock(
    left: str,
    middle: str,
    right: str,
    left_clock: pd.DataFrame,
    middle_clock: pd.DataFrame,
    right_clock: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build inclusive 6h same-side consensus for one unordered triple.

    At entry time t and side s, all three components must have a latest same-side
    event in [t-6h, t], and at least one selected event must occur exactly at t.
    If both sides qualify at the same t, both rows are dropped as ambiguous.
    """
    pre, diagnostics = build_async_three_way_consensus_pre_clock(left, middle, right, left_clock, middle_clock, right_clock)
    output = reserve_half_open(pre) if not pre.empty else pre
    _validate_clock(output)
    diagnostics = {
        **diagnostics,
        "post_reservation_rows": len(output),
        "reservation_dropped_rows": len(pre) - len(output),
    }
    return output, diagnostics


def _validate_clock(clock: pd.DataFrame, *, require_reservation: bool = True) -> None:
    if clock.empty:
        return
    if not clock["entry_time"].ge(TRAIN_START).all() or not clock["exit_time"].le(TRAIN_END).all():
        raise RuntimeError(f"{POLICY_ID} triple clock escaped train prefix")
    if not clock["decision_time"].le(clock["entry_time"]).all():
        raise RuntimeError(f"{POLICY_ID} triple decision after entry")
    if not clock["feature_available_time"].le(clock["entry_time"]).all():
        raise RuntimeError(f"{POLICY_ID} triple feature after entry")
    if not clock["side"].isin((-1, 1)).all():
        raise RuntimeError(f"{POLICY_ID} triple non-strict side")
    for column in ("left_selected_entry_time", "middle_selected_entry_time", "right_selected_entry_time"):
        lag = clock["entry_time"] - clock[column]
        if not (lag.ge(pd.Timedelta(0)).all() and lag.le(CONFIRMATION_WINDOW).all()):
            raise RuntimeError(f"{POLICY_ID} triple confirmation lag drift: {column}")
    if require_reservation and len(clock) > 1 and not clock["entry_time"].iloc[1:].reset_index(drop=True).ge(
        clock["exit_time"].iloc[:-1].reset_index(drop=True)
    ).all():
        raise RuntimeError(f"{POLICY_ID} triple reservation overlap")


def support_stats(clock: pd.DataFrame) -> dict[str, float | int]:
    if clock.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
            "distinct_iso_weeks": 0,
            "first_half_events": 0,
            "second_half_events": 0,
        }
    longs = int(clock["side"].eq(1).sum())
    shorts = int(clock["side"].eq(-1).sum())
    events = len(clock)
    month_counts = clock["entry_time"].dt.strftime("%Y-%m").value_counts()
    iso = clock["entry_time"].dt.isocalendar()
    weeks = set(zip(iso["year"].astype(int), iso["week"].astype(int), strict=True))
    midpoint = TRAIN_START + (TRAIN_END - TRAIN_START) / 2
    first_half = int(clock["entry_time"].lt(midpoint).sum())
    return {
        "events": events,
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / events,
        "max_month_share": int(month_counts.max()) / events,
        "distinct_iso_weeks": len(weeks),
        "first_half_events": first_half,
        "second_half_events": events - first_half,
    }


def support_checks(stats: Mapping[str, float | int]) -> dict[str, bool]:
    return {
        "minimum_events": stats["events"] >= SOURCE_GATES["minimum_events"],
        "side_balance": stats["minority_side_share"] >= SOURCE_GATES["minority_side_share_min"],
        "month_concentration": stats["max_month_share"] <= SOURCE_GATES["max_month_share"],
        "distinct_iso_weeks": stats["distinct_iso_weeks"] >= SOURCE_GATES["distinct_iso_weeks_min"],
        "both_train_halves": (
            stats["first_half_events"] >= SOURCE_GATES["minimum_events_each_train_half"]
            and stats["second_half_events"] >= SOURCE_GATES["minimum_events_each_train_half"]
        ),
    }


def _clock_key_set(clock: pd.DataFrame) -> set[tuple[pd.Timestamp, int]]:
    if clock.empty:
        return set()
    return set(zip(pd.to_datetime(clock["entry_time"], utc=True), clock["side"].astype(int), strict=True))


def _schedule_signature(clock: pd.DataFrame) -> tuple[tuple[str, str, int], ...]:
    if clock.empty:
        return tuple()
    ordered = clock.sort_values(["entry_time", "exit_time", "side"], kind="stable")
    return tuple(
        (pd.Timestamp(row.entry_time).isoformat(), pd.Timestamp(row.exit_time).isoformat(), int(row.side))
        for row in ordered.itertuples(index=False)
    )


def _matched_share(candidate_keys: set[tuple[pd.Timestamp, int]], prior_keys: set[tuple[pd.Timestamp, int]]) -> float:
    if not candidate_keys:
        return 0.0
    remaining = sorted(prior_keys)
    matched = 0
    for entry_time, side in sorted(candidate_keys):
        for index, (prior_time, prior_side) in enumerate(remaining):
            if prior_side == side and abs(entry_time - prior_time) <= CONFIRMATION_WINDOW:
                matched += 1
                remaining.pop(index)
                break
    return matched / len(candidate_keys)


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
    if not frame["entry_time"].ge(TRAIN_START).all() or not frame["entry_time"].lt(TRAIN_END).all() or not frame["exit_time"].le(TRAIN_END).all():
        raise RuntimeError(f"{POLICY_ID} prior clock schedule escaped train containment: {path}")
    if len(frame) != expected_rows:
        raise RuntimeError(f"{POLICY_ID} prior clock row-count drift: {path}")
    return frame.sort_values(["entry_time", "side"], kind="stable").reset_index(drop=True)


def _validate_manifest_hash(value: Mapping[str, Any], *, label: str) -> None:
    core = dict(value)
    manifest_hash = core.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} {label} source-support manifest drift")


def _prior_binding_by_policy(registration: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    bindings = registration.get("prior_clock_source_support_artifacts")
    if not isinstance(bindings, list):
        raise RuntimeError(f"{POLICY_ID} missing preregistered prior source-support bindings")
    by_policy: dict[str, Mapping[str, Any]] = {}
    for binding in bindings:
        if not isinstance(binding, dict) or not isinstance(binding.get("policy_id"), str):
            raise RuntimeError(f"{POLICY_ID} malformed preregistered prior source-support binding")
        policy_id = str(binding["policy_id"])
        if policy_id in by_policy:
            raise RuntimeError(f"{POLICY_ID} duplicate preregistered prior source-support binding")
        by_policy[policy_id] = binding
    return by_policy


def load_validated_prior_source_artifacts(registration: Mapping[str, Any]) -> dict[str, Any]:
    bound = _prior_binding_by_policy(registration)
    artifacts = {
        "same_side": {
            "binding": bound.get(same_side.POLICY_ID),
            "expected_policy_id": same_side.POLICY_ID,
            "expected_family": same_side.candidate_family(),
            "pairs_key": "pairs",
        },
        "handoff": {
            "binding": bound.get(handoff.POLICY_ID),
            "expected_policy_id": handoff.POLICY_ID,
            "expected_family": handoff.candidate_family(),
            "pairs_key": "pairs",
        },
    }
    out: dict[str, Any] = {}
    for label, spec in artifacts.items():
        binding = spec["binding"]
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"{POLICY_ID} missing preregistered {label} prior source-support binding")
        path = Path(str(binding.get("path", "")))
        if not path.exists():
            raise RuntimeError(f"{POLICY_ID} missing prior source-support artifact: {path}")
        observed_sha = sha256_file(path)
        if observed_sha != binding.get("sha256"):
            raise RuntimeError(f"{POLICY_ID} {label} prior source-support artifact SHA drift")
        value = _read_json_object(path)
        _validate_manifest_hash(value, label=label)
        if value.get("manifest_hash") != binding.get("manifest_hash"):
            raise RuntimeError(f"{POLICY_ID} {label} prior source-support manifest binding drift")
        if value.get("policy_id") != binding.get("policy_id") or value.get("policy_id") != spec["expected_policy_id"]:
            raise RuntimeError(f"{POLICY_ID} {label} prior policy drift")
        if tuple(value.get("candidate_family", ())) != tuple(spec["expected_family"]):
            raise RuntimeError(f"{POLICY_ID} {label} prior candidate-family drift")
        pairs = value.get(spec["pairs_key"], {})
        if not isinstance(pairs, dict) or len(pairs) != 36:
            raise RuntimeError(f"{POLICY_ID} {label} prior family rows drift")
        out[label] = {
            "path": str(path),
            "sha256": observed_sha,
            "manifest_hash": value["manifest_hash"],
            "policy_id": value["policy_id"],
            "candidate_family_size": value.get("candidate_family_size"),
            "schedule_scope": binding.get("schedule_scope"),
            "binding": dict(binding),
            "pairs": pairs,
        }
    return out


def load_prior_clock_schedules(
    component_order: Sequence[str] = COMPONENT_ORDER,
    same_side_clock_dir: Path = SAME_SIDE_CLOCK_DIR,
    handoff_clock_dir: Path = HANDOFF_CLOCK_DIR,
    prior_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if prior_artifacts is None:
        _, registration = load_validated_preregistration()
        prior_artifacts = load_validated_prior_source_artifacts(registration)
    schedules: dict[str, dict[str, Any]] = {}
    for left, right in combinations(component_order, 2):
        same_id = same_side.pair_id(left, right)
        handoff_id = handoff.pair_id(left, right)
        for family, candidate, configured_dir in (
            ("same_side", same_id, same_side_clock_dir),
            ("handoff", handoff_id, handoff_clock_dir),
        ):
            prior = prior_artifacts[family]["pairs"].get(candidate)
            if not prior:
                raise RuntimeError(f"{POLICY_ID} missing prior family candidate binding: {candidate}")
            clock_binding = prior.get("clock", {})
            path = Path(clock_binding.get("path", ""))
            expected_path = configured_dir / f"{candidate}.csv.gz"
            if path != expected_path:
                raise RuntimeError(f"{POLICY_ID} prior clock path binding drift: {candidate}")
            clock = _load_prior_clock(
                path,
                expected_sha256=str(clock_binding.get("sha256", "")),
                expected_rows=int(clock_binding.get("rows", -1)),
            )
            schedules[candidate] = {
                "family": family,
                "components": [left, right],
                "path": str(path),
                "sha256": clock_binding.get("sha256"),
                "rows": len(clock),
                "keys": _clock_key_set(clock),
                "signature": _schedule_signature(clock),
            }
    return schedules

def triple_pre_reservation_key_disclosure(
    pre_clock: pd.DataFrame,
    triple: Sequence[str],
    component_clocks: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    triple_keys = _clock_key_set(pre_clock)
    constituent: dict[str, Any] = {}
    union_keys: set[tuple[pd.Timestamp, int]] = set()
    for left, right in combinations(triple, 2):
        pair = same_side.pair_id(left, right)
        pair_keys = handoff.reconstruct_same_side_pre_reservation_keys(
            left, right, component_clocks[left], component_clocks[right]
        )
        union_keys.update(pair_keys)
        intersection = triple_keys & pair_keys
        union = triple_keys | pair_keys
        constituent[pair] = {
            "exact_intersection": len(intersection),
            "exact_jaccard": len(intersection) / len(union) if union else 1.0,
            "matched_share_6h": _matched_share(triple_keys, pair_keys),
            "pre_reservation_keys": len(pair_keys),
        }
    uncovered = triple_keys - union_keys
    if uncovered:
        raise RuntimeError(f"{POLICY_ID} triple pre-reservation keys not covered by constituent same-side pair union")
    union = triple_keys | union_keys
    return {
        "triple_pre_reservation_keys": len(triple_keys),
        "constituent_same_side_pair_pre_reservation_intersections": constituent,
        "constituent_same_side_pair_pre_reservation_union": {
            "keys": len(union_keys),
            "exact_intersection": len(triple_keys & union_keys),
            "exact_jaccard": len(triple_keys & union_keys) / len(union) if union else 1.0,
            "triple_key_coverage_share": 1.0 if not triple_keys else len(triple_keys & union_keys) / len(triple_keys),
        },
    }


def prior_clock_disclosure(
    candidate_clock: pd.DataFrame,
    triple: Sequence[str],
    prior_schedules: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_keys = _clock_key_set(candidate_clock)
    candidate_signature = _schedule_signature(candidate_clock)
    exact_duplicates: list[str] = []
    constituent: dict[str, Any] = {}
    handoff_exact_overlap = 0
    union_keys_by_family: dict[str, set[tuple[pd.Timestamp, int]]] = {"same_side": set(), "handoff": set()}
    prior_candidate_overlaps: dict[str, Any] = {}
    triple_set = set(triple)
    for prior_id, prior in prior_schedules.items():
        prior_keys = set(prior["keys"])
        family = str(prior["family"])
        union_keys_by_family[family].update(prior_keys)
        intersection = candidate_keys & prior_keys
        union = candidate_keys | prior_keys
        jaccard = len(intersection) / len(union) if union else 1.0
        matched = _matched_share(candidate_keys, prior_keys)
        if family == "handoff":
            handoff_exact_overlap += len(intersection)
        if candidate_signature == prior["signature"]:
            exact_duplicates.append(prior_id)
        if set(prior["components"]).issubset(triple_set):
            constituent[prior_id] = {
                "family": family,
                "exact_intersection": len(intersection),
                "exact_jaccard": jaccard,
                "matched_share_6h": matched,
                "rows": prior["rows"],
            }
        if intersection or matched:
            prior_candidate_overlaps[prior_id] = {
                "family": family,
                "exact_intersection": len(intersection),
                "exact_jaccard": jaccard,
                "matched_share_6h": matched,
            }
    union_disclosure = {}
    for family, keys in union_keys_by_family.items():
        intersection = candidate_keys & keys
        union = candidate_keys | keys
        union_disclosure[family] = {
            "exact_intersection": len(intersection),
            "exact_jaccard": len(intersection) / len(union) if union else 1.0,
            "matched_share_6h": _matched_share(candidate_keys, keys),
            "union_rows": len(keys),
        }
    return {
        "constituent_pair_intersections": constituent,
        "prior_family_unions": union_disclosure,
        "prior_candidate_overlaps": prior_candidate_overlaps,
        "incidental_handoff_exact_overlap": handoff_exact_overlap,
        "exact_full_schedule_duplicates": exact_duplicates,
        "exact_duplicate_reject": bool(exact_duplicates),
    }


def run(
    clock_dir: Path = CLOCK_DIR,
    result_path: Path = RESULT,
    same_side_clock_dir: Path = SAME_SIDE_CLOCK_DIR,
    handoff_clock_dir: Path = HANDOFF_CLOCK_DIR,
) -> dict[str, Any]:
    prereg, registration = load_validated_preregistration()
    verified = verify_bound_component_artifacts()
    clocks = {component: load_train_prefix_clock(component) for component in COMPONENT_ORDER}
    prior_artifacts = load_validated_prior_source_artifacts(registration)
    prior_schedules = load_prior_clock_schedules(COMPONENT_ORDER, same_side_clock_dir, handoff_clock_dir, prior_artifacts)
    clock_dir.mkdir(parents=True, exist_ok=True)

    triples: dict[str, Any] = {}
    passed_triples: list[str] = []
    duplicate_rejects: list[str] = []
    for left, middle, right in combinations(COMPONENT_ORDER, 3):
        candidate = triple_id(left, middle, right)
        pre_clock, pre_diagnostics = build_async_three_way_consensus_pre_clock(
            left, middle, right, clocks[left], clocks[middle], clocks[right]
        )
        clock = reserve_half_open(pre_clock) if not pre_clock.empty else pre_clock
        _validate_clock(clock)
        diagnostics = {
            **pre_diagnostics,
            "post_reservation_rows": len(clock),
            "reservation_dropped_rows": len(pre_clock) - len(clock),
        }
        pre_disclosure = triple_pre_reservation_key_disclosure(pre_clock, (left, middle, right), clocks)
        disclosure = prior_clock_disclosure(clock, (left, middle, right), prior_schedules)
        path = clock_dir / f"{candidate}.csv.gz"
        _write_gzip_csv(clock, path)
        stats = support_stats(clock)
        checks = support_checks(stats)
        duplicate_reject = bool(disclosure["exact_duplicate_reject"])
        passed = all(checks.values()) and not duplicate_reject
        if duplicate_reject:
            duplicate_rejects.append(candidate)
        if passed:
            passed_triples.append(candidate)
        triples[candidate] = {
            "components": [left, middle, right],
            "operator": "symmetric_async_same_side_three_way_latest_each_component_within_6h",
            "clock": {"path": str(path), "sha256": sha256_file(path), "rows": len(clock)},
            "construction_diagnostics": diagnostics,
            "pre_reservation_disclosure": pre_disclosure,
            "prior_clock_disclosure": disclosure,
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
        "candidate_family_size": 84,
        "preregistration": {
            "path": str(getattr(prereg, "DEFAULT_OUTPUT")),
            "sha256": sha256_file(getattr(prereg, "DEFAULT_OUTPUT")),
            "manifest_hash": registration["manifest_hash"],
        },
        "verified_component_artifacts": verified,
        "train_prefix": {
            "start_inclusive": TRAIN_START.isoformat().replace("+00:00", "Z"),
            "end_exclusive": TRAIN_END.isoformat().replace("+00:00", "Z"),
            "component_rows_loaded": {component: len(clock) for component, clock in clocks.items()},
        },
        "prior_source_support_artifacts": {
            family: {key: value for key, value in artifact.items() if key != "pairs"}
            for family, artifact in prior_artifacts.items()
        },
        "prior_clock_schedules_opened": {
            "same_side_family_size": sum(1 for item in prior_schedules.values() if item["family"] == "same_side"),
            "handoff_family_size": sum(1 for item in prior_schedules.values() if item["family"] == "handoff"),
            "total_prior_schedules": len(prior_schedules),
            "missing_prior_schedules": [],
            "authenticated_via_prior_source_support_artifacts": True,
        },
        "construction": {
            "operator": "for each unordered triple and side, emit at t when all three components have latest same-side entries in inclusive [t-6h,t] and at least one selected entry is exactly t",
            "trigger_component": "earliest frozen component order among selected components whose selected entry equals t",
            "ambiguous_same_timestamp_policy": "if long and short both qualify at the same t, drop both",
            "decision_time": "max(selected component decision_time)",
            "feature_available_time": "max(selected component feature_available_time)",
            "entry_time": "consensus trigger timestamp t",
            "exit_time": "entry_time + 8 elapsed hours",
            "reservation": "chronological half-open reservation per triple after dedupe and ambiguity drop",
            "exact_full_schedule_duplicate_policy": "reject exact duplicates against prior 72 clock schedules only; overlaps are disclosure-only",
        },
        "source_support_gates": SOURCE_GATES,
        "triples": triples,
        "passed_triples": passed_triples,
        "duplicate_rejects": duplicate_rejects,
        "support_passed_any_triple": bool(passed_triples),
        "evidence_boundary": {
            "component_clock_fields_opened": list(COMMON_CLOCK_FIELDS),
            "component_clock_rows_materialized_train_prefix_only": True,
            "prior_clock_schedule_fields_opened": ["entry_time", "exit_time", "side"],
            "prior_clock_schedules_opened_for_disclosure_only": True,
            "gross9_rows_opened": False,
            "market_rows_opened": False,
            "entry_exit_prices_opened": False,
            "funding_opened": False,
            "combination_returns_or_pnl_opened": False,
            "combination_economic_outcomes_opened": False,
            "bound_component_train_economics_artifact_opened_for_pass_authentication": True,
            "component_metric_fields_used_for_selection": [],
            "oos_component_rows_materialized": 0,
        },
        "decision": "pass_supported_triples_to_gross9_novelty" if passed_triples else "terminal_no_source_supported_triples",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clock-dir", type=Path, default=CLOCK_DIR)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--same-side-clock-dir", type=Path, default=SAME_SIDE_CLOCK_DIR)
    parser.add_argument("--handoff-clock-dir", type=Path, default=HANDOFF_CLOCK_DIR)
    args = parser.parse_args()
    report = run(args.clock_dir, args.result, args.same_side_clock_dir, args.handoff_clock_dir)
    print(json.dumps({"passed_triples": report["passed_triples"], "candidate_family_size": report["candidate_family_size"]}, indent=2))
