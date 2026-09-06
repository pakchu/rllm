"""Materialize train-only Gross9 strict-opposition handoff pair clocks.

This builder is deliberately outcome-blind: it authenticates the same nine
hash-bound Gross9-passing component clocks, opens only primary clock timing/side
fields, keeps only the frozen train prefix, and writes deterministic eight-hour
hold clocks for all unordered component pairs.  It opens no market, funding,
Gross9 sleeve rows, returns, or PnL.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from training import build_gross9_async_pair_train_clocks as same_side
from training import preregister_gross9_async_opposition_handoff_search as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


POLICY_ID = prereg.POLICY_ID
PROTOCOL_VERSION = "gross9_async_opposition_handoff_train_clocks_v1"
TRAIN_START = same_side.TRAIN_START
TRAIN_END = same_side.TRAIN_END
CONFIRMATION_WINDOW = same_side.CONFIRMATION_WINDOW
HOLD = same_side.HOLD
COMPONENT_ORDER = prereg.COMPONENT_ORDER
COMPONENT_ARTIFACTS = prereg.COMPONENT_ARTIFACTS
COMMON_CLOCK_FIELDS = same_side.COMMON_CLOCK_FIELDS
CLOCK_DIR = Path("data/gross9_async_opposition_handoff_train_clocks_2026-09-02")
RESULT = Path("results/gross9_async_opposition_handoff_train_clock_source_support_2026-09-02.json")
SAME_SIDE_CLOCK_DIR = same_side.CLOCK_DIR
SOURCE_GATES = {
    "minimum_events": 8,
    "minority_side_share_min": 0.20,
    "max_month_share": 0.45,
    "distinct_iso_weeks_min": 9,
}
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


def sha256_file(path: str | Path) -> str:
    return same_side.sha256_file(path)


def canonical_hash(value: Any) -> str:
    return same_side.canonical_hash(value)


def pair_id(left: str, right: str) -> str:
    return f"{left}__ASYNC_OPPOSITION_HANDOFF_6H__{right}"


def same_side_pair_id(left: str, right: str) -> str:
    return same_side.pair_id(left, right)


def candidate_family(component_order: Sequence[str] = COMPONENT_ORDER) -> tuple[str, ...]:
    family = tuple(
        pair_id(left, right)
        for index, left in enumerate(component_order)
        for right in component_order[index + 1 :]
    )
    if tuple(component_order) == COMPONENT_ORDER and family != prereg.CANDIDATE_FAMILY:
        raise RuntimeError(f"{POLICY_ID} candidate family differs from preregistration")
    return family


def load_train_prefix_clock(component: str, component_artifacts: Mapping[str, Mapping[str, Mapping[str, str]]] = COMPONENT_ARTIFACTS) -> pd.DataFrame:
    return same_side.load_train_prefix_clock(component, component_artifacts)


def verify_bound_component_artifacts(
    component_artifacts: Mapping[str, Mapping[str, Mapping[str, str]]] = COMPONENT_ARTIFACTS,
    component_order: Sequence[str] = COMPONENT_ORDER,
) -> dict[str, Any]:
    return same_side.verify_bound_component_artifacts(component_artifacts, component_order)


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


def _strict_window(clock: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    return clock.loc[
        clock["entry_time"].lt(t)
        & clock["entry_time"].ge(t - CONFIRMATION_WINDOW)
    ].sort_values("entry_time", kind="stable")


def build_async_opposition_handoff_clock(
    left: str,
    right: str,
    left_clock: pd.DataFrame,
    right_clock: pd.DataFrame,
    *,
    same_side_pre_reservation_keys: set[tuple[pd.Timestamp, int]] | None = None,
    same_side_post_reservation_keys: set[tuple[pd.Timestamp, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build strict 6h opposition-handoff confirmations for one unordered pair.

    A later trigger at ``t`` is emitted only when the other component has at least
    one event in strict ``[t-6h, t)`` and every such event is opposite the trigger
    side.  Simultaneous cross-component events are excluded by the strict upper
    bound.  The latest opposite event is the confirmer.
    """
    for component, clock in ((left, left_clock), (right, right_clock)):
        if clock.empty or not clock["candidate"].eq(component).all():
            raise RuntimeError(f"{POLICY_ID} invalid component clock for {component}")

    by_component = {left: left_clock, right: right_clock}
    rows: list[dict[str, Any]] = []
    simultaneous_exclusions = 0
    same_side_window_rejections = 0
    no_other_window_rejections = 0
    events = pd.concat(
        [left_clock.assign(_component=left), right_clock.assign(_component=right)],
        ignore_index=True,
    ).sort_values(["entry_time", "_component"], kind="stable")

    simultaneous_times = set(left_clock["entry_time"]).intersection(set(right_clock["entry_time"]))
    for _, event in events.iterrows():
        trigger = str(event["_component"])
        other = right if trigger == left else left
        entry_time = pd.Timestamp(event["entry_time"])
        if entry_time in simultaneous_times:
            simultaneous_exclusions += 1
            continue
        other_window = _strict_window(by_component[other], entry_time)
        if other_window.empty:
            no_other_window_rejections += 1
            continue
        same_side_count = int(other_window["side"].eq(event["side"]).sum())
        if same_side_count:
            same_side_window_rejections += same_side_count
            continue
        opposite = other_window.loc[other_window["side"].eq(-int(event["side"]))]
        if opposite.empty:
            raise RuntimeError(f"{POLICY_ID} strict-window invariant drift for {left}/{right}")
        confirmer = opposite.iloc[-1]
        rows.append(
            {
                "candidate": pair_id(left, right),
                "control": "primary",
                "split": "train",
                "left_component_id": left,
                "right_component_id": right,
                "trigger_component_id": trigger,
                "confirming_component_id": other,
                "trigger_entry_time": entry_time,
                "confirming_entry_time": confirmer["entry_time"],
                "decision_time": max(event["decision_time"], confirmer["decision_time"]),
                "feature_available_time": max(
                    event["decision_time"],
                    event["feature_available_time"],
                    confirmer["decision_time"],
                    confirmer["feature_available_time"],
                ),
                "entry_time": entry_time,
                "exit_time": entry_time + HOLD,
                "side": int(event["side"]),
            }
        )

    pre_reservation = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if not pre_reservation.empty:
        pre_reservation = pre_reservation.drop_duplicates(["candidate", "entry_time", "side"], keep="first").reset_index(drop=True)
    pre_overlap_count = 0
    post_overlap_count = 0
    if not pre_reservation.empty:
        handoff_keys = set(zip(pre_reservation["entry_time"], pre_reservation["side"].astype(int), strict=True))
        if same_side_pre_reservation_keys is not None:
            pre_overlap_count = len(handoff_keys.intersection(same_side_pre_reservation_keys))
            if pre_overlap_count:
                raise RuntimeError(f"{POLICY_ID} pre-reservation same-side entry intersection for {left}/{right}: {pre_overlap_count}")
        if same_side_post_reservation_keys is not None:
            post_overlap_count = len(handoff_keys.intersection(same_side_post_reservation_keys))

    output = reserve_half_open(pre_reservation) if not pre_reservation.empty else pre_reservation
    _validate_clock(output)
    diagnostics = {
        "pre_reservation_rows": len(pre_reservation),
        "post_reservation_rows": len(output),
        "reservation_dropped_rows": len(pre_reservation) - len(output),
        "simultaneous_component_event_exclusions": simultaneous_exclusions,
        "no_other_strict_window_rejections": no_other_window_rejections,
        "same_side_strict_window_rejections": same_side_window_rejections,
        "same_side_pre_reservation_entry_intersection": pre_overlap_count,
        "same_side_post_reservation_entry_intersection_diagnostic": post_overlap_count,
    }
    return output, diagnostics


def _validate_clock(clock: pd.DataFrame) -> None:
    if clock.empty:
        return
    if not clock["entry_time"].ge(TRAIN_START).all() or not clock["exit_time"].le(TRAIN_END).all():
        raise RuntimeError(f"{POLICY_ID} pair clock escaped train prefix")
    if not clock["decision_time"].le(clock["entry_time"]).all():
        raise RuntimeError(f"{POLICY_ID} pair decision after entry")
    if not clock["feature_available_time"].le(clock["entry_time"]).all():
        raise RuntimeError(f"{POLICY_ID} pair feature after entry")
    if not clock["side"].isin((-1, 1)).all():
        raise RuntimeError(f"{POLICY_ID} pair non-strict side")
    lag = clock["entry_time"] - clock["confirming_entry_time"]
    if not (lag.gt(pd.Timedelta(0)).all() and lag.le(CONFIRMATION_WINDOW).all()):
        raise RuntimeError(f"{POLICY_ID} pair strict confirmation lag drift")
    if (clock["side"].to_numpy() == 0).any():
        raise RuntimeError(f"{POLICY_ID} zero side drift")
    if len(clock) > 1 and not clock["entry_time"].iloc[1:].reset_index(drop=True).ge(
        clock["exit_time"].iloc[:-1].reset_index(drop=True)
    ).all():
        raise RuntimeError(f"{POLICY_ID} pair reservation overlap")


def reconstruct_same_side_pre_reservation_keys(
    left: str,
    right: str,
    left_clock: pd.DataFrame,
    right_clock: pd.DataFrame,
) -> set[tuple[pd.Timestamp, int]]:
    """Rebuild same-side PRE-reservation keys from bound components only.

    Mirrors G9ASYNCPAIR-8 construction before reservation: other component's
    latest same-side event in inclusive [t-6h,t], with exact simultaneous rows
    deduped once by (entry_time, side).  Returns pair-local (entry_time, side)
    keys for the disjointness invariant.
    """
    by_component = {left: left_clock, right: right_clock}
    keys: set[tuple[pd.Timestamp, int]] = set()
    seen_simultaneous: set[tuple[pd.Timestamp, int]] = set()
    events = pd.concat(
        [left_clock.assign(_component=left), right_clock.assign(_component=right)],
        ignore_index=True,
    ).sort_values(["entry_time", "_component"], kind="stable")
    for _, event in events.iterrows():
        trigger = str(event["_component"])
        other = right if trigger == left else left
        entry_time = pd.Timestamp(event["entry_time"])
        side = int(event["side"])
        candidates = by_component[other].loc[
            by_component[other]["side"].eq(side)
            & by_component[other]["entry_time"].le(entry_time)
            & by_component[other]["entry_time"].ge(entry_time - CONFIRMATION_WINDOW)
        ]
        if candidates.empty:
            continue
        confirmer_entry = pd.Timestamp(candidates.iloc[-1]["entry_time"])
        if confirmer_entry == entry_time:
            simultaneous_key = (entry_time, side)
            if simultaneous_key in seen_simultaneous:
                continue
            seen_simultaneous.add(simultaneous_key)
        keys.add((entry_time, side))
    return keys


def load_same_side_post_reservation_keys(left: str, right: str, clock_dir: Path = SAME_SIDE_CLOCK_DIR) -> set[tuple[pd.Timestamp, int]]:
    path = clock_dir / f"{same_side_pair_id(left, right)}.csv.gz"
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["entry_time", "side"])
    if frame.empty:
        return set()
    entries = pd.to_datetime(frame["entry_time"], utc=True)
    return set(zip(entries, frame["side"].astype(int), strict=True))


def support_stats(clock: pd.DataFrame) -> dict[str, float | int]:
    if clock.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
            "distinct_iso_weeks": 0,
        }
    longs = int(clock["side"].eq(1).sum())
    shorts = int(clock["side"].eq(-1).sum())
    events = len(clock)
    month_counts = clock["entry_time"].dt.strftime("%Y-%m").value_counts()
    iso = clock["entry_time"].dt.isocalendar()
    weeks = set(zip(iso["year"].astype(int), iso["week"].astype(int), strict=True))
    return {
        "events": events,
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / events,
        "max_month_share": int(month_counts.max()) / events,
        "distinct_iso_weeks": len(weeks),
    }


def support_checks(stats: Mapping[str, float | int]) -> dict[str, bool]:
    return {
        "minimum_events": stats["events"] >= SOURCE_GATES["minimum_events"],
        "side_balance": stats["minority_side_share"] >= SOURCE_GATES["minority_side_share_min"],
        "month_concentration": stats["max_month_share"] <= SOURCE_GATES["max_month_share"],
        "distinct_iso_weeks": stats["distinct_iso_weeks"] >= SOURCE_GATES["distinct_iso_weeks_min"],
    }


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} expected JSON object: {path}")
    return value


def load_validated_preregistration() -> Mapping[str, Any]:
    if not prereg.DEFAULT_OUTPUT.is_file():
        raise RuntimeError(f"{POLICY_ID} missing committed preregistration artifact")
    registration = _read_json_object(prereg.DEFAULT_OUTPUT)
    prereg.validate(registration)
    if registration != prereg.build():
        raise RuntimeError(f"{POLICY_ID} preregistration artifact differs from code")
    return registration


def run(clock_dir: Path = CLOCK_DIR, result_path: Path = RESULT, same_side_clock_dir: Path = SAME_SIDE_CLOCK_DIR) -> dict[str, Any]:
    registration = load_validated_preregistration()
    verified = verify_bound_component_artifacts()
    clocks = {component: load_train_prefix_clock(component) for component in COMPONENT_ORDER}
    clock_dir.mkdir(parents=True, exist_ok=True)

    pairs: dict[str, Any] = {}
    passed_pairs: list[str] = []
    for index, left in enumerate(COMPONENT_ORDER):
        for right in COMPONENT_ORDER[index + 1 :]:
            candidate = pair_id(left, right)
            same_side_pre_keys = reconstruct_same_side_pre_reservation_keys(left, right, clocks[left], clocks[right])
            same_side_post_keys = load_same_side_post_reservation_keys(left, right, same_side_clock_dir)
            clock, diagnostics = build_async_opposition_handoff_clock(
                left,
                right,
                clocks[left],
                clocks[right],
                same_side_pre_reservation_keys=same_side_pre_keys,
                same_side_post_reservation_keys=same_side_post_keys,
            )
            path = clock_dir / f"{candidate}.csv.gz"
            _write_gzip_csv(clock, path)
            stats = support_stats(clock)
            checks = support_checks(stats)
            passed = all(checks.values())
            if passed:
                passed_pairs.append(candidate)
            pairs[candidate] = {
                "components": [left, right],
                "operator": "symmetric_async_strict_opposition_handoff_latest_other_within_6h",
                "clock": {"path": str(path), "sha256": sha256_file(path), "rows": len(clock)},
                "construction_diagnostics": diagnostics,
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
            "operator": "for each unordered pair, emit at unique later trigger t when the other component has at least one opposite event and zero same-side events in strict [t-6h,t)",
            "simultaneous_component_events_excluded": True,
            "confirming_entry": "latest opposite event in the other component strict window",
            "decision_time": "max(trigger decision_time, confirming decision_time)",
            "feature_available_time": "max(trigger feature_available_time, confirming feature_available_time)",
            "entry_time": "trigger component entry_time t",
            "exit_time": "entry_time + 8 elapsed hours",
            "reservation": "chronological half-open reservation per pair after candidate materialization",
            "same_side_pre_reservation_entry_intersection_required": 0,
            "same_side_pre_reservation_reconstruction": "local reconstruction from the two bound component clocks using G9ASYNCPAIR-8 inclusive [t-6h,t] same-side semantics before reservation",
            "same_side_post_reservation_overlap": "diagnostic only; not used as a substitute for pre-reservation invariant",
        },
        "source_support_gates": SOURCE_GATES,
        "pairs": pairs,
        "passed_pairs": passed_pairs,
        "support_passed_any_pair": bool(passed_pairs),
        "evidence_boundary": {
            "component_clock_fields_opened": list(COMMON_CLOCK_FIELDS),
            "component_clock_rows_materialized_train_prefix_only": True,
            "source_incidence_may_have_been_seen_pre_artifact_by_parallel_scratch": True,
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
    parser.add_argument("--same-side-clock-dir", type=Path, default=SAME_SIDE_CLOCK_DIR)
    args = parser.parse_args()
    report = run(args.clock_dir, args.result, args.same_side_clock_dir)
    print(json.dumps({"passed_pairs": report["passed_pairs"], "candidate_family_size": report["candidate_family_size"]}, indent=2))
