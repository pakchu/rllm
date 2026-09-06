"""Outcome-blind preregistration for the Gross9 async 3-way same-side search.

This freezes only the G9ASYNC3WAY-8 preregistration contract.  It binds the
same nine immutable train-economic PASS components and both terminal predecessor
economic artifacts, while opening no new 3-way incidence, market, funding, or
outcome rows.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import preregister_gross9_async_pair_search as same_side


POLICY_ID = "G9ASYNC3WAY-8"
PROTOCOL_VERSION = "gross9_async_same_side_three_way_consensus_search_preregistration_v1"
AS_OF_DATE = "2026-09-02"
DEFAULT_OUTPUT = Path("results/gross9_async_same_side_three_way_consensus_search_preregistration_2026-09-02.json")
FAMILYWISE_ALPHA = 0.10
COMPONENT_ORDER = same_side.COMPONENT_ORDER
COMPONENT_ARTIFACTS = same_side.COMPONENT_ARTIFACTS
GROSS9_PRE2025_CLOCK_MANIFEST = same_side.GROSS9_PRE2025_CLOCK_MANIFEST
CANDIDATE_FAMILY = tuple(
    f"{first}__ASYNC_SAME_SIDE_3WAY_6H__{second}__{third}"
    for first, second, third in combinations(COMPONENT_ORDER, 3)
)
FAMILY_SIZE = 84
BONFERRONI_RAW_P_MAX = FAMILYWISE_ALPHA / FAMILY_SIZE
TRAIN_WINDOW = same_side.TRAIN_WINDOW
SAME_SIDE_TERMINAL_TRAIN_ECONOMICS = {
    "policy_id": "G9ASYNCPAIR-8",
    "path": "results/gross9_async_pair_train_economics_2026-09-02.json",
    "sha256": "0b822d77415ca70a409d2e7f3c35ebe44cbf481aa7e0d2eb02605646bdb3f874",
    "manifest_hash": "bb3ed8afa1eec6cddf2344515d89736a36314157ad7eeac495c759adadc45b16",
    "decision": "terminal_train_reject_no_substitution",
}
OPPOSITION_HANDOFF_TERMINAL_TRAIN_ECONOMICS = {
    "policy_id": "G9ASYNCHANDOFF-8",
    "path": "results/gross9_async_opposition_handoff_train_economics_2026-09-02.json",
    "sha256": "a2c86ae78940a331f1e0209fa5bbb8bdb374fd2d4438030900ffd3a097b85e64",
    "manifest_hash": "3ad6368a44519359cf7661b7707a9f099ae23fb679c6c2115aafab095a51aa3a",
    "decision": "terminal_train_reject_no_substitution",
}

PRIOR_CLOCK_SOURCE_SUPPORT_ARTIFACTS = [
    {
        "policy_id": "G9ASYNCPAIR-8",
        "path": "results/gross9_async_pair_train_clock_source_support_2026-09-02.json",
        "sha256": "c6d3929f282ba1075c2ebc091e4bc62164b923a038bce94de32884aaf7ff0009",
        "manifest_hash": "b92d3afb7a3539cdd194eddc1ab09bc65068716135d0bca575db0531ac450011",
        "schedule_scope": "same-side pair family post-reservation clocks plus constituent same-side pre-reservation schedules for overlap disclosure",
    },
    {
        "policy_id": "G9ASYNCHANDOFF-8",
        "path": "results/gross9_async_opposition_handoff_train_clock_source_support_2026-09-02.json",
        "sha256": "a8982c1b6e155f65f76af4559ca2d01b2a7824cb5c58524a260b72beb997f754",
        "manifest_hash": "92501aa4c921bba20d05378b6f658f33d6c712e8b3adb9f095940dd44ac3f3b0",
        "schedule_scope": "opposition handoff pair family post-reservation clocks for overlap disclosure",
    },
]


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "objective": "continue the non-LLM Gross9-like signal-alpha search with an untried asynchronous same-side three-component consensus operator over the exact same nine frozen components",
        "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False,
        "predecessor_family_terminal_results": [
            copy.deepcopy(SAME_SIDE_TERMINAL_TRAIN_ECONOMICS),
            copy.deepcopy(OPPOSITION_HANDOFF_TERMINAL_TRAIN_ECONOMICS),
        ],
        "prior_clock_source_support_artifacts": copy.deepcopy(PRIOR_CLOCK_SOURCE_SUPPORT_ARTIFACTS),
        "component_order": list(COMPONENT_ORDER),
        "component_count": len(COMPONENT_ORDER),
        "component_artifacts": copy.deepcopy(COMPONENT_ARTIFACTS),
        "gross9_pre2025_clock_manifest": copy.deepcopy(GROSS9_PRE2025_CLOCK_MANIFEST),
        "implementation": {
            "preregister": {
                "path": "training/preregister_gross9_async_three_way_consensus_search.py",
                "sha256": sha256_file(__file__),
            },
            "train_clock_builder": {
                "path": "training/build_gross9_async_three_way_consensus_train_clocks.py",
                "sha256": "f6aa724081826475dbfa7b489678003a97655171f3ef9446a3af35b2d571e0a3",
            },
        },
        "candidate_family": list(CANDIDATE_FAMILY),
        "candidate_family_size": FAMILY_SIZE,
        "construction": {
            "operator": "asynchronous same-side three-component consensus",
            "triple_definition": "all 84 unordered triples from the same nine frozen train-economic PASS components used by G9ASYNCPAIR-8 and G9ASYNCHANDOFF-8",
            "entry_rule": "for a triple (A,B,C), at time t and side s, require each of the three components to have its latest same-side event in the inclusive elapsed-time window [t-6h,t], with at least one selected event exactly at t; opposite-side events are ignored for this operator",
            "trigger_canonicalization": "simultaneous two-of-three or three-of-three triggers are allowed; when more than one component has the selected event at t, the canonical trigger component is the earliest component in frozen component_order",
            "dual_side_same_timestamp_policy": "if both long and short sides qualify for the same triple at the same timestamp t, drop both candidate rows before dedupe and reservation",
            "availability": "max(decision_time and feature_available_time across the three selected latest same-side events) must be <= t; no selected event after t may affect the decision",
            "lookback_window": "inclusive [t-6h,t] == [candidate timestamp minus 6 elapsed hours, candidate timestamp] for each component's latest same-side event",
            "output_entry_time": "consensus timestamp t",
            "output_side": "qualified consensus side s",
            "duplicate_policy": "deduplicate constructed rows by candidate, entry_time, and side before reservation",
            "reservation": "one chronological half-open 8h reservation inside each candidate-triple clock after dedupe; touching intervals are allowed",
            "prior_family_duplicate_gate": "after triple-local reservation, an exact post-reservation schedule duplicate of any prior 72 same-side/opposition candidate clocks is a source-support failure; pre-reservation constituent overlap is disclosure-only",
            "overlap_disclosure": "report overlap against constituent same-side pre-reservation schedules and all prior-72 same-side/opposition post-reservation schedules; only exact post-reservation schedule duplication is a hard gate",
            "grids_or_variants": "none; no lag, hold, side, operator, threshold, component, subset, weight, ordering, or overlap-threshold variants",
            "component_formula_threshold_clock_mutability": "immutable",
        },
        "clock": {
            "entry": "same-side three-component consensus timestamp t",
            "hold": "8 elapsed hours",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
        },
        "stages": {
            "train": list(TRAIN_WINDOW),
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "must_pass_before_economics": True,
            "minimum_events": {"train": 10, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.20,
            "max_month_share": 0.45,
            "distinct_iso_weeks_min": 10,
            "each_calendar_half_min_events": 1,
            "prior_family_exact_duplicate_gate": "no exact post-reservation schedule duplicate of any prior 72 same-side/opposition candidates",
            "prior_family_overlap_disclosure_scope": "constituent same-side pre-reservation schedules and all prior-72 same-side/opposition post-reservation schedules",
            "prior_family_overlap_disclosure_required": True,
            "prior_family_overlap_non_exact_is_gate": False,
        },
        "gross9_novelty_gates": {
            "must_pass_before_economics": True,
            "exact_entry_jaccard_max": 0.10,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "comparator": GROSS9_PRE2025_CLOCK_MANIFEST,
        },
        "economic_gates": {
            "train_window_only_for_initial_rank_and_gate": list(TRAIN_WINDOW),
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": BONFERRONI_RAW_P_MAX,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "costs": {"base_each_notional_side_bp": 6, "stress_each_notional_side_bp": 10},
            "accounting": "fixed quantity, exact funding, every held 5m favorable then adverse, global HWM, full-calendar CAGR",
        },
        "selection": {
            "eligibility": "triple passes source-support including prior-family exact-duplicate gate and Gross9 novelty gates before any economics",
            "ranking_metric": "descending train base-cost CAGR divided by strict MDD",
            "tie_breaks": ["descending train base-cost absolute return", "ascending frozen candidate_family order"],
            "raw_rank_one_no_substitution": True,
            "winner_train_gate": "raw rank-one triple must pass every train economic gate; failure terminates the family",
            "later_stages": "after winner freeze, evaluate test/eval/final sequentially; stop on first failure; no rerank or repair",
        },
        "familywise_multiplicity": {
            "family": "all C(9,3)=84 async same-side three-way consensus triples, including candidates later failing source, prior-family duplicate, or Gross9 gates",
            "scope_boundary": "controls this fixed 84-hypothesis family only; it does not control the cumulative adaptive exploratory research program across prior or future families",
            "rule": "Bonferroni",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "number_of_hypotheses": FAMILY_SIZE,
            "winner_raw_weekly_signflip_p_max": BONFERRONI_RAW_P_MAX,
            "equivalent_adjusted_p": "min(1, 84 * raw one-sided weekly sign-flip p)",
        },
        "research_boundary": {
            "llm_path_paused": True,
            "same_side_terminal_artifact_bound": True,
            "opposition_handoff_terminal_artifact_bound": True,
            "predecessor_economic_scalars_used_to_tune_operator_or_gates": False,
            "predecessor_economic_scalars_used_to_change_components": False,
            "component_standalone_train_outcomes_known": True,
            "component_standalone_outcomes_used_to_change_components": False,
            "component_formulas_thresholds_clocks_frozen": True,
            "component_clock_rows_opened_by_preregistration": 0,
            "three_way_combination_incidence_opened_by_preregistration": False,
            "three_way_combination_outcomes_opened_by_preregistration": False,
            "market_or_funding_rows_opened_by_preregistration": False,
            "test_eval_final_outcomes_opened_before_winner_freeze": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": "Fixed sequence over the frozen 84-triple family: source support including overlap disclosure and exact prior-family duplicate gate, Gross9 novelty, train-only raw ranking and rank-one train gates, then frozen-winner test/eval/final economics. Stop on first failure with no triple substitution, formula, threshold, side, clock, hold, subset, cost, control, overlap, or rerank repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    core = dict(value)
    manifest_hash = core.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(core):
        raise RuntimeError("G9ASYNC3WAY-8 preregistration drift")
    if value.get("protocol_version") != PROTOCOL_VERSION or value.get("policy_id") != POLICY_ID:
        raise RuntimeError("G9ASYNC3WAY-8 protocol identity drift")
    if tuple(value.get("component_order", ())) != COMPONENT_ORDER:
        raise RuntimeError("G9ASYNC3WAY-8 component family drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("G9ASYNC3WAY-8 candidate family drift")
    if value.get("candidate_family_size") != FAMILY_SIZE:
        raise RuntimeError("G9ASYNC3WAY-8 family size drift")
    if value.get("economic_gates", {}).get("weekly_signflip_one_sided_p_max") != BONFERRONI_RAW_P_MAX:
        raise RuntimeError("G9ASYNC3WAY-8 Bonferroni drift")
    source = value.get("source_support_gates", {})
    if source.get("minimum_events", {}).get("train") != 10:
        raise RuntimeError("G9ASYNC3WAY-8 train event source gate drift")
    if source.get("distinct_iso_weeks_min") != 10:
        raise RuntimeError("G9ASYNC3WAY-8 distinct-week gate drift")
    if source.get("each_calendar_half_min_events") != 1:
        raise RuntimeError("G9ASYNC3WAY-8 calendar-half source gate drift")
    if source.get("prior_family_overlap_disclosure_required") is not True or source.get("prior_family_overlap_non_exact_is_gate") is not False:
        raise RuntimeError("G9ASYNC3WAY-8 overlap disclosure gate drift")
    if "exact post-reservation schedule duplicate" not in value.get("construction", {}).get("prior_family_duplicate_gate", ""):
        raise RuntimeError("G9ASYNC3WAY-8 prior-family duplicate gate drift")
    if "constituent same-side pre-reservation" not in value.get("construction", {}).get("overlap_disclosure", ""):
        raise RuntimeError("G9ASYNC3WAY-8 overlap disclosure scope drift")
    if "both long and short sides qualify" not in value.get("construction", {}).get("dual_side_same_timestamp_policy", ""):
        raise RuntimeError("G9ASYNC3WAY-8 dual-side timestamp policy drift")
    predecessors = value.get("predecessor_family_terminal_results")
    if predecessors != [SAME_SIDE_TERMINAL_TRAIN_ECONOMICS, OPPOSITION_HANDOFF_TERMINAL_TRAIN_ECONOMICS]:
        raise RuntimeError("G9ASYNC3WAY-8 predecessor terminal binding drift")
    if value.get("prior_clock_source_support_artifacts") != PRIOR_CLOCK_SOURCE_SUPPORT_ARTIFACTS:
        raise RuntimeError("G9ASYNC3WAY-8 prior source-support binding drift")
    boundary = value.get("research_boundary", {})
    if boundary.get("predecessor_economic_scalars_used_to_tune_operator_or_gates") is not False:
        raise RuntimeError("G9ASYNC3WAY-8 predecessor scalar tuning boundary drift")
    if boundary.get("three_way_combination_incidence_opened_by_preregistration") is not False:
        raise RuntimeError("G9ASYNC3WAY-8 incidence boundary drift")
    if boundary.get("three_way_combination_outcomes_opened_by_preregistration") is not False:
        raise RuntimeError("G9ASYNC3WAY-8 outcome boundary drift")
    if "cumulative adaptive exploratory research program" not in value.get("familywise_multiplicity", {}).get("scope_boundary", ""):
        raise RuntimeError("G9ASYNC3WAY-8 cumulative-control disclosure drift")


def _check_bool(row: Mapping[str, Any], flag: str) -> bool:
    value = row.get(flag)
    if type(value) is not bool:
        raise ValueError(f"G9ASYNC3WAY-8 {flag} must be boolean")
    return value


def select_train_winner(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze the raw rank-one row after the future 84-triple train search is run."""
    if len(rows) != FAMILY_SIZE:
        raise ValueError("G9ASYNC3WAY-8 selection requires exactly 84 candidate rows")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        candidate = row.get("candidate")
        if not isinstance(candidate, str) or candidate in by_id:
            raise ValueError("G9ASYNC3WAY-8 candidate IDs must be unique strings")
        by_id[candidate] = row
    if set(by_id) != set(CANDIDATE_FAMILY):
        raise ValueError("G9ASYNC3WAY-8 selection requires the exact frozen family")

    eligible: list[tuple[int, Mapping[str, Any], float, float]] = []
    for order, candidate in enumerate(CANDIDATE_FAMILY):
        row = by_id[candidate]
        source_pass = _check_bool(row, "source_pass")
        gross9_pass = _check_bool(row, "gross9_pass")
        train_pass = _check_bool(row, "train_economic_pass")
        if not source_pass or not gross9_pass:
            continue
        ratio = row.get("train_cagr_to_strict_mdd")
        absolute = row.get("train_absolute_return")
        if (
            isinstance(ratio, bool)
            or isinstance(absolute, bool)
            or not isinstance(ratio, (int, float))
            or not isinstance(absolute, (int, float))
            or not math.isfinite(float(ratio))
            or not math.isfinite(float(absolute))
        ):
            raise ValueError("G9ASYNC3WAY-8 train ranking metrics must be numeric")
        eligible.append((order, row, float(ratio), float(absolute)))
    if not eligible:
        raise RuntimeError("G9ASYNC3WAY-8 no source/Gross9 eligible triples")
    eligible.sort(key=lambda item: (-item[2], -item[3], item[0]))
    _, winner, ratio, absolute = eligible[0]
    if winner.get("train_economic_pass") is not True:
        raise RuntimeError("G9ASYNC3WAY-8 raw rank one failed train; no substitution")
    return {
        "candidate": winner["candidate"],
        "train_cagr_to_strict_mdd": ratio,
        "train_absolute_return": absolute,
        "frozen_before_test": True,
        "substitution_authorized": False,
        "rerank_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    value = build()
    validate(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
