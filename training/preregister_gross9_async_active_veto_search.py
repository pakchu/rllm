"""Outcome-blind preregistration for the Gross9 async active-veto search.

This freezes only the G9ASYNCACTIVEVETO-8 preregistration contract.  It binds
same nine immutable train-economic PASS components and the terminal predecessor
receipts from the prior Gross9 async families, while opening no new active-veto
incidence, market, funding, or outcome rows.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from itertools import permutations
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import preregister_gross9_async_pair_search as same_side


POLICY_ID = "G9ASYNCACTIVEVETO-8"
PROTOCOL_VERSION = "gross9_async_active_opposite_veto_search_preregistration_v1"
AS_OF_DATE = "2026-09-02"
DEFAULT_OUTPUT = Path("results/gross9_async_active_opposite_veto_search_preregistration_2026-09-02.json")
FAMILYWISE_ALPHA = 0.10
COMPONENT_ORDER = same_side.COMPONENT_ORDER
COMPONENT_ARTIFACTS = same_side.COMPONENT_ARTIFACTS
GROSS9_PRE2025_CLOCK_MANIFEST = same_side.GROSS9_PRE2025_CLOCK_MANIFEST
CANDIDATE_FAMILY = tuple(
    f"{base}__ASYNC_ACTIVE_OPPOSITE_VETO_6H__{veto}"
    for base, veto in permutations(COMPONENT_ORDER, 2)
)
FAMILY_SIZE = 72
BONFERRONI_RAW_P_MAX = FAMILYWISE_ALPHA / FAMILY_SIZE
TRAIN_WINDOW = same_side.TRAIN_WINDOW
BUILDER_SHA256 = "a015c229e2a42aad2bdf9c086b619b0dc42d793c4e981157f64fca7bfde45152"

PREDECESSOR_TERMINAL_RECEIPTS = [
    {
        "policy_id": "G9ASYNCPAIR-8",
        "path": "results/gross9_async_pair_train_economics_2026-09-02.json",
        "sha256": "0b822d77415ca70a409d2e7f3c35ebe44cbf481aa7e0d2eb02605646bdb3f874",
        "manifest_hash": "bb3ed8afa1eec6cddf2344515d89736a36314157ad7eeac495c759adadc45b16",
        "decision": "terminal_train_reject_no_substitution",
        "schedule_scope": "same-side36 post-reservation schedules are prior-family exact-duplicate comparators",
    },
    {
        "policy_id": "G9ASYNCHANDOFF-8",
        "path": "results/gross9_async_opposition_handoff_train_economics_2026-09-02.json",
        "sha256": "a2c86ae78940a331f1e0209fa5bbb8bdb374fd2d4438030900ffd3a097b85e64",
        "manifest_hash": "3ad6368a44519359cf7661b7707a9f099ae23fb679c6c2115aafab095a51aa3a",
        "decision": "terminal_train_reject_no_substitution",
        "schedule_scope": "handoff36 post-reservation schedules are prior-family exact-duplicate comparators",
    },
    {
        "policy_id": "G9ASYNC3WAY-8",
        "path": "results/gross9_async_three_way_consensus_train_clock_source_support_2026-09-02.json",
        "sha256": "0b9c9366d0d0d214787e1fdb6f3fad9e604e2dbfad49fedd8f4b84fadbcb5265",
        "manifest_hash": "de32635d2e1853359cfe62ca4ef779442fec0dd29caf680433693f07ce6b6495",
        "decision": "terminal_no_source_supported_triples",
        "schedule_scope": "triple84 post-reservation schedules are prior-family exact-duplicate comparators; triple counts and support scalars are not tuning inputs",
    },
]


PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT = {
    "commit": "1bfddd3c",
    "path": "results/gross9_async_active_veto_train_clock_source_support_2026-09-02.json",
    "sha256": "ce95d6373655ded0daba9d6f5635908106337827fbf1a98c978cf41d8231e6e3",
    "manifest_hash": "88ce540e6ce329e0d9f763c128b2f431949c191772d207b6a1b6b65ee4fb3e6d",
    "builder": {
        "path": "training/build_gross9_async_active_veto_train_clocks.py",
        "sha256": "bf8bfaf41d0ca761a2bc0f2db53de5ad05103fe596b880eb0fd8acbbbc6c90df",
    },
    "preregistration_artifact_with_placeholder_builder_binding": {
        "path": "results/gross9_async_active_opposite_veto_search_preregistration_2026-09-02.json",
        "sha256": "b70dbeea6a6d1bde63ea60c854fcfa09688060bf56c0f5a08f3f21073a5f4cba",
        "manifest_hash": "8cc95042fe2e76c5193f3679f6d1f073e2e97bd0a69b658c57599b9fba06ba28",
        "tracked_at_commit": False,
        "note": "preliminary materialization was made against this preregistration JSON artifact while it still carried a placeholder builder binding; this updated preregistration replaces that placeholder with the bound current builder SHA and does not change family/operator/gates/thresholds/order",
    },
    "untracked_preregistration_code_with_placeholder_builder_binding": {
        "path": "training/preregister_gross9_async_active_veto_search.py",
        "sha256": "f14becdeb93904c581cc89809ec161a692aea6e25b66ad7ae8c718584cc6ec59",
        "tracked_at_commit": False,
    },
    "placeholder_builder_value": "PENDING_G9ASYNCACTIVEVETO_BUILDER_FOLLOWUP",
    "support_count_disclosure": {
        "passed_candidates": 14,
        "used_to_retune_family_operator_gates_thresholds_or_order": False,
    },
}

PRIOR_SOURCE_SUPPORT_ARTIFACTS = [
    {
        "policy_id": "G9ASYNCPAIR-8",
        "path": "results/gross9_async_pair_train_clock_source_support_2026-09-02.json",
        "sha256": "c6d3929f282ba1075c2ebc091e4bc62164b923a038bce94de32884aaf7ff0009",
        "manifest_hash": "b92d3afb7a3539cdd194eddc1ab09bc65068716135d0bca575db0531ac450011",
        "schedule_scope": "same-side36 source-support artifact and schedules bound for exact-duplicate gates and overlap disclosure",
    },
    {
        "policy_id": "G9ASYNCHANDOFF-8",
        "path": "results/gross9_async_opposition_handoff_train_clock_source_support_2026-09-02.json",
        "sha256": "a8982c1b6e155f65f76af4559ca2d01b2a7824cb5c58524a260b72beb997f754",
        "manifest_hash": "92501aa4c921bba20d05378b6f658f33d6c712e8b3adb9f095940dd44ac3f3b0",
        "schedule_scope": "handoff36 source-support artifact and schedules bound for exact-duplicate gates and overlap disclosure",
    },
    {
        "policy_id": "G9ASYNC3WAY-8",
        "path": "results/gross9_async_three_way_consensus_train_clock_source_support_2026-09-02.json",
        "sha256": "0b9c9366d0d0d214787e1fdb6f3fad9e604e2dbfad49fedd8f4b84fadbcb5265",
        "manifest_hash": "de32635d2e1853359cfe62ca4ef779442fec0dd29caf680433693f07ce6b6495",
        "schedule_scope": "triple84 source-support artifact and schedules bound for exact-duplicate gates and overlap disclosure",
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
        "objective": "continue the non-LLM Gross9-like signal-alpha search with an untried ordered asynchronous active opposite-veto operator over the exact same nine frozen components",
        "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False,
        "predecessor_terminal_receipts": copy.deepcopy(PREDECESSOR_TERMINAL_RECEIPTS),
        "preliminary_source_materialization_receipt": copy.deepcopy(PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT),
        "prior_source_support_artifacts": copy.deepcopy(PRIOR_SOURCE_SUPPORT_ARTIFACTS),
        "component_order": list(COMPONENT_ORDER),
        "component_count": len(COMPONENT_ORDER),
        "component_artifacts": copy.deepcopy(COMPONENT_ARTIFACTS),
        "gross9_pre2025_clock_manifest": copy.deepcopy(GROSS9_PRE2025_CLOCK_MANIFEST),
        "implementation": {
            "preregister": {
                "path": "training/preregister_gross9_async_active_veto_search.py",
                "sha256": sha256_file(__file__),
            },
            "train_clock_builder": {
                "path": "training/build_gross9_async_active_veto_train_clocks.py",
                "sha256": BUILDER_SHA256,
            },
        },
        "candidate_family": list(CANDIDATE_FAMILY),
        "candidate_family_size": FAMILY_SIZE,
        "construction": {
            "operator": "asynchronous active opposite veto over ordered base/veto components",
            "ordered_pair_definition": "all 72 ordered A!=B pairs from the same nine frozen train-economic PASS components; outer loop follows frozen component_order for base A and inner loop follows frozen component_order for veto B",
            "candidate_id_format": "A__ASYNC_ACTIVE_OPPOSITE_VETO_6H__B",
            "base_event_rule": "at each base event (t,s) from A, inspect B for its unique latest active veto event v satisfying v.entry_time <= t < v.entry_time + 6 elapsed hours, equivalent to strict lower t-6h < v.entry_time <= t",
            "veto_rule": "if no active B event exists in the window, emit the base event (t,s); if the latest active B event is same-side, emit the base event (t,s); if the latest active B event is opposite-side, cash/no row; never reverse side",
            "latest_supersedes_older": True,
            "same_timestamp_veto_allowed": True,
            "availability": "hard fail source clock availability/duplicate drift; every selected base event and active veto decision must have decision_time and feature_available_time <= t",
            "source_clock_integrity": "hard fail missing source clocks, bound SHA drift, duplicated normalized source events, or component row-count drift before constructing any candidate schedule",
            "duplicate_policy": "deduplicate emitted rows by candidate, entry_time, and side before reservation",
            "reservation": "one chronological half-open 8h reservation inside each ordered candidate clock after dedupe; touching intervals are allowed",
            "normalized_base_controls": "for each of the nine base components, freeze a no-veto normalized 8h base-control clock using the same dedupe/reservation/gross rules; controls are disclosure-only and never ranked or substituted",
            "current_family_exact_duplicate_gate": "after candidate-local reservation, if any nonempty exact post-reservation schedule appears in more than one of the current 72 ordered candidates, reject all members of that duplicate group as source-support failures; empty schedules are not duplicates",
            "prior_schedule_exact_duplicate_gate": "after candidate-local reservation, any exact post-reservation schedule duplicate of any of the nine normalized base controls or any bound prior same-side36, handoff36, or triple84 schedule is a source-support failure; empty schedules are not duplicates",
            "overlap_disclosure": "nonexact overlap against the nine base controls, all other current-72 candidates, and all bound prior same-side36/handoff36/triple84 schedules is disclosure-only, not a gate",
            "grids_or_variants": "none; no lag, hold, side, operator, threshold, component, subset, weight, ordering, control, or overlap-threshold variants",
            "component_formula_threshold_clock_mutability": "immutable",
        },
        "clock": {
            "entry": "surviving base event timestamp t after active opposite-veto filtering",
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
            "opposite_suppressions_min": 1,
            "source_clock_availability_and_duplicate_drift_hard_fail": True,
            "current_family_exact_duplicate_gate": "reject all nonempty exact post-reservation schedule duplicates among the current 72 ordered candidates",
            "normalized_base_control_exact_duplicate_gate": "reject exact post-reservation schedule duplicates of the nine normalized no-veto 8h base controls",
            "prior_family_exact_duplicate_gate": "reject exact post-reservation schedule duplicates of all bound prior same-side36, handoff36, and triple84 schedules",
            "empty_schedule_is_duplicate": False,
            "nonexact_overlap_disclosure_only": True,
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
            "eligibility": "ordered candidate passes source-support including availability/drift, suppression, current/prior/base exact-duplicate gates, and Gross9 novelty gates before any economics",
            "ranking_metric": "descending train base-cost CAGR divided by strict MDD",
            "tie_breaks": ["descending train base-cost absolute return", "ascending frozen candidate_family order"],
            "raw_rank_one_no_substitution": True,
            "winner_train_gate": "raw rank-one ordered candidate must pass every train economic gate; failure terminates the family",
            "later_stages": "after winner freeze, evaluate test/eval/final sequentially; stop on first failure; no rerank or repair",
        },
        "familywise_multiplicity": {
            "family": "all 9*8=72 ordered async active opposite-veto candidates, including candidates later failing source, duplicate, or Gross9 gates",
            "scope_boundary": "controls this fixed 72-hypothesis family only; it does not control the cumulative adaptive exploratory research program across prior or future families",
            "rule": "Bonferroni",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "number_of_hypotheses": FAMILY_SIZE,
            "winner_raw_weekly_signflip_p_max": BONFERRONI_RAW_P_MAX,
            "equivalent_adjusted_p": "min(1, 72 * raw one-sided weekly sign-flip p)",
        },
        "research_boundary": {
            "llm_path_paused": True,
            "predecessor_terminal_receipts_bound": True,
            "design_family_operator_and_gates_fixed_before_preliminary_source_materialization": True,
            "source_incidence_and_support_counts_opened_before_committed_preregistration": True,
            "family_operator_gate_threshold_or_order_changed_after_preliminary_source_materialization": False,
            "preliminary_14_source_passes_used_to_retune": False,
            "gross9_market_funding_or_pnl_opened_by_preregistration": False,
            "triple_source_counts_or_economic_scalars_used_to_tune_operator_or_gates": False,
            "predecessor_economic_scalars_used_to_tune_operator_or_gates": False,
            "predecessor_counts_used_to_change_components": False,
            "component_standalone_train_outcomes_known": True,
            "component_standalone_outcomes_used_to_change_components": False,
            "component_formulas_thresholds_clocks_frozen": True,
            "component_clock_rows_opened_by_preregistration": 0,
            "active_veto_combination_incidence_opened_by_preregistration": False,
            "active_veto_combination_outcomes_opened_by_preregistration": False,
            "market_or_funding_rows_opened_by_preregistration": False,
            "test_eval_final_outcomes_opened_before_winner_freeze": False,
            "classification": "adaptive exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": "Fixed sequence over the frozen 72 ordered active-veto family: source clock integrity, no-veto base control disclosure, active opposite-veto construction, duplicate gates, source support, Gross9 novelty, train-only raw ranking and rank-one train gates, then frozen-winner test/eval/final economics. Stop on first failure with no ordered-pair substitution, formula, threshold, side, clock, hold, subset, cost, control, overlap, or rerank repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    core = dict(value)
    manifest_hash = core.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(core):
        raise RuntimeError("G9ASYNCACTIVEVETO-8 preregistration drift")
    if value.get("protocol_version") != PROTOCOL_VERSION or value.get("policy_id") != POLICY_ID:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 protocol identity drift")
    if tuple(value.get("component_order", ())) != COMPONENT_ORDER:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 component family drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 candidate family drift")
    if value.get("candidate_family_size") != FAMILY_SIZE:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 family size drift")
    if value.get("economic_gates", {}).get("weekly_signflip_one_sided_p_max") != BONFERRONI_RAW_P_MAX:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 Bonferroni drift")
    source = value.get("source_support_gates", {})
    if source.get("minimum_events", {}).get("train") != 10:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 train event source gate drift")
    if source.get("distinct_iso_weeks_min") != 10:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 distinct-week gate drift")
    if source.get("each_calendar_half_min_events") != 1:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 calendar-half source gate drift")
    if source.get("opposite_suppressions_min") != 1:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 suppression gate drift")
    if source.get("source_clock_availability_and_duplicate_drift_hard_fail") is not True:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 source clock integrity gate drift")
    if source.get("empty_schedule_is_duplicate") is not False or source.get("nonexact_overlap_disclosure_only") is not True:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 duplicate/overlap gate drift")
    construction = value.get("construction", {})
    if "strict lower t-6h < v.entry_time <= t" not in construction.get("base_event_rule", ""):
        raise RuntimeError("G9ASYNCACTIVEVETO-8 active-veto window drift")
    if "cash/no row" not in construction.get("veto_rule", "") or "never reverse side" not in construction.get("veto_rule", ""):
        raise RuntimeError("G9ASYNCACTIVEVETO-8 veto semantics drift")
    if "reject all members" not in construction.get("current_family_exact_duplicate_gate", ""):
        raise RuntimeError("G9ASYNCACTIVEVETO-8 current duplicate gate drift")
    if "nine normalized base controls" not in construction.get("prior_schedule_exact_duplicate_gate", ""):
        raise RuntimeError("G9ASYNCACTIVEVETO-8 base-control duplicate gate drift")
    if "disclosure-only" not in construction.get("overlap_disclosure", ""):
        raise RuntimeError("G9ASYNCACTIVEVETO-8 nonexact overlap disclosure drift")
    implementation = value.get("implementation", {})
    if implementation.get("train_clock_builder", {}).get("sha256") != BUILDER_SHA256:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 builder binding drift")
    if value.get("preliminary_source_materialization_receipt") != PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 preliminary source materialization binding drift")
    if value.get("predecessor_terminal_receipts") != PREDECESSOR_TERMINAL_RECEIPTS:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 predecessor terminal binding drift")
    if value.get("prior_source_support_artifacts") != PRIOR_SOURCE_SUPPORT_ARTIFACTS:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 prior source-support binding drift")
    boundary = value.get("research_boundary", {})
    if boundary.get("design_family_operator_and_gates_fixed_before_preliminary_source_materialization") is not True:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 preliminary design-freeze boundary drift")
    if boundary.get("source_incidence_and_support_counts_opened_before_committed_preregistration") is not True:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 preliminary incidence disclosure drift")
    if boundary.get("family_operator_gate_threshold_or_order_changed_after_preliminary_source_materialization") is not False:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 post-materialization retune boundary drift")
    if boundary.get("preliminary_14_source_passes_used_to_retune") is not False:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 preliminary source-pass retune boundary drift")
    if boundary.get("gross9_market_funding_or_pnl_opened_by_preregistration") is not False:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 Gross9/market/PnL boundary drift")
    if boundary.get("triple_source_counts_or_economic_scalars_used_to_tune_operator_or_gates") is not False:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 triple tuning boundary drift")
    if boundary.get("active_veto_combination_incidence_opened_by_preregistration") is not False:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 incidence boundary drift")
    if boundary.get("active_veto_combination_outcomes_opened_by_preregistration") is not False:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 outcome boundary drift")
    if "cumulative adaptive exploratory research program" not in value.get("familywise_multiplicity", {}).get("scope_boundary", ""):
        raise RuntimeError("G9ASYNCACTIVEVETO-8 cumulative-control disclosure drift")


def _check_bool(row: Mapping[str, Any], flag: str) -> bool:
    value = row.get(flag)
    if type(value) is not bool:
        raise ValueError(f"G9ASYNCACTIVEVETO-8 {flag} must be boolean")
    return value


def select_train_winner(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze the raw rank-one row after the future 72-candidate train search is run."""
    if len(rows) != FAMILY_SIZE:
        raise ValueError("G9ASYNCACTIVEVETO-8 selection requires exactly 72 candidate rows")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        candidate = row.get("candidate")
        if not isinstance(candidate, str) or candidate in by_id:
            raise ValueError("G9ASYNCACTIVEVETO-8 candidate IDs must be unique strings")
        by_id[candidate] = row
    if set(by_id) != set(CANDIDATE_FAMILY):
        raise ValueError("G9ASYNCACTIVEVETO-8 selection requires the exact frozen family")

    eligible: list[tuple[int, Mapping[str, Any], float, float]] = []
    for order, candidate in enumerate(CANDIDATE_FAMILY):
        row = by_id[candidate]
        source_pass = _check_bool(row, "source_pass")
        duplicate_pass = _check_bool(row, "exact_duplicate_pass")
        gross9_pass = _check_bool(row, "gross9_pass")
        train_pass = _check_bool(row, "train_economic_pass")
        if not source_pass or not duplicate_pass or not gross9_pass:
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
            raise ValueError("G9ASYNCACTIVEVETO-8 train ranking metrics must be numeric")
        eligible.append((order, row, float(ratio), float(absolute)))
    if not eligible:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 no source/duplicate/Gross9 eligible candidates")
    eligible.sort(key=lambda item: (-item[2], -item[3], item[0]))
    _, winner, ratio, absolute = eligible[0]
    if winner.get("train_economic_pass") is not True:
        raise RuntimeError("G9ASYNCACTIVEVETO-8 raw rank one failed train; no substitution")
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
