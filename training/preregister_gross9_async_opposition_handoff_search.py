"""Outcome-blind preregistration for the Gross9 async opposition handoff search.

This freezes a constructionally disjoint, non-LLM signal-alpha family after the
terminal G9ASYNCPAIR-8 train result.  It binds the same nine immutable
train-economic PASS components plus the completed same-side family terminal
artifact, but opens no new handoff-family incidence or outcome rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import copy
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import preregister_gross9_async_pair_search as same_side


POLICY_ID = "G9ASYNCHANDOFF-8"
PROTOCOL_VERSION = "gross9_async_opposition_handoff_search_preregistration_v1"
AS_OF_DATE = "2026-09-02"
DEFAULT_OUTPUT = Path("results/gross9_async_opposition_handoff_search_preregistration_2026-09-02.json")
FAMILYWISE_ALPHA = 0.10
COMPONENT_ORDER = same_side.COMPONENT_ORDER
COMPONENT_ARTIFACTS = same_side.COMPONENT_ARTIFACTS
GROSS9_PRE2025_CLOCK_MANIFEST = same_side.GROSS9_PRE2025_CLOCK_MANIFEST
CANDIDATE_FAMILY = tuple(
    f"{left}__ASYNC_OPPOSITION_HANDOFF_6H__{right}"
    for left, right in combinations(COMPONENT_ORDER, 2)
)
FAMILY_SIZE = 36
BONFERRONI_RAW_P_MAX = FAMILYWISE_ALPHA / FAMILY_SIZE
TRAIN_WINDOW = same_side.TRAIN_WINDOW
SAME_SIDE_TERMINAL_TRAIN_ECONOMICS = {
    "policy_id": "G9ASYNCPAIR-8",
    "path": "results/gross9_async_pair_train_economics_2026-09-02.json",
    "sha256": "0b822d77415ca70a409d2e7f3c35ebe44cbf481aa7e0d2eb02605646bdb3f874",
    "manifest_hash": "bb3ed8afa1eec6cddf2344515d89736a36314157ad7eeac495c759adadc45b16",
    "decision": "terminal_train_reject_no_substitution",
    "raw_rank_one_candidate": "HVDIMIO-8__ASYNC_SAME_SIDE_6H__HVLVR-8",
    "selection_error": "G9ASYNCPAIR-8 raw rank one failed train; no substitution",
}


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
        "objective": "continue non-LLM Gross9-like signal-alpha search with an untried asynchronous opposition handoff operator over the exact same nine frozen components",
        "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False,
        "predecessor_family_terminal_result": copy.deepcopy(SAME_SIDE_TERMINAL_TRAIN_ECONOMICS),
        "component_order": list(COMPONENT_ORDER),
        "component_count": len(COMPONENT_ORDER),
        "component_artifacts": copy.deepcopy(COMPONENT_ARTIFACTS),
        "gross9_pre2025_clock_manifest": copy.deepcopy(GROSS9_PRE2025_CLOCK_MANIFEST),
        "implementation": {
            "preregister": {
                "path": "training/preregister_gross9_async_opposition_handoff_search.py",
                "sha256": sha256_file(__file__),
            },
            "train_clock_builder": {
                "path": "training/build_gross9_async_opposition_handoff_train_clocks.py",
                "sha256": sha256_file(Path(__file__).with_name("build_gross9_async_opposition_handoff_train_clocks.py")),
            },
        },
        "candidate_family": list(CANDIDATE_FAMILY),
        "candidate_family_size": FAMILY_SIZE,
        "construction": {
            "operator": "asynchronous opposition handoff two-component confirmation",
            "pair_definition": "all 36 unordered pairs from the same nine frozen train-economic PASS components used by G9ASYNCPAIR-8",
            "entry_rule": "for a pair (A,B), accept a later trigger event at t only when exactly one/unique trigger pair component has an event at t, the other component has at least one strictly prior opposite-side event in [t-6h,t), and the other component has zero same-side events in [t-6h,t)",
            "lookback_window": "strictly prior elapsed-time window [t-6h,t) == [later entry minus 6 elapsed hours, later entry); same-timestamp confirmation is forbidden",
            "confirming_event": "the latest opposite-side event from the other component inside the strict lookback window",
            "veto": "any same-side event from the other component inside [t-6h,t) rejects the trigger",
            "output_entry_time": "newer trigger component entry_time t",
            "output_side": "newer trigger component side, opposite to the latest confirming component side",
            "availability": "max(trigger decision_time, trigger feature_available_time, confirming decision_time, confirming feature_available_time) must be <= t",
            "duplicate_policy": "deduplicate constructed rows by candidate, entry_time, and side before reservation",
            "reservation": "one chronological half-open reservation inside each candidate-pair clock after dedupe; touching intervals allowed",
            "same_side_family_disjointness_invariant": "before handoff reservation, the set of (unordered pair, entry_time, side) handoff candidates must have exact zero intersection with the pre-reservation G9ASYNCPAIR-8 same-side family construction",
            "grids_or_variants": "none; no lag, hold, side, operator, threshold, component, subset, weight, or ordering variants",
            "component_formula_threshold_clock_mutability": "immutable",
        },
        "clock": {
            "entry": "later opposition-handoff trigger event",
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
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.20,
            "max_month_share": 0.45,
            "distinct_iso_weeks_min": 9,
            "statistical_feasibility_gate": "distinct ISO weeks in train must be at least 9 before any train economics",
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
            "eligibility": "pair passes source-support, disjointness, and Gross9 novelty gates before any economics",
            "ranking_metric": "descending train base-cost CAGR divided by strict MDD",
            "tie_breaks": ["descending train base-cost absolute return", "ascending frozen candidate_family order"],
            "raw_rank_one_no_substitution": True,
            "winner_train_gate": "raw rank-one pair must pass every train economic gate; failure terminates the family",
            "later_stages": "after winner freeze, evaluate test/eval/final sequentially; stop on first failure; no rerank or repair",
        },
        "familywise_multiplicity": {
            "family": "all C(9,2)=36 async opposition handoff pairs, including candidates later failing source, disjointness, or Gross9 gates",
            "scope_boundary": "controls this fixed 36-hypothesis family only; it does not control the cumulative adaptive research program across prior and future families",
            "rule": "Bonferroni",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "number_of_hypotheses": FAMILY_SIZE,
            "winner_raw_weekly_signflip_p_max": BONFERRONI_RAW_P_MAX,
            "equivalent_adjusted_p": "min(1, 36 * raw one-sided weekly sign-flip p)",
        },
        "research_boundary": {
            "llm_path_paused": True,
            "same_side_terminal_artifact_bound": True,
            "same_side_terminal_train_outcomes_used_to_choose_or_tune_operator_gates": False,
            "same_side_terminal_outcomes_used_to_change_components": False,
            "component_standalone_train_outcomes_known": True,
            "component_standalone_outcomes_used_to_change_components": False,
            "component_formulas_thresholds_clocks_frozen": True,
            "component_clock_rows_opened_by_preregistration": 0,
            "design_fixed_before_scratch_source_feasibility_check": True,
            "source_pair_incidence_and_support_counts_opened_before_persistent_preregistration_artifact": True,
            "handoff_pair_combination_incidence_opened_before_artifact": True,
            "handoff_pair_combination_incidence_opened_by_this_preregistration_script": False,
            "handoff_pair_combination_outcomes_opened_by_preregistration": False,
            "test_eval_final_outcomes_opened_before_winner_freeze": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": "Fixed sequence over the frozen 36-pair handoff family, acknowledging pre-artifact source incidence/support scratch after design freeze: source support including zero same-side pre-reservation intersection and distinct-week feasibility, Gross9 novelty, train-only raw ranking and rank-one train gates, then frozen-winner test/eval/final economics. Stop on first failure with no pair substitution, formula, threshold, side, clock, hold, subset, cost, control, or rerank repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    core = dict(value)
    manifest_hash = core.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(core):
        raise RuntimeError("G9ASYNCHANDOFF-8 preregistration drift")
    if value.get("protocol_version") != PROTOCOL_VERSION or value.get("policy_id") != POLICY_ID:
        raise RuntimeError("G9ASYNCHANDOFF-8 protocol identity drift")
    if tuple(value.get("component_order", ())) != COMPONENT_ORDER:
        raise RuntimeError("G9ASYNCHANDOFF-8 component family drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("G9ASYNCHANDOFF-8 candidate family drift")
    if value.get("candidate_family_size") != FAMILY_SIZE:
        raise RuntimeError("G9ASYNCHANDOFF-8 family size drift")
    if value.get("economic_gates", {}).get("weekly_signflip_one_sided_p_max") != BONFERRONI_RAW_P_MAX:
        raise RuntimeError("G9ASYNCHANDOFF-8 Bonferroni drift")
    if value.get("source_support_gates", {}).get("distinct_iso_weeks_min") != 9:
        raise RuntimeError("G9ASYNCHANDOFF-8 distinct-week gate drift")
    if "exact zero intersection" not in value.get("construction", {}).get("same_side_family_disjointness_invariant", ""):
        raise RuntimeError("G9ASYNCHANDOFF-8 disjointness invariant drift")
    boundary = value.get("research_boundary", {})
    if boundary.get("source_pair_incidence_and_support_counts_opened_before_persistent_preregistration_artifact") is not True:
        raise RuntimeError("G9ASYNCHANDOFF-8 pre-artifact incidence disclosure drift")
    if boundary.get("handoff_pair_combination_incidence_opened_by_this_preregistration_script") is not False:
        raise RuntimeError("G9ASYNCHANDOFF-8 incidence boundary drift")
    if boundary.get("handoff_pair_combination_outcomes_opened_by_preregistration") is not False:
        raise RuntimeError("G9ASYNCHANDOFF-8 outcome boundary drift")
    predecessor = value.get("predecessor_family_terminal_result", {})
    if predecessor != SAME_SIDE_TERMINAL_TRAIN_ECONOMICS:
        raise RuntimeError("G9ASYNCHANDOFF-8 predecessor terminal binding drift")


def _check_bool(row: Mapping[str, Any], flag: str) -> bool:
    value = row.get(flag)
    if type(value) is not bool:
        raise ValueError(f"G9ASYNCHANDOFF-8 {flag} must be boolean")
    return value


def select_train_winner(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze the raw rank-one row after the future train search is run."""
    if len(rows) != FAMILY_SIZE:
        raise ValueError("G9ASYNCHANDOFF-8 selection requires exactly 36 candidate rows")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        candidate = row.get("candidate")
        if not isinstance(candidate, str) or candidate in by_id:
            raise ValueError("G9ASYNCHANDOFF-8 candidate IDs must be unique strings")
        by_id[candidate] = row
    if set(by_id) != set(CANDIDATE_FAMILY):
        raise ValueError("G9ASYNCHANDOFF-8 selection requires the exact frozen family")

    eligible: list[tuple[int, Mapping[str, Any], float, float]] = []
    for order, candidate in enumerate(CANDIDATE_FAMILY):
        row = by_id[candidate]
        source_pass = _check_bool(row, "source_pass")
        disjoint_pass = _check_bool(row, "same_side_disjointness_pass")
        gross9_pass = _check_bool(row, "gross9_pass")
        train_pass = _check_bool(row, "train_economic_pass")
        if not source_pass or not disjoint_pass or not gross9_pass:
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
            raise ValueError("G9ASYNCHANDOFF-8 train ranking metrics must be numeric")
        eligible.append((order, row, float(ratio), float(absolute)))
    if not eligible:
        raise RuntimeError("G9ASYNCHANDOFF-8 no source/disjoint/Gross9 eligible pairs")
    eligible.sort(key=lambda item: (-item[2], -item[3], item[0]))
    _, winner, ratio, absolute = eligible[0]
    if winner.get("train_economic_pass") is not True:
        raise RuntimeError("G9ASYNCHANDOFF-8 raw rank one failed train; no substitution")
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
