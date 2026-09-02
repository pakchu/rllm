"""Preregister the train-only Gross9 QTR active-veto distillation candidate.

This freezes one adaptive shadow candidate, G9QTR-DISTILL-8, distilled only
from the terminal 2026-09-02 active-veto train artifact.  It does not open or
score any 2024/2025/2026 out-of-sample rows; later OOS evaluation must be
sequential and no-repair.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import preregister_gross9_async_active_veto_search as active_veto


POLICY_ID = "G9QTR-DISTILL-8"
PROTOCOL_VERSION = "gross9_qtr_distill_shadow_preregistration_v1"
AS_OF_DATE = "2026-09-02"
DEFAULT_OUTPUT = Path("results/gross9_qtr_distill_shadow_preregistration_2026-09-02.json")
ACTIVE_VETO_OPERATOR = "ASYNC_ACTIVE_OPPOSITE_VETO_6H"
DISTILLATION_VETO = "HVCQTR-24"
DISTILLED_BASES = ("HVDEMWMV-24", "HVCPF17-8", "HVDIMIO-8", "HVLVR-8")
DISTILLED_SLEEVES = tuple(f"{base}__{ACTIVE_VETO_OPERATOR}__{DISTILLATION_VETO}" for base in DISTILLED_BASES)
# Fixed gross-exposure weights.  They sum to 0.5 and deliberately do not renormalize.
SLEEVE_WEIGHTS = {
    "HVDEMWMV-24__ASYNC_ACTIVE_OPPOSITE_VETO_6H__HVCQTR-24": 1.0 / 6.0,
    "HVCPF17-8__ASYNC_ACTIVE_OPPOSITE_VETO_6H__HVCQTR-24": 1.0 / 6.0,
    "HVDIMIO-8__ASYNC_ACTIVE_OPPOSITE_VETO_6H__HVCQTR-24": 1.0 / 12.0,
    "HVLVR-8__ASYNC_ACTIVE_OPPOSITE_VETO_6H__HVCQTR-24": 1.0 / 12.0,
}
LEGACY_ACTIVE_VETO_FAMILY_SIZE = 72
LEGACY_ACTIVE_VETO_FAMILYWISE_ALPHA = 0.10
LEGACY_BONFERRONI_P_MAX = LEGACY_ACTIVE_VETO_FAMILYWISE_ALPHA / LEGACY_ACTIVE_VETO_FAMILY_SIZE
SINGLE_HYPOTHESIS_OOS_P_MAX = 0.10
TRAIN_WINDOW = active_veto.TRAIN_WINDOW

ACTIVE_VETO_TERMINAL_ARTIFACTS = {
    "preregistration": {
        "path": "results/gross9_async_active_opposite_veto_search_preregistration_2026-09-02.json",
        "sha256": "5bb0abae46a5716451b07268a268cdd112a78829786772c4aeec8bc43f383f25",
        "manifest_hash": "871c7fb8c8825cb30c0967cab46a2a8cc7342f46f37c673372b45d2501d6aa6e",
    },
    "source_support": {
        "path": "results/gross9_async_active_veto_train_clock_source_support_2026-09-02.json",
        "sha256": "ee966e59e219886b561a23e605cf225f44d393128f210a360048addfeba42f20",
        "manifest_hash": "ec32caa65a0945fc73b6d863cb1b3fa810f4c58ffd8aed68408fff949e4d6f32",
    },
    "gross9_novelty": {
        "path": "results/gross9_async_active_veto_train_gross9_novelty_2026-09-02.json",
        "sha256": "64a8cbd12edc04ebb02d30649687c3319a1d06558005ddd8c73ab22f81d884cf",
        "manifest_hash": "27cd576b8175556ff23e0ee87ea1e99092dcaa41cdb8b78fcf6cb202e845c40f",
    },
    "train_economics": {
        "path": "results/gross9_async_active_veto_train_economics_2026-09-02.json",
        "sha256": "c1dc920c7a8ca02525bd9839c09e1add90a721d0e3c167b2ff3fabcc53593cb7",
        "manifest_hash": "e54b40077fa4a8dec97507f6dab829b4cf838d2c3c00e9d4c7e5345e5bfd4043",
        "decision": "terminal_train_reject_no_substitution",
    },
}

DISTILLED_TRAIN_DIAGNOSTICS = {
    "HVDEMWMV-24__ASYNC_ACTIVE_OPPOSITE_VETO_6H__HVCQTR-24": {
        "return_pct": 7.922399595610163,
        "cagr_to_strict_mdd": 9.177313615358774,
        "strict_mdd_pct": 1.7804556318606557,
        "stress_return_pct": 7.10726895371443,
        "stress_cagr_to_strict_mdd": 7.169579671870408,
        "weekly_signflip_p": 0.022299777002229976,
        "train_rows": 19,
    },
    "HVCPF17-8__ASYNC_ACTIVE_OPPOSITE_VETO_6H__HVCQTR-24": {
        "return_pct": 12.626810740720806,
        "cagr_to_strict_mdd": 4.82711650645433,
        "strict_mdd_pct": 5.515249783123622,
        "stress_return_pct": 9.651609650616887,
        "stress_cagr_to_strict_mdd": 3.6255423261258537,
        "weekly_signflip_p": 0.015579844201557985,
        "train_rows": 67,
    },
    "HVDIMIO-8__ASYNC_ACTIVE_OPPOSITE_VETO_6H__HVCQTR-24": {
        "return_pct": 8.652754644461357,
        "cagr_to_strict_mdd": 3.465537607827908,
        "strict_mdd_pct": 5.167417556721587,
        "stress_return_pct": 7.272340954859979,
        "stress_cagr_to_strict_mdd": 2.8826352566936677,
        "weekly_signflip_p": 0.04009959900400996,
        "train_rows": 32,
    },
    "HVLVR-8__ASYNC_ACTIVE_OPPOSITE_VETO_6H__HVCQTR-24": {
        "return_pct": 8.087189468215229,
        "cagr_to_strict_mdd": 3.2303747535707132,
        "strict_mdd_pct": 5.167417556721587,
        "stress_return_pct": 6.713737605476555,
        "stress_cagr_to_strict_mdd": 2.654150327348563,
        "weekly_signflip_p": 0.048799512004879954,
        "train_rows": 32,
    },
}


IMPLEMENTATION_BINDINGS = {
    "portfolio_builder": {
        "path": "training/build_gross9_qtr_distill_clocks.py",
        "sha256": "48fa4427c07a81843645a1a1ad0216f4cf35cd0d4cf0c5b5f4d174f2871032a8",
    },
    "gross9_novelty_evaluator": {
        "path": "training/evaluate_gross9_qtr_distill_novelty.py",
        "sha256": "aa16dc39b3ec03546d4fd9b071718ea78f7098089ae2a3f03d777ece3e2c22c5",
    },
    "economics_evaluator": {
        "path": "training/evaluate_gross9_qtr_distill_economics.py",
        "sha256": "8bd5aa7b2d62ab353d307705131694ce3f365b7590aed2a8be24e4958c3d69cf",
    },
}

PRELIMINARY_SEQUENCING_RECEIPT = {
    "commit": "cbb5f8bc",
    "event": "preliminary_train_artifact_before_later_prereg_source_commit",
    "train_artifact": {
        "path": "results/gross9_qtr_distill_train_economics_2026-09-02.json",
        "sha256": "2a09706548198f5756325b1d672f8a3d4d6664e6e2a83d077a385231f690cae7",
    },
    "preliminary_evaluator": {
        "path": "training/evaluate_gross9_qtr_distill_economics.py",
        "sha256": "d9b2f346e300d9cf2ca52085a9ea81a3412f048b7b6ed6f689ab62fe565a298d",
    },
    "prereg_and_source_status_at_preliminary_train": "untracked_then_committed_later_at_be957b81",
    "later_commit": "be957b81",
    "train_values_used_to_change_formula_weights_or_gates": False,
    "oos_outcomes_opened": False,
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


def _bool(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be boolean")
    return value


def _shape_pass_except_familywise_weekly_p(row: Mapping[str, Any]) -> bool:
    if not (_bool(row, "source_pass") and _bool(row, "exact_duplicate_pass") and _bool(row, "gross9_pass")):
        return False
    checks = row.get("checks")
    if not isinstance(checks, Mapping):
        return False
    required_true = (
        "absolute_return_positive",
        "cagr_to_strict_mdd_min_3",
        "strict_mdd_max_15",
        "mean_gross_move_min_20bp",
        "stress_absolute_return_positive",
        "stress_cagr_to_strict_mdd_min_2_5",
        "each_calendar_half_positive",
    )
    return all(checks.get(key) is True for key in required_true) and checks.get("cluster_signflip_p_max_bonferroni_0_1_over_72") is False


def select_distilled_sleeves(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Reproduce the train-only deterministic distillation choice.

    Eligibility is the terminal active-veto candidate set that passed source,
    duplicate and Gross9 novelty and all economic shape checks except the legacy
    72-family weekly sign-flip Bonferroni gate.  Group by veto clock, choose the
    veto with the most distinct eligible bases, and break ties by frozen
    component_order.
    """
    if set(rows) != set(active_veto.CANDIDATE_FAMILY):
        raise ValueError("active-veto family drift")
    grouped: dict[str, list[str]] = {component: [] for component in active_veto.COMPONENT_ORDER}
    for candidate in active_veto.CANDIDATE_FAMILY:
        row = rows[candidate]
        if not _shape_pass_except_familywise_weekly_p(row):
            continue
        base, veto = candidate.split(f"__{ACTIVE_VETO_OPERATOR}__")
        grouped[veto].append(base)
    order = {component: idx for idx, component in enumerate(active_veto.COMPONENT_ORDER)}
    winner_veto = min(grouped, key=lambda veto: (-len(set(grouped[veto])), order[veto]))
    selected = tuple(base for base in DISTILLED_BASES if base in set(grouped[winner_veto]))
    if winner_veto != DISTILLATION_VETO or selected != DISTILLED_BASES:
        raise RuntimeError("G9QTR-DISTILL-8 train-only selection proof drift")
    return {
        "winner_veto": winner_veto,
        "eligible_base_count_by_veto": {veto: len(set(bases)) for veto, bases in grouped.items() if bases},
        "selected_sleeves": list(DISTILLED_SLEEVES),
        "selected_bases": list(selected),
        "frozen_component_order_tie_break": list(active_veto.COMPONENT_ORDER),
    }


def build() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "objective": "distill the best train-only Gross9 active-veto shadow evidence into one fixed low-gross portfolio candidate for sequential OOS validation",
        "research_status": "adaptive_exploratory_shadow_until_all_oos_stages_pass",
        "train_classification": "post_selection_train_shape_shadow",
        "fresh_confirmatory_evidence": False,
        "llm_path_paused": True,
        "preliminary_sequencing_receipt": copy.deepcopy(PRELIMINARY_SEQUENCING_RECEIPT),
        "active_veto_terminal_artifacts": copy.deepcopy(ACTIVE_VETO_TERMINAL_ARTIFACTS),
        "component_order": list(active_veto.COMPONENT_ORDER),
        "gross9_pre2025_clock_manifest": copy.deepcopy(active_veto.GROSS9_PRE2025_CLOCK_MANIFEST),
        "selection_rule": {
            "input_scope": "terminal active-veto train artifact only; no 2024/2025/2026 OOS outcomes opened",
            "eligibility": "source_pass and exact_duplicate_pass and Gross9 novelty pass and every train economic shape gate except the legacy familywise weekly sign-flip p gate",
            "grouping": "group eligible ordered active-veto candidates by veto component",
            "primary_choice": "choose veto with maximum distinct eligible bases",
            "tie_break": "frozen component_order",
            "winner_veto": DISTILLATION_VETO,
            "selected_bases": list(DISTILLED_BASES),
            "selected_sleeves": list(DISTILLED_SLEEVES),
            "substitution_authorized": False,
            "rerank_authorized": False,
            "repair_authorized": False,
        },
        "distilled_train_diagnostics": copy.deepcopy(DISTILLED_TRAIN_DIAGNOSTICS),
        "legacy_multiplicity_disclosure": {
            "active_veto_family_hypotheses": LEGACY_ACTIVE_VETO_FAMILY_SIZE,
            "familywise_alpha": LEGACY_ACTIVE_VETO_FAMILYWISE_ALPHA,
            "legacy_raw_weekly_p_threshold": LEGACY_BONFERRONI_P_MAX,
            "all_selected_sleeves_failed_legacy_familywise_weekly_p": True,
            "legacy_p_non_authorizing": True,
            "not_relabelled_train_pass": True,
        },
        "gross9_novelty_scope": {
            "classification": "train_only_structural_prerequisite_persists_for_oos",
            "discovery_input_active_veto_artifact": copy.deepcopy(
                ACTIVE_VETO_TERMINAL_ARTIFACTS["gross9_novelty"]
            ),
            "gates": copy.deepcopy(active_veto.build()["gross9_novelty_gates"]),
            "required_pass_before_any_economics": True,
            "pre2025_comparator_unavailable_for_eval_final_retest": True,
            "oos_novelty_retest_required": False,
            "oos_economic_gates_do_not_include_gross9_novelty": True,
        },
        "portfolio_construction": {
            "candidate_id": POLICY_ID,
            "component_sleeves": list(DISTILLED_SLEEVES),
            "sleeve_weights": copy.deepcopy(SLEEVE_WEIGHTS),
            "gross_exposure_sum": sum(SLEEVE_WEIGHTS.values()),
            "per_sleeve_rule": "exact active-veto strict-lower6h candidate schedule with 8 elapsed hour hold",
            "accounting": "fixed-quantity overlapping sleeves; exit-before-entry atomic transitions; same direction sums; opposite direction nets; costs and funding on aggregate net quantity change/position",
            "no_renormalization": True,
            "no_vol_target": True,
            "no_weight_retune": True,
            "known_redundancy_disclosure": {
                "pair": [
                    "HVDIMIO-8__ASYNC_ACTIVE_OPPOSITE_VETO_6H__HVCQTR-24",
                    "HVLVR-8__ASYNC_ACTIVE_OPPOSITE_VETO_6H__HVCQTR-24",
                ],
                "train_entry_jaccard": 0.939393,
                "mitigation": "half-sized sleeves 1/12 each; disclosed, not optimized",
            },
        },
        "stages": {
            "train_diagnostic": list(TRAIN_WINDOW),
            "test2024": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval2025": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final2026": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "oos_gate_rule": {
            "sequence": ["test2024", "eval2025", "final2026"],
            "single_hypothesis_weekly_signflip_one_sided_p_max": SINGLE_HYPOTHESIS_OOS_P_MAX,
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "minimum_nonzero_signed_episodes": {"test2024": 12, "eval2025": 12, "final2026": 8},
            "source_min_nonzero_signed_episodes": {"test2024": 12, "eval2025": 12, "final2026": 8},
            "gross9_train_structural_prerequisite_persists": True,
            "gross9_novelty_retest_in_oos": False,
            "stop_on_first_failure": True,
            "repair_authorized_after_failure": False,
        },
        "implementation": {
            "preregister": {
                "path": "training/preregister_gross9_qtr_distill.py",
                "sha256": sha256_file(__file__),
            },
            **copy.deepcopy(IMPLEMENTATION_BINDINGS),
        },
        "evidence_boundary": {
            "oos_outcomes_opened_by_this_preregistration": False,
            "oos_exact_rule_left_untouched": True,
            "train_legacy_p_not_relabeled_as_pass": True,
            "adaptive_exploratory_until_all_oos_pass": True,
        },
    }
    core["manifest_hash"] = canonical_hash(core)
    return core


def validate(value: Mapping[str, Any]) -> None:
    core = dict(value)
    manifest_hash = core.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(core):
        raise RuntimeError("G9QTR-DISTILL-8 preregistration drift")
    if value.get("protocol_version") != PROTOCOL_VERSION or value.get("policy_id") != POLICY_ID:
        raise RuntimeError("G9QTR-DISTILL-8 identity drift")
    if tuple(value.get("component_order", ())) != active_veto.COMPONENT_ORDER:
        raise RuntimeError("G9QTR-DISTILL-8 component-order drift")
    selection = value.get("selection_rule", {})
    if selection.get("winner_veto") != DISTILLATION_VETO or tuple(selection.get("selected_bases", ())) != DISTILLED_BASES:
        raise RuntimeError("G9QTR-DISTILL-8 selection drift")
    if tuple(selection.get("selected_sleeves", ())) != DISTILLED_SLEEVES:
        raise RuntimeError("G9QTR-DISTILL-8 sleeve drift")
    construction = value.get("portfolio_construction", {})
    weights = construction.get("sleeve_weights", {})
    if weights != SLEEVE_WEIGHTS or not math.isclose(float(construction.get("gross_exposure_sum", -1)), 0.5):
        raise RuntimeError("G9QTR-DISTILL-8 weight drift")
    if construction.get("no_renormalization") is not True or construction.get("no_vol_target") is not True:
        raise RuntimeError("G9QTR-DISTILL-8 portfolio normalization drift")
    if "opposite direction nets" not in construction.get("accounting", ""):
        raise RuntimeError("G9QTR-DISTILL-8 netting semantics drift")
    if value.get("train_classification") != "post_selection_train_shape_shadow":
        raise RuntimeError("G9QTR-DISTILL-8 train classification drift")
    legacy = value.get("legacy_multiplicity_disclosure", {})
    if legacy.get("not_relabelled_train_pass") is not True or legacy.get("legacy_p_non_authorizing") is not True:
        raise RuntimeError("G9QTR-DISTILL-8 legacy p disclosure drift")
    if value.get("preliminary_sequencing_receipt") != PRELIMINARY_SEQUENCING_RECEIPT:
        raise RuntimeError("G9QTR-DISTILL-8 preliminary sequencing receipt drift")
    novelty_scope = value.get("gross9_novelty_scope", {})
    if (
        novelty_scope.get("classification") != "train_only_structural_prerequisite_persists_for_oos"
        or novelty_scope.get("gates") != active_veto.build()["gross9_novelty_gates"]
        or novelty_scope.get("required_pass_before_any_economics") is not True
        or novelty_scope.get("pre2025_comparator_unavailable_for_eval_final_retest") is not True
        or novelty_scope.get("oos_novelty_retest_required") is not False
        or novelty_scope.get("oos_economic_gates_do_not_include_gross9_novelty") is not True
    ):
        raise RuntimeError("G9QTR-DISTILL-8 Gross9 novelty scope drift")
    gates = value.get("oos_gate_rule", {})
    if gates.get("sequence") != ["test2024", "eval2025", "final2026"]:
        raise RuntimeError("G9QTR-DISTILL-8 OOS sequence drift")
    if gates.get("single_hypothesis_weekly_signflip_one_sided_p_max") != SINGLE_HYPOTHESIS_OOS_P_MAX:
        raise RuntimeError("G9QTR-DISTILL-8 OOS p gate drift")
    if gates.get("stop_on_first_failure") is not True or gates.get("repair_authorized_after_failure") is not False:
        raise RuntimeError("G9QTR-DISTILL-8 no-repair drift")
    if gates.get("source_min_nonzero_signed_episodes") != {"test2024": 12, "eval2025": 12, "final2026": 8}:
        raise RuntimeError("G9QTR-DISTILL-8 source episode gate drift")
    if "gross9_novelty_gates" in gates or gates.get("gross9_novelty_retest_in_oos") is not False:
        raise RuntimeError("G9QTR-DISTILL-8 OOS novelty scope drift")
    if value.get("active_veto_terminal_artifacts") != ACTIVE_VETO_TERMINAL_ARTIFACTS:
        raise RuntimeError("G9QTR-DISTILL-8 terminal artifact binding drift")
    implementation = value.get("implementation", {})
    for name, binding in IMPLEMENTATION_BINDINGS.items():
        observed = implementation.get(name)
        if observed != binding:
            raise RuntimeError(f"G9QTR-DISTILL-8 {name} binding drift")
        sha = observed.get("sha256") if isinstance(observed, Mapping) else None
        path = observed.get("path") if isinstance(observed, Mapping) else None
        if not isinstance(sha, str) or len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            raise RuntimeError(f"G9QTR-DISTILL-8 {name} binding is not a real sha256")
        if not isinstance(path, str) or sha256_file(path) != sha:
            raise RuntimeError(f"G9QTR-DISTILL-8 {name} current file hash mismatch")
    boundary = value.get("evidence_boundary", {})
    if boundary.get("oos_outcomes_opened_by_this_preregistration") is not False:
        raise RuntimeError("G9QTR-DISTILL-8 OOS boundary drift")


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
