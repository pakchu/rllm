"""Outcome-blind preregistration for the fixed HVCSPR-8 priority-router battery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_state_ordered_filter as hvsof

POLICY_ID = "HVCSPR-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_cross_structure_priority_router_preregistration_2026-08-14.json"
)
ACTION_ORDER = ("CARSC-8", "HVCMMI-8", "HVIABR-8")
PRIORITY_ORDERS = (
    ("CARSC-8", "HVCMMI-8", "HVIABR-8"),
    ("HVIABR-8", "HVCMMI-8", "CARSC-8"),
)
ELIGIBILITY_ORDER = ("HVTCCR-8", "HVLZC-8")


def candidate_id(priority_order: Sequence[str], eligibility: str) -> str:
    return f"{'__THEN__'.join(priority_order)}__ELIGIBLE_BY__{eligibility}"


CANDIDATE_FAMILY = tuple(
    candidate_id(priority_order, eligibility)
    for priority_order in PRIORITY_ORDERS
    for eligibility in ELIGIBILITY_ORDER
)
FAMILY_SIZE = 4
FAMILYWISE_ALPHA = 0.10
BONFERRONI_RAW_P_MAX = 0.025

# Pin each cross-structure action to its frozen standalone contract and primary clock.
ACTION_ARTIFACTS: dict[str, dict[str, dict[str, str]]] = {
    "CARSC-8": {
        "preregistration": {
            "path": "results/cross_alt_return_synchrony_continuation_relay_preregistration_2026-08-10.json",
            "sha256": "fa9ed9a30e8cbd0532b690acd585d371c91c91a725e7c1b8f4f3187544d0c8e4",
        },
        "support": {
            "path": "results/cross_alt_return_synchrony_continuation_relay_support_2026-08-10.json",
            "sha256": "4302f99eae2772618c94af86af1f951efae1e7b0148c8c6432f2368e802b78a1",
        },
        "gross9": {
            "path": "results/cross_alt_return_synchrony_continuation_relay_gross9_novelty_2026-08-10.json",
            "sha256": "847be3fa882f345b487aa0fac363f7da2681b436df44294bbdc3b8ea54448d62",
        },
        "clock": {
            "path": "data/cross_alt_return_synchrony_continuation_relay_clocks_2023_2026.csv.gz",
            "sha256": "871d7201a04ae938294948d321abb23e99e7b745ddccc1149491940562b69e52",
        },
    },
    "HVCMMI-8": {
        "preregistration": {
            "path": "results/high_volatility_crypto_market_mode_ignition_relay_preregistration_2026-08-10.json",
            "sha256": "78c1a90c8aef0d67cb00f1951afb8fb1b49c427632b490b9cd37aa4dfc10e9e2",
        },
        "support": {
            "path": "results/high_volatility_crypto_market_mode_ignition_relay_support_2026-08-10.json",
            "sha256": "ca2e22d1dd96d90a015fdb3024b64571edeb94e38d0aed02ae91bb28359b382c",
        },
        "gross9": {
            "path": "results/high_volatility_crypto_market_mode_ignition_relay_gross9_novelty_2026-08-10.json",
            "sha256": "d00a81c1a4021bad945d9498e449819e04e43505ef44b83688f57d6a8ddbebfc",
        },
        "clock": {
            "path": "data/high_volatility_crypto_market_mode_ignition_relay_clocks_2023_2026.csv.gz",
            "sha256": "faa4f8c09b107c6846f0d8143aa740d9f5e665c4c628696af7fc535f572c3aab",
        },
    },
    "HVIABR-8": {
        "preregistration": {
            "path": "results/high_volatility_intrabar_acceptance_breadth_relay_preregistration_2026-08-10.json",
            "sha256": "619674be60397f0004ad79b089c8599db59553fbdbd3499cdbc9e3fca502c7ef",
        },
        "support": {
            "path": "results/high_volatility_intrabar_acceptance_breadth_relay_support_2026-08-10.json",
            "sha256": "624a8f7b9a4fa0e253e91a5c21f4bdc1ac80ba99eb9675836e26dc3dc2e4e151",
        },
        "gross9": {
            "path": "results/high_volatility_intrabar_acceptance_breadth_relay_gross9_novelty_2026-08-10.json",
            "sha256": "0967f5849c3c86403cc31c42f484a281d59a6f9751b77fe039629ac4c1241da0",
        },
        "clock": {
            "path": "data/high_volatility_intrabar_acceptance_breadth_relay_clocks_2023_2026.csv.gz",
            "sha256": "31441774050939b8a2695fa20d2097b0eb246830fb02dc225cb38033216b6711",
        },
    },
}

# Eligibility contracts and source-state artifacts are reused exactly from HVSOF-8.
ELIGIBILITY_ARTIFACTS = hvsof.ELIGIBILITY_ARTIFACTS


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict[str, Any]:
    candidates = [
        {
            "candidate": candidate_id(priority_order, eligibility),
            "priority_order": list(priority_order),
            "eligibility": eligibility,
        }
        for priority_order in PRIORITY_ORDERS
        for eligibility in ELIGIBILITY_ORDER
    ]
    core = {
        "protocol_version": "high_volatility_cross_structure_priority_router_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-14",
        "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False,
        "router_incidence_opened": False,
        "router_outcomes_opened": False,
        "action_order": list(ACTION_ORDER),
        "priority_orders": [list(order) for order in PRIORITY_ORDERS],
        "eligibility_order": list(ELIGIBILITY_ORDER),
        "candidates": candidates,
        "candidate_family": list(CANDIDATE_FAMILY),
        "candidate_family_size": FAMILY_SIZE,
        "artifact_binding_source": {
            "actions": "exact standalone preregistration, support, Gross9, and primary-clock artifacts",
            "eligibility_policy_id": hvsof.POLICY_ID,
            "eligibility": "reuse the exact eligibility contracts and artifact bindings frozen by HVSOF-8",
        },
        "action_artifacts": ACTION_ARTIFACTS,
        "eligibility_artifacts": ELIGIBILITY_ARTIFACTS,
        "component_gate_status": {
            "actions": {
                component: {
                    "source_support_passed": True,
                    "gross9_novelty_passed": True,
                    "primary_clock_immutable": True,
                }
                for component in ACTION_ORDER
            },
            "eligibility_sources": {
                component: {
                    "source_support_passed": True,
                    "gross9_novelty_passed": True,
                    "source_state_panel_immutable": True,
                }
                for component in ELIGIBILITY_ORDER
            },
        },
        "construction": {
            "operator": "state-gated strict priority router",
            "family_definition": "two frozen action priority orders, each under each of two frozen eligibility states",
            "decision_join": "exact equality among the action decision_time and eligibility state-panel decision_time",
            "timestamp_tolerance": "none",
            "eligibility_rule": "route only at an exact decision where the candidate eligibility state is true",
            "active_action": "an action is active only when its exact frozen clock has an action at that exact decision",
            "routing_rule": "choose the first active action in the candidate priority_order",
            "simultaneous_or_conflicting_actions": "resolve only by priority_order",
            "no_active_action": "cash",
            "action_entry_side_hold": "retain the chosen action entry_time, side, and 8 elapsed hour hold exactly",
            "reservation": "retain the chosen action reservation exactly",
            "never_average": True,
            "never_intersect": True,
            "never_flip": True,
            "eligibility_has_side_or_action": False,
            "action_formula_threshold_clock_mutability": "immutable",
            "eligibility_formula_threshold_decision_mutability": "immutable",
        },
        "eligibility_conditions": {
            "HVTCCR-8": {
                "state_true": "source_valid is true and concentration_rank >= 0.80",
                "source_valid_required": True,
                "field": "concentration_rank",
                "operator": ">=",
                "threshold": 0.80,
                "uses_original_candidate_eligible_or_onset": False,
            },
            "HVLZC-8": {
                "state_true": "source_valid is true and complexity_rank <= 0.25",
                "source_valid_required": True,
                "field": "complexity_rank",
                "operator": "<=",
                "threshold": 0.25,
                "uses_original_candidate_eligible_or_onset": False,
            },
            "same_frozen_source_state_panels_as_HVSOF_8": True,
            "same_decisions": True,
            "no_side_or_action": True,
        },
        "clock": {
            "action_decisions": "exact 00:00, 08:00 and 16:00 UTC",
            "entry": "exact chosen-action D+5m entry",
            "hold": "8 elapsed hours",
            "sides": "exact chosen-action side",
            "action_already_supplies_high_volatility_gate": True,
            "eligibility_adds_or_changes_high_volatility_gate": False,
            "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.20,
            "max_month_share": 0.45,
        },
        "gross9_novelty_gates": {
            "exact_entry_jaccard_max": 0.10,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": BONFERRONI_RAW_P_MAX,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR",
        },
        "train_only_selection": {
            "eligibility": "candidate passes its frozen all-stage source-support gate and Gross9 novelty gate",
            "ranking_metric": "descending train base-cost CAGR divided by strict MDD",
            "tie_breaks": [
                "descending train base-cost absolute return",
                "ascending fixed candidate_family order",
            ],
            "winner_train_gate": "raw rank-one candidate must pass every train economic gate",
            "freeze_deadline": "winner identity and complete train evidence are written before any test outcome is opened",
            "no_substitution": "if raw rank one fails train or any later gate, terminate; never substitute rank two or another candidate",
            "future_reselection_or_repair": False,
        },
        "familywise_multiplicity": {
            "family": "all four frozen priority-order-by-eligibility candidates, including candidates failing source or Gross9 gates",
            "rule": "Bonferroni",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "number_of_hypotheses": FAMILY_SIZE,
            "winner_raw_weekly_signflip_p_max": BONFERRONI_RAW_P_MAX,
            "equivalent_adjusted_p": "min(1, 4 * raw one-sided weekly sign-flip p)",
        },
        "research_boundary": {
            "all_action_component_outcomes_known": True,
            "all_eligibility_component_outcomes_known": True,
            "all_component_outcomes_known": True,
            "all_constituent_outcomes_known": True,
            "all_prior_outcomes_known": True,
            "all_prior_family_outcomes_known": True,
            "prior_HVSOF_outcomes_known": True,
            "prior_CARSC_outcomes_known": True,
            "prior_HVCMMI_outcomes_known": True,
            "prior_HVIABR_outcomes_known": True,
            "prior_HVMCPAC_outcomes_known": True,
            "prior_HVMDPAC_outcomes_known": True,
            "known_outcomes_used_to_change_frozen_components": False,
            "router_incidence_opened": False,
            "router_postentry_returns_or_pnl_opened": False,
            "test_outcomes_opened_before_winner_freeze": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": "Fixed sequence per candidate: source support, Gross9 novelty, train-only ranking and raw rank-one train gates, then frozen-winner test/eval/final economics. Stop on first failure with no substitution or action/eligibility formula, threshold, priority, direction, onset, entry, side, hold, reservation, subset, blending, intersection, flip, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def select_train_winner(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze raw train rank one from result rows supplied only after preregistration."""
    if len(rows) != FAMILY_SIZE:
        raise ValueError("HVCSPR-8 selection requires exactly four candidate rows")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        candidate = row.get("candidate")
        if not isinstance(candidate, str) or candidate in by_id:
            raise ValueError("HVCSPR-8 candidate IDs must be unique strings")
        by_id[candidate] = row
    if set(by_id) != set(CANDIDATE_FAMILY):
        raise ValueError("HVCSPR-8 selection requires the exact frozen family")

    eligible: list[tuple[int, Mapping[str, Any], float, float]] = []
    for order, candidate in enumerate(CANDIDATE_FAMILY):
        row = by_id[candidate]
        if (
            type(row.get("source_pass")) is not bool
            or type(row.get("gross9_pass")) is not bool
        ):
            raise ValueError("HVCSPR-8 pass flags must be booleans")
        if not (row["source_pass"] and row["gross9_pass"]):
            continue
        ratio = row.get("train_cagr_to_strict_mdd")
        absolute_return = row.get("train_absolute_return")
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
        ):
            raise ValueError("HVCSPR-8 train ratio must be finite")
        if (
            isinstance(absolute_return, bool)
            or not isinstance(absolute_return, (int, float))
            or not math.isfinite(float(absolute_return))
        ):
            raise ValueError("HVCSPR-8 train return must be finite")
        if type(row.get("train_economic_pass")) is not bool:
            raise ValueError("HVCSPR-8 train economic pass must be boolean")
        eligible.append((order, row, float(ratio), float(absolute_return)))
    if not eligible:
        raise RuntimeError("HVCSPR-8 has no source-and-Gross9-pass candidate")
    order, winner, ratio, absolute_return = min(
        eligible, key=lambda item: (-item[2], -item[3], item[0])
    )
    if not winner["train_economic_pass"]:
        raise RuntimeError("HVCSPR-8 raw rank one failed train; no substitution")
    return {
        "candidate": winner["candidate"],
        "family_order": order + 1,
        "train_cagr_to_strict_mdd": ratio,
        "train_absolute_return": absolute_return,
        "frozen_before_test": True,
        "substitution_authorized": False,
    }


def validate(value: Mapping[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVCSPR-8 preregistration drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("HVCSPR-8 candidate family drift")
    if value.get("priority_orders") != [list(order) for order in PRIORITY_ORDERS]:
        raise RuntimeError("HVCSPR-8 priority order drift")
    if value.get("action_artifacts") != ACTION_ARTIFACTS:
        raise RuntimeError("HVCSPR-8 action artifact bindings drift")
    if value.get("eligibility_artifacts") != hvsof.ELIGIBILITY_ARTIFACTS:
        raise RuntimeError("HVCSPR-8 eligibility artifact bindings drift")
    if (
        value.get("familywise_multiplicity", {}).get("winner_raw_weekly_signflip_p_max")
        != BONFERRONI_RAW_P_MAX
    ):
        raise RuntimeError("HVCSPR-8 multiplicity drift")
    for artifacts in (*ACTION_ARTIFACTS.values(), *ELIGIBILITY_ARTIFACTS.values()):
        for artifact_type, artifact in artifacts.items():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(f"HVCSPR-8 {artifact_type} artifact drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
