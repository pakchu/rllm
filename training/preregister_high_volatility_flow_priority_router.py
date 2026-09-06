"""Outcome-blind preregistration for the fixed HVFPR-6 flow priority router."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_state_ordered_filter as hvsof


POLICY_ID = "HVFPR-6"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_flow_priority_router_preregistration_2026-08-16.json"
)
ACTION_ORDER = ("HVAFC-6", "HVELR-6", "RIVSCR-6")
PRIORITY_ORDERS = (
    ("HVAFC-6", "HVELR-6", "RIVSCR-6"),
    ("RIVSCR-6", "HVELR-6", "HVAFC-6"),
)
ELIGIBILITY_POLICY = "HVTCCR-8"


def candidate_id(priority_order: Sequence[str]) -> str:
    return f"{'__THEN__'.join(priority_order)}__ELIGIBLE_BY__{ELIGIBILITY_POLICY}"


CANDIDATE_FAMILY = tuple(candidate_id(order) for order in PRIORITY_ORDERS)
FAMILY_SIZE = 2
FAMILYWISE_ALPHA = 0.10
BONFERRONI_RAW_P_MAX = 0.05

ACTION_ARTIFACTS: dict[str, dict[str, dict[str, str]]] = {
    "HVAFC-6": {
        "preregistration": {
            "path": "results/high_volatility_aggressive_flow_confirmation_relay_preregistration_2026-08-09.json",
            "sha256": "49a1da1075a28e03a518032335d481112e62cd2eee0e042f169a58615c4a0276",
        },
        "support": {
            "path": "results/high_volatility_aggressive_flow_confirmation_relay_support_2026-08-08.json",
            "sha256": "6b88ef1337fe626a404ab5fe227d926842ff8f591a135bbb2d5408e64ac64a22",
        },
        "gross9": {
            "path": "results/high_volatility_aggressive_flow_confirmation_relay_gross9_novelty_2026-08-09.json",
            "sha256": "39ae6ab9da84cd37e42fbb792b3e8b6bb13048c2e9244abb1fec7c60679b128b",
        },
        "clock": {
            "path": "data/high_volatility_aggressive_flow_confirmation_relay_clocks_2023_2026.csv.gz",
            "sha256": "d6f576924e1170ae03799c2ab1652bed3a69f593f60df1dc372111127b09e721",
        },
    },
    "HVELR-6": {
        "preregistration": {
            "path": "results/high_volatility_eth_leadership_relay_preregistration_2026-08-09.json",
            "sha256": "b91ca28c4edcf38c33d66faa764790ea62b36eca3ba7e1bd12abb1f382c4017a",
        },
        "support": {
            "path": "results/high_volatility_eth_leadership_relay_support_2026-08-08.json",
            "sha256": "e81ce4a52cd6a201d1e7deb5c2a5d29c5f24244b306badfc4d4fa2d66e7101c2",
        },
        "gross9": {
            "path": "results/high_volatility_eth_leadership_relay_gross9_novelty_2026-08-09.json",
            "sha256": "5d1ce1ddfdeaaab45c1a575f4852a3f0d3f0c51ca63d7652ccdb618b1c8d796b",
        },
        "clock": {
            "path": "data/high_volatility_eth_leadership_relay_clocks_2023_2026.csv.gz",
            "sha256": "e3a2c92be1a6839823fd7b4e37bef16df128d766fa8eb79b0eb6d5520f6e1e44",
        },
    },
    "RIVSCR-6": {
        "preregistration": {
            "path": "results/realized_over_implied_volatility_shock_continuation_relay_preregistration_2026-08-08.json",
            "sha256": "7422e25c72a83117527a8ca99b2b20539b4cc95934934a094b52375fdd8fa034",
        },
        "support": {
            "path": "results/realized_over_implied_volatility_shock_continuation_relay_support_2026-08-08.json",
            "sha256": "db7a88628a6468d926de1231c955731bdf7fbc39ac82fe3edbe430a105c7634e",
        },
        "gross9": {
            "path": "results/realized_over_implied_volatility_shock_continuation_relay_gross9_novelty_2026-08-08.json",
            "sha256": "78ec7720c638b323f3ce1e50a07cbcb9d9dd3b7dcdcfd52c261460c5c3ca55c6",
        },
        "clock": {
            "path": "data/realized_over_implied_volatility_shock_continuation_relay_clocks_2023_2026.csv.gz",
            "sha256": "7cf742c93331044960a7ad5e6bef5d0baedcd75dc3ce83714bb8a8c69653362d",
        },
    },
}

# Reuse the frozen HVTCCR source-state contract, not its action/onset clock.
ELIGIBILITY_ARTIFACTS = {ELIGIBILITY_POLICY: hvsof.ELIGIBILITY_ARTIFACTS[ELIGIBILITY_POLICY]}


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
            "candidate": candidate_id(priority_order),
            "priority_order": list(priority_order),
            "eligibility": ELIGIBILITY_POLICY,
        }
        for priority_order in PRIORITY_ORDERS
    ]
    core = {
        "protocol_version": "high_volatility_flow_priority_router_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-16",
        "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False,
        "router_incidence_opened": False,
        "router_outcomes_opened": False,
        "action_order": list(ACTION_ORDER),
        "priority_orders": [list(order) for order in PRIORITY_ORDERS],
        "eligibility_policy": ELIGIBILITY_POLICY,
        "eligibility_order": [ELIGIBILITY_POLICY],
        "candidates": candidates,
        "candidate_family": list(CANDIDATE_FAMILY),
        "candidate_family_size": FAMILY_SIZE,
        "action_artifacts": ACTION_ARTIFACTS,
        "eligibility_artifacts": ELIGIBILITY_ARTIFACTS,
        "component_gate_status": {
            "actions": {
                action: {
                    "source_support_passed": True,
                    "gross9_novelty_passed": True,
                    "primary_clock_immutable": True,
                }
                for action in ACTION_ORDER
            },
            "eligibility_source": {
                "policy_id": ELIGIBILITY_POLICY,
                "source_support_passed": True,
                "gross9_novelty_passed": True,
                "source_state_panel_immutable": True,
            },
        },
        "construction": {
            "operator": "state-gated strict priority router",
            "family_definition": "exactly the forward and reverse frozen action priority orders under one frozen HVTCCR source state",
            "decision_join": "exact equality among action decision_time and HVTCCR state-panel decision_time",
            "timestamp_tolerance": "none",
            "eligibility_rule": "route only when HVTCCR source_valid is true and concentration_rank >= 0.80 at the same exact decision",
            "active_action": "an action is active only when its exact frozen clock has an action at that exact decision",
            "routing_rule": "choose the first active action in the candidate priority_order",
            "simultaneous_or_conflicting_actions": "resolve only by priority_order",
            "no_active_action": "cash; emit no action",
            "action_entry_side_hold": "retain the chosen action side, exact D+5m entry, and 6 elapsed hour hold",
            "reservation": "retain the chosen action reservation exactly",
            "never_average": True,
            "never_intersect": True,
            "never_flip": True,
            "eligibility_has_side_or_action": False,
            "action_formula_threshold_clock_mutability": "immutable",
            "eligibility_formula_threshold_decision_mutability": "immutable",
        },
        "eligibility_condition": {
            "policy_id": ELIGIBILITY_POLICY,
            "state_true": "source_valid is true and concentration_rank >= 0.80",
            "source_valid_required": True,
            "field": "concentration_rank",
            "operator": ">=",
            "threshold": 0.80,
            "same_decision_required": True,
            "uses_original_candidate_eligible_or_onset": False,
            "no_side_or_action": True,
        },
        "clock": {
            "action_decisions": "exact 00:00, 08:00 and 16:00 UTC",
            "entry": "exact chosen-action D+5m entry",
            "hold": "6 elapsed hours",
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
            "occupied_5m_bar_jaccard_max": 0.25,
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
            "no_substitution": "if raw rank one fails train or any later gate, terminate; never substitute rank two",
            "future_reselection_or_repair": False,
        },
        "familywise_multiplicity": {
            "family": "both frozen priority-order candidates, including candidates failing source or Gross9 gates",
            "rule": "Bonferroni",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "number_of_hypotheses": FAMILY_SIZE,
            "winner_raw_weekly_signflip_p_max": BONFERRONI_RAW_P_MAX,
            "equivalent_adjusted_p": "min(1, 2 * raw one-sided weekly sign-flip p)",
        },
        "research_boundary": {
            "all_action_component_outcomes_known": True,
            "eligibility_component_outcomes_known": True,
            "all_component_outcomes_known": True,
            "all_constituent_outcomes_known": True,
            "all_prior_outcomes_known": True,
            "all_prior_family_outcomes_known": True,
            "prior_HVAFC_outcomes_known": True,
            "prior_HVELR_outcomes_known": True,
            "prior_RIVSCR_outcomes_known": True,
            "prior_HVTCCR_outcomes_known": True,
            "known_outcomes_used_to_change_frozen_components": False,
            "router_incidence_opened": False,
            "router_postentry_returns_or_pnl_opened": False,
            "test_outcomes_opened_before_winner_freeze": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": "Fixed sequence per candidate: source support, Gross9 novelty, train-only ranking and raw rank-one train gates, then frozen-winner test/eval/final economics. Stop on first failure with no substitution or action/eligibility formula, threshold, priority, direction, entry, side, hold, reservation, subset, blending, intersection, flip, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def select_train_winner(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze raw train rank one from result rows supplied after preregistration."""
    if len(rows) != FAMILY_SIZE:
        raise ValueError("HVFPR-6 selection requires exactly two candidate rows")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        candidate = row.get("candidate")
        if not isinstance(candidate, str) or candidate in by_id:
            raise ValueError("HVFPR-6 candidate IDs must be unique strings")
        by_id[candidate] = row
    if set(by_id) != set(CANDIDATE_FAMILY):
        raise ValueError("HVFPR-6 selection requires the exact frozen family")

    eligible: list[tuple[int, Mapping[str, Any], float, float]] = []
    for order, candidate in enumerate(CANDIDATE_FAMILY):
        row = by_id[candidate]
        if (
            type(row.get("source_pass")) is not bool
            or type(row.get("gross9_pass")) is not bool
        ):
            raise ValueError("HVFPR-6 pass flags must be booleans")
        if not (row["source_pass"] and row["gross9_pass"]):
            continue
        ratio = row.get("train_cagr_to_strict_mdd")
        absolute_return = row.get("train_absolute_return")
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
        ):
            raise ValueError("HVFPR-6 train ratio must be finite")
        if (
            isinstance(absolute_return, bool)
            or not isinstance(absolute_return, (int, float))
            or not math.isfinite(float(absolute_return))
        ):
            raise ValueError("HVFPR-6 train return must be finite")
        if type(row.get("train_economic_pass")) is not bool:
            raise ValueError("HVFPR-6 train economic pass must be boolean")
        eligible.append((order, row, float(ratio), float(absolute_return)))
    if not eligible:
        raise RuntimeError("HVFPR-6 has no source-and-Gross9-pass candidate")
    order, winner, ratio, absolute_return = min(
        eligible, key=lambda item: (-item[2], -item[3], item[0])
    )
    if not winner["train_economic_pass"]:
        raise RuntimeError("HVFPR-6 raw rank one failed train; no substitution")
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
        raise RuntimeError("HVFPR-6 preregistration drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("HVFPR-6 candidate family drift")
    if value.get("priority_orders") != [list(order) for order in PRIORITY_ORDERS]:
        raise RuntimeError("HVFPR-6 priority order drift")
    if value.get("action_artifacts") != ACTION_ARTIFACTS:
        raise RuntimeError("HVFPR-6 action artifact bindings drift")
    if value.get("eligibility_artifacts") != ELIGIBILITY_ARTIFACTS:
        raise RuntimeError("HVFPR-6 eligibility artifact bindings drift")
    if (
        value.get("familywise_multiplicity", {}).get(
            "winner_raw_weekly_signflip_p_max"
        )
        != BONFERRONI_RAW_P_MAX
    ):
        raise RuntimeError("HVFPR-6 multiplicity drift")
    for artifacts in (*ACTION_ARTIFACTS.values(), *ELIGIBILITY_ARTIFACTS.values()):
        for artifact_type, artifact in artifacts.items():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(f"HVFPR-6 {artifact_type} artifact drift")


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
