"""Outcome-blind preregistration for the fixed HVSOF-8 ordered-filter battery."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_ID = "HVSOF-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_state_ordered_filter_preregistration_2026-08-14.json"
)
ACTION_ORDER = ("HVRSSR-8", "HVTFR-8", "HVSVF-8")
FILTER_ORDER = ("HVTCCR-8", "HVLZC-8")
CANDIDATE_FAMILY = tuple(
    f"{action}__FILTERED_BY__{state_filter}"
    for action in ACTION_ORDER
    for state_filter in FILTER_ORDER
)
FAMILY_SIZE = 6
FAMILYWISE_ALPHA = 0.10
BONFERRONI_RAW_P_MAX = FAMILYWISE_ALPHA / FAMILY_SIZE

ACTION_ARTIFACTS: dict[str, dict[str, dict[str, str]]] = {
    "HVRSSR-8": {
        "preregistration": {"path": "results/high_volatility_roll_spread_shock_reversal_preregistration_2026-08-10.json", "sha256": "b712a8b708a1c7fb28a2bef021a6534f04ed241b4b4fd5dd165359d5d1dcdf81"},
        "support": {"path": "results/high_volatility_roll_spread_shock_reversal_support_2026-08-10.json", "sha256": "2f531003ff623c316d70edc8b4bd9b20d03be3916b6d5b4eddbe4dd5f19f2cd1"},
        "gross9": {"path": "results/high_volatility_roll_spread_shock_reversal_gross9_novelty_2026-08-10.json", "sha256": "306a59dab9943196d0e3321ca5be50b216ccef4c9c7347b1010cb9897ffe4cff"},
        "clock": {"path": "data/high_volatility_roll_spread_shock_reversal_clocks_2023_2026.csv.gz", "sha256": "d1e5c102e486abd9dd41ce42b822113ced3103d3eb892008a97b201f5ba1574f"},
    },
    "HVTFR-8": {
        "preregistration": {"path": "results/high_volatility_time_price_trend_fit_relay_preregistration_2026-08-10.json", "sha256": "2bbf372142c15f00714bfc564013a4de4e326f291e447491d8c3d28519d67745"},
        "support": {"path": "results/high_volatility_time_price_trend_fit_relay_support_2026-08-10.json", "sha256": "5547ab700e27c98f04b47fa07756775b0245ff18bb7e466d56f5f0348b570588"},
        "gross9": {"path": "results/high_volatility_time_price_trend_fit_relay_gross9_novelty_2026-08-10.json", "sha256": "890196f90cc3e9ffd58c4fe0d71dcee61681246b509e858c3f7591277a93e984"},
        "clock": {"path": "data/high_volatility_time_price_trend_fit_relay_clocks_2023_2026.csv.gz", "sha256": "f01c22a8409ef9b7f12b17d26b3a3b25564b3aad210a6c00fff789756cbe4151"},
    },
    "HVSVF-8": {
        "preregistration": {"path": "results/high_volatility_signed_variance_feedback_relay_preregistration_2026-08-10.json", "sha256": "c7193999573aa0dffe77afbd8637d7fcf482598a8124907b73bfd12d967508e8"},
        "support": {"path": "results/high_volatility_signed_variance_feedback_relay_support_2026-08-10.json", "sha256": "e0566e354f2ade805201db19d6c81a7c717447937ab993a962ad9954cc1f471b"},
        "gross9": {"path": "results/high_volatility_signed_variance_feedback_relay_gross9_novelty_2026-08-10.json", "sha256": "d6cffcf528c6d095123cba430706898473ac150ccff11f05ddac9882efee2b69"},
        "clock": {"path": "data/high_volatility_signed_variance_feedback_relay_clocks_2023_2026.csv.gz", "sha256": "4b584277f27fc8791f7ca4bc3902099b477d97f69cf9d7d36ab2e414ae6b287d"},
    },
}

ELIGIBILITY_ARTIFACTS: dict[str, dict[str, dict[str, str]]] = {
    "HVTCCR-8": {
        "preregistration": {"path": "results/high_volatility_quote_turnover_concentration_continuation_relay_preregistration_2026-08-10.json", "sha256": "b3fd974ae3ba804679e63d30ea02ee6d0ad3246981e80b7308242d09d19e3d26"},
        "support": {"path": "results/high_volatility_quote_turnover_concentration_continuation_relay_support_2026-08-10.json", "sha256": "d874a611ae2ea65637fbbf7ca607062cc3df9c4d5989019bb23bf1abbb2713b3"},
        "gross9": {"path": "results/high_volatility_quote_turnover_concentration_continuation_relay_gross9_novelty_2026-08-10.json", "sha256": "910993f4a70ba770a9e66273630e14ca4df175864ba8f7009cb388c31fd8c28f"},
        "state_panel": {"path": "data/high_volatility_quote_turnover_concentration_continuation_relay_sources_2023_2026/block_states.csv.gz", "sha256": "3bb6161fcf13ea40cd8909fd942f0bce0abc95c96d88bee083437329b7fe15f4"},
        "source_manifest": {"path": "data/high_volatility_quote_turnover_concentration_continuation_relay_sources_2023_2026/manifest.json", "sha256": "fd24f2a65e7543739bf063cd290f815bbaf7a3b58e7357fb10478c7f5a412a54"},
    },
    "HVLZC-8": {
        "preregistration": {"path": "results/high_volatility_lempel_ziv_compressibility_relay_preregistration_2026-08-10.json", "sha256": "1aff82bc58760b8dfc8c14798c20085dc8e9387caaec71f478564855e6afda8b"},
        "support": {"path": "results/high_volatility_lempel_ziv_compressibility_relay_support_2026-08-10.json", "sha256": "d85fabbd757859f1c9fa38385d992b2a79bf6b9ebfd4ac32673f88e30d4bce3c"},
        "gross9": {"path": "results/high_volatility_lempel_ziv_compressibility_relay_gross9_novelty_2026-08-10.json", "sha256": "3ca73e95678052c3dc9677fa01b16e3576bc00f53cfef7ffb6dbaad40fed8e59"},
        "state_panel": {"path": "data/high_volatility_lempel_ziv_compressibility_sources_2023_2026/states.csv.gz", "sha256": "646687d9d0aca16c03148bd8ea7294286e61220cf33f8ce755570cd5633cb60a"},
        "source_manifest": {"path": "data/high_volatility_lempel_ziv_compressibility_sources_2023_2026/manifest.json", "sha256": "a46d8e482bb54979b713ace9a3543b5db0492ae793433ddda34559632165056a"},
    },
}


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_state_ordered_filter_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-14",
        "exploratory_discovery": True,
        "fresh_confirmatory_evidence": False,
        "filter_incidence_opened": False,
        "filter_outcomes_opened": False,
        "action_order": list(ACTION_ORDER),
        "filter_order": list(FILTER_ORDER),
        "candidate_family": list(CANDIDATE_FAMILY),
        "candidate_family_size": FAMILY_SIZE,
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
                for component in FILTER_ORDER
            },
        },
        "construction": {
            "operator": "ordered action-by-filter",
            "family_definition": "Cartesian product of three frozen actions in action_order by two frozen eligibility filters in filter_order",
            "decision_join": "exact equality between the action decision_time and filter state-panel decision_time",
            "timestamp_tolerance": "none",
            "retention": "retain an action candidate exactly when its ordered filter state is true at the corresponding decision",
            "action_entry_side_hold": "retain the action entry_time, side, and hold exactly",
            "reservation": "retain the action reservation exactly; the filter does not add or change reservation",
            "filter_has_side_or_action": False,
            "filter_may_change_variation_direction_onset_or_reservation": False,
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
            "same_frozen_source_state_panels": True,
            "same_decisions": True,
            "no_side_or_action": True,
        },
        "clock": {
            "action_decisions": "exact 00:00, 08:00 and 16:00 UTC",
            "entry": "exact action D+5m entry",
            "hold": "8 elapsed hours",
            "action_already_supplies_high_volatility_gate": True,
            "filter_adds_or_changes_high_volatility_gate": False,
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
            "weekly_signflip_one_sided_p_max": FAMILYWISE_ALPHA,
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
            "family": "all six ordered action-by-filter candidates, including candidates failing source or Gross9 gates",
            "rule": "Bonferroni",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "number_of_hypotheses": FAMILY_SIZE,
            "winner_raw_weekly_signflip_p_max": BONFERRONI_RAW_P_MAX,
            "equivalent_adjusted_p": "min(1, 6 * raw one-sided weekly sign-flip p)",
        },
        "research_boundary": {
            "all_action_component_outcomes_known": True,
            "all_eligibility_component_outcomes_known": True,
            "all_component_outcomes_known": True,
            "prior_HVMCPAC_outcomes_known": True,
            "prior_HVMDPAC_outcomes_known": True,
            "known_outcomes_used_to_change_frozen_components": False,
            "filter_incidence_opened": False,
            "filter_postentry_returns_or_pnl_opened": False,
            "test_outcomes_opened_before_winner_freeze": False,
            "classification": "exploratory discovery; not fresh confirmatory evidence",
        },
        "stopping_rule": "Fixed sequence per candidate: source support, Gross9 novelty, train-only ranking and raw rank-one train gates, then frozen-winner test/eval/final economics. Stop on first failure with no substitution or action/filter formula, threshold, variation, direction, onset, entry, side, hold, reservation, subset, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def select_train_winner(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze raw train rank one from result rows supplied only after preregistration."""
    if len(rows) != FAMILY_SIZE:
        raise ValueError("HVSOF-8 selection requires exactly six candidate rows")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        candidate = row.get("candidate")
        if not isinstance(candidate, str) or candidate in by_id:
            raise ValueError("HVSOF-8 candidate IDs must be unique strings")
        by_id[candidate] = row
    if set(by_id) != set(CANDIDATE_FAMILY):
        raise ValueError("HVSOF-8 selection requires the exact frozen family")

    eligible: list[tuple[int, Mapping[str, Any], float, float]] = []
    for order, candidate in enumerate(CANDIDATE_FAMILY):
        row = by_id[candidate]
        if type(row.get("source_pass")) is not bool or type(row.get("gross9_pass")) is not bool:
            raise ValueError("HVSOF-8 pass flags must be booleans")
        if not (row["source_pass"] and row["gross9_pass"]):
            continue
        ratio = row.get("train_cagr_to_strict_mdd")
        absolute_return = row.get("train_absolute_return")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(float(ratio)):
            raise ValueError("HVSOF-8 train ratio must be finite")
        if isinstance(absolute_return, bool) or not isinstance(absolute_return, (int, float)) or not math.isfinite(float(absolute_return)):
            raise ValueError("HVSOF-8 train return must be finite")
        if type(row.get("train_economic_pass")) is not bool:
            raise ValueError("HVSOF-8 train economic pass must be boolean")
        eligible.append((order, row, float(ratio), float(absolute_return)))
    if not eligible:
        raise RuntimeError("HVSOF-8 has no source-and-Gross9-pass candidate")
    order, winner, ratio, absolute_return = sorted(
        eligible, key=lambda item: (-item[2], -item[3], item[0])
    )[0]
    if not winner["train_economic_pass"]:
        raise RuntimeError("HVSOF-8 raw rank one failed train; no substitution")
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
        raise RuntimeError("HVSOF-8 preregistration drift")
    if tuple(value.get("candidate_family", ())) != CANDIDATE_FAMILY:
        raise RuntimeError("HVSOF-8 candidate family drift")
    if value.get("action_artifacts") != ACTION_ARTIFACTS:
        raise RuntimeError("HVSOF-8 action artifact bindings drift")
    if value.get("eligibility_artifacts") != ELIGIBILITY_ARTIFACTS:
        raise RuntimeError("HVSOF-8 eligibility artifact bindings drift")
    for artifacts in (*ACTION_ARTIFACTS.values(), *ELIGIBILITY_ARTIFACTS.values()):
        for artifact_type, artifact in artifacts.items():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(f"HVSOF-8 {artifact_type} artifact drift")


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
