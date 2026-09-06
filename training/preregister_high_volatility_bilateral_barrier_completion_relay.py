"""Outcome-blind preregistration for HVBBC-8."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVBBC-8"
DEFAULT_OUTPUT = Path("results/high_volatility_bilateral_barrier_completion_relay_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_bilateral_barrier_completion_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-13",
        mechanism={
            "claim": "A completed high-variation eight-hour BTC path that closes through both symmetric, strictly-prior-volatility-scaled barriers has transferred control from the first passage direction to the opposite second passage direction. When the final hour and terminal displacement confirm that second passage, follow the completed transfer for eight elapsed hours.",
            "side": "direction of the second symmetric-barrier first passage",
            "why_distinct": "HVBBC orders first passages through two fixed barriers around the block opening price. Daily range traversal orders endogenous absolute extrema; directional-change candidates repeatedly reset running extrema; Donchian reclaim uses trailing historical price bounds; reversal-completion candidates aggregate fixed temporal halves. HVBBC uses no absolute-extreme identity, resetting state machine, fitted outcome, volume, flow, OI, funding, prior event, control, or profile input.",
            "why_suited_to_volatile_regimes": "both completed eight-hour variation and a full bilateral volatility-scaled traversal are required, directly targeting turbulent control transfer",
            "why_low_gross9_overlap_is_plausible": "three fixed UTC bilateral first-passage completions with eight-hour reservation are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "symmetric first-passage barrier construction",
            "fixed_definition": "freeze a one-hour standard-deviation barrier from the preceding twenty-four hours, then order the first completed-close passages above and below the opening reference",
            "selection_use": "the symmetric first-passage construction only; no directional return, incidence, or outcome claim is imported",
        },
        features={
            "decision_grid": "exact 00:00, 08:00, and 16:00 UTC boundaries",
            "prior_scale_window": "288 exact completed five-minute closes over [T-32h,T-8h), yielding 287 close-to-close log returns",
            "barrier_scale": "sqrt(sum squared prior-scale returns / 24), a finite strict-positive one-hour realized standard-deviation scale frozen before the current block",
            "current_path": "96 exact completed five-minute closes over [T-8h,T), with reference equal to the first bar open",
            "upper_barrier": "reference*exp(barrier_scale)",
            "lower_barrier": "reference*exp(-barrier_scale)",
            "first_passages": "earliest completed five-minute close >= upper barrier and earliest completed five-minute close <= lower barrier; both must exist at distinct timestamps",
            "second_passage_direction": "+1 when the lower passage occurs first and upper passage second; -1 when upper occurs first and lower second",
            "terminal_confirmation": "final completed close is strictly beyond the reference in the second-passage direction and the final twelve-bar close-to-close return has the same strict sign",
            "current_variation": "sqrt(sum squared close-to-close log returns across the 96 current-path closes), finite strict positive",
            "variation_rank": "strict-prior midrank over at most 270 earlier source-valid decisions, minimum 180, current excluded; rank>=0.65",
            "eligibility": "both distinct passages, terminal confirmation, and variation rank gate all hold",
            "no_imputation": True,
        },
        clock={
            "feature_available": "eight-hour boundary after all current-path closes complete",
            "entry": "exact BTCUSDT open five elapsed minutes later",
            "side": "second-passage direction",
            "hold": "8 elapsed hours",
            "reservation": "global half-open first-eligible reservation; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "bar_minutes": 5, "prior_scale_bars": 288, "current_path_bars": 96,
            "decision_hours": 8, "variation_history_decisions": 270,
            "minimum_history_decisions": 180, "variation_rank_min": 0.65,
            "terminal_confirmation_bars": 12, "entry_delay_minutes": 5,
            "hold_hours": 8, "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": ["no_variation_gate", "no_terminal_confirmation", "one_decision_stale_barrier", "direction_flip", "forced_long"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2023-05-25T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "symmetric_first_passage_definition_read": True,
            "repository_exact_bilateral_barrier_completion_candidate_found": False,
            "adjacent_range_traversal_directional_change_and_reversal_completion_candidates_known": True,
            "adjacent_candidate_outcomes_used_to_set_formula_side_hold_clock_or_threshold": False,
            "prior_event_sets_reused": False, "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False,
            "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "fixed symmetric prior-volatility barrier first-passage completion under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no scale window, barrier formula, path, confirmation, variation rank, side, hold, clock, subset, threshold, or control repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVBBC preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
