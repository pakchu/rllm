"""Outcome-blind preregistration for HVRVTE-8."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVRVTE-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_return_volume_transfer_entropy_relay_preregistration_2026-08-13.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_return_volume_transfer_entropy_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-13",
        mechanism={
            "claim": (
                "In a completed volatile BTC auction, unusually large conditional mutual information "
                "from the previous five-minute participation state to the next return sign indicates "
                "that turnover is organizing directional price discovery beyond simple return-sign "
                "persistence. Follow the sign of the corresponding conditional up-probability lift "
                "for eight elapsed hours."
            ),
            "side": "strict sign of the turnover-conditioned next-up probability lift after conditioning on the previous return sign",
            "why_distinct": (
                "HVTLRR uses a linear correlation between continuous turnover and the next continuous "
                "return. HVVRCR uses same-bar covariance, HVRVSC uses low-frequency cross-spectral "
                "coherence, and HVPST fits a continuous temporary-impact regression. HVRVTE instead "
                "uses a discrete plug-in conditional mutual information I(turnover_state[t-1]; "
                "return_sign[t] | return_sign[t-1]) and a separately frozen conditional probability "
                "lift for direction. It uses no taker split, funding, OI, premium, fitted outcome, "
                "reused event set, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "the completed block variation must occupy its causal upper thirty-five percent and "
                "the information-transfer statistic its causal upper quartile"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "offset conditional-information onsets at 01:05, 09:05, and 17:05 UTC are absent "
                "from Gross9 primitives"
            ),
        },
        external_basis={
            "origin": "discrete transfer entropy as conditional mutual information",
            "fixed_definition": "I(X[t-1];Y[t]|Y[t-1]) estimated by empirical frequencies with natural logarithms",
            "selection_use": "the information-flow statistic only; no incidence or directional outcome claim is imported",
        },
        features={
            "decision_grid": "exact 01:00, 09:00, and 17:00 UTC boundaries D",
            "block": "96 exact epoch-aligned five-minute aggregates from 480 unique coherent BTCUSDT perpetual one-minute rows over [D-8h,D)",
            "five_minute_return": "log(last constituent close/first constituent open), finite and strictly nonzero",
            "five_minute_turnover": "sum finite strictly positive quote_asset_volume over each group",
            "turnover_state": "one when five-minute turnover is strictly greater than the within-block numpy median of all 96 turnovers, otherwise zero; both states required",
            "return_sign_state": "one for a strictly positive five-minute return and zero for a strictly negative return",
            "transition_samples": "the 95 ordered triples (turnover_state[t-1],return_sign_state[t],return_sign_state[t-1]) for t=1..95",
            "cell_support": "each of the four conditioning cells (turnover_state[t-1],return_sign_state[t-1]) contains at least five samples",
            "transfer_entropy": "empirical plug-in conditional mutual information I(turnover_state[t-1];return_sign_state[t]|return_sign_state[t-1]) using natural logarithms; finite and strictly positive",
            "conditional_up_lift": "sum over previous-sign z of empirical P(z) times [P(next_up|high_turnover,z)-P(next_up|low_turnover,z)]; finite and strictly nonzero",
            "information_rank": "strict-prior midrank of transfer_entropy over at most 270 earlier source-valid decisions, minimum 180, current excluded; rank>=0.75",
            "realized_variation": "sqrt(sum squared 96 five-minute returns), finite and strictly positive",
            "variation_rank": "strict-prior midrank over at most 270 earlier source-valid decisions, minimum 180, current excluded; rank>=0.65",
            "eligibility": "source-valid, information-rank gate, variation-rank gate, and strict nonzero conditional_up_lift",
            "onset": "eligible now and immediately previous source-valid decision ineligible; insufficient rank history counts as ineligible",
            "no_imputation": True,
        },
        clock={
            "feature_available": "decision boundary after all 480 source minutes complete",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "sign of conditional_up_lift",
            "hold": "8 elapsed hours",
            "reservation": "global chronological half-open first-eligible reservation; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "bar_minutes": 5,
            "block_bars": 96,
            "minimum_conditioning_cell_count": 5,
            "decision_hours": [1, 9, 17],
            "history_decisions": 270,
            "minimum_history_decisions": 180,
            "information_rank_min": 0.75,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": [
                "no_information_tail",
                "no_variation_gate",
                "unconditional_transition_lift",
                "contemporaneous_conditioned_information",
                "one_decision_stale_information",
                "direction_flip",
                "same_clock_forced_long",
            ],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume"],
                "window": ["2022-12-31T17:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "conditional_mutual_information_definition_read": True,
            "repository_exact_return_volume_transfer_entropy_candidate_found": False,
            "adjacent_turnover_lead_response_covariance_spectral_and_temporary_impact_candidates_known": True,
            "adjacent_candidate_outcomes_used_to_set_formula_side_hold_clock_or_threshold": False,
            "prior_event_sets_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "fixed discrete participation-to-return-sign conditional-information mechanism under the requested high-variation regime",
        },
        stopping_rule=(
            "Terminal first failure; no discretization, median, lag, conditioning state, cell support, "
            "information estimator, lift, rank, variation gate, onset, side, hold, clock, subset, "
            "threshold, or control repair."
        ),
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVRVTE preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
