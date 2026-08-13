"""Outcome-blind preregistration for HVFXTE-12."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVFXTE-12"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_fx_transfer_entropy_network_relay_preregistration_2026-08-13.json"
)
SYMBOLS = ("EURUSD", "GBPUSD", "USDAUD", "USDCAD", "USDCHF", "USDJPY")
MULTIPLIERS = {"EURUSD": -1, "GBPUSD": -1, "USDAUD": 1, "USDCAD": 1, "USDCHF": 1, "USDJPY": 1}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_fx_transfer_entropy_network_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-13",
        mechanism={
            "claim": (
                "A liquid-session FX node whose return signs provide more conditional information to "
                "the next signs of at least four peers than those peers provide in reverse is a "
                "nonlinear source of a global-dollar repricing network. During elevated completed BTC "
                "variation, relay opposite that source node's canonical dollar direction for twelve hours."
            ),
            "side": "negative strict sign of the selected source node's completed canonical-dollar session return",
            "why_distinct": (
                "HVDFXPR uses antisymmetric one-minute linear lagged Pearson correlations. FX "
                "synchronization uses contemporaneous correlation, dollar breadth uses endpoint signs, "
                "and HVRVTE is a BTC-internal turnover-to-return statistic. HVFXTE instead constructs "
                "pairwise categorical transfer entropies I(sign_i[t-1];sign_j[t]|sign_j[t-1]) and "
                "selects a unique net outgoing nonlinear-information source. It uses no BTC return "
                "direction, fitted outcome, reused event set, funding, OI, premium, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "the nonlinear FX source strength must occupy its causal upper quartile and completed "
                "BTC twenty-four-hour variation its causal upper thirty-five percent"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "sparse weekday 21:05 UTC entries selected by an external nonlinear FX network are "
                "absent from Gross9 primitives"
            ),
        },
        external_basis={
            "origin": "transfer entropy as directional conditional mutual information",
            "fixed_definition": "TE(i->j)=I(sign_i[t-1];sign_j[t]|sign_j[t-1]) by empirical frequencies and natural logarithms",
            "selection_use": "the nonlinear information-direction object only; no candidate incidence or BTC outcome claim is imported",
        },
        features={
            "fx_universe": list(SYMBOLS),
            "canonical_dollar_multipliers": MULTIPLIERS,
            "session": "each Monday-Friday exact half-open [13:00,21:00) UTC session S",
            "source_rows": "bars_polygon one-minute finite positive coherent OHLC rows; no imputation",
            "common_path": "exact timestamp intersection across all six pairs, first timestamp no later than 13:05, last no earlier than 20:55, at least 420 timestamps",
            "canonical_return_sign": "strict sign of multiplier_i*log(close_i[t]/close_i[t-1]) on exact consecutive common minutes; any zero return makes that transition unavailable",
            "pair_samples": "for each ordered i!=j, exact transitions with finite strict signs for i[t-1], j[t-1], and j[t], minimum 360",
            "conditioning_support": "each binary (sign_i[t-1],sign_j[t-1]) cell has at least 20 samples",
            "directed_transfer_entropy": "empirical plug-in I(sign_i[t-1];sign_j[t]|sign_j[t-1]) using natural logarithms, finite nonnegative",
            "antisymmetric_edge": "A(i,j)=TE(i->j)-TE(j->i), with A(j,i)=-A(i,j)",
            "node_score": "sum A(i,j) over the other five nodes",
            "source_node": "unique strict maximum node_score; ties invalidate; score finite and strictly positive",
            "source_breadth": "at least four of five outgoing A(source,j) values strictly positive",
            "source_strength_rank": "strict-prior midrank of source node_score over at most 90 earlier source-valid sessions, minimum 60, current excluded; rank>=0.75",
            "source_direction": "canonical log(last common close/first common close) of the selected source node, finite strict nonzero",
            "btc_variation": "sqrt(sum squared BTCUSDT perpetual one-hour log returns)) over 24 exact completed hours ending at 21:00 UTC",
            "btc_variation_rank": "strict-prior midrank over at most 90 earlier source-valid sessions, minimum 60, current excluded; rank>=0.65",
            "eligibility": "source-valid unique source, breadth>=4, source-strength rank gate, BTC-variation rank gate, and strict source direction",
            "onset": "eligible now and immediately previous source-valid weekday session ineligible; insufficient history counts as ineligible",
            "no_imputation": True,
        },
        clock={
            "decision": "exact weekday 21:00 UTC after FX session and BTC variation complete",
            "entry": "exact BTCUSDT perpetual 21:05 UTC open",
            "side": "opposite selected source-node canonical dollar session direction",
            "hold": "12 elapsed hours",
            "reservation": "global chronological half-open first-eligible reservation; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "minimum_common_minutes": 420,
            "minimum_pair_transitions": 360,
            "minimum_conditioning_cell_count": 20,
            "minimum_positive_outgoing_edges": 4,
            "rank_history_sessions": 90,
            "rank_minimum_sessions": 60,
            "source_strength_rank_min": 0.75,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 12,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": [
                "no_source_strength_tail",
                "no_variation_gate",
                "no_breadth_gate",
                "linear_lag_network",
                "one_session_stale_network",
                "direction_flip",
                "same_clock_forced_long",
            ],
            "cannot_be_promoted": True,
        },
        source_plan={
            "fx": {
                "table": "bars_polygon",
                "symbols": list(SYMBOLS),
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "btc": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "close"],
                "completed_variation_only": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "transfer_entropy_definition_read": True,
            "repository_exact_fx_transfer_entropy_network_candidate_found": False,
            "adjacent_linear_directed_fx_synchronization_and_breadth_candidates_known": True,
            "adjacent_candidate_outcomes_used_to_set_formula_side_hold_clock_or_threshold": False,
            "prior_event_sets_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "fixed nonlinear directional-information network across six liquid FX pairs under the requested high-variation BTC regime",
        },
        stopping_rule=(
            "Terminal first failure; no universe, session, sign encoding, lag, conditioning, cell support, "
            "entropy estimator, graph, breadth, rank, side, hold, clock, subset, threshold, or control repair."
        ),
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVFXTE preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
