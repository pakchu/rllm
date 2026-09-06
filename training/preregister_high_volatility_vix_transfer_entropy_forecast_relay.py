"""Outcome-blind preregistration for HVVIXTE-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVVIXTE-24"
DEFAULT_OUTPUT = Path("results/high_volatility_vix_transfer_entropy_forecast_relay_preregistration_2026-08-14.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build()); contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_vix_transfer_entropy_forecast_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-14",
        mechanism={
            "claim": (
                "During elevated BTC variation, the latest completed VIX direction can alter the conditional "
                "distribution of the next BTC daily direction beyond BTC's own immediately preceding direction. "
                "A strong strictly-prior conditional transition forecast should relay for one day."
            ),
            "side": "strict sign of P(next BTC sign=+1 | latest VIX sign, prior BTC sign)-0.5",
            "why_distinct": (
                "CCVTR and prior VIX candidates map the current VIX change directly to BTC, optionally requiring "
                "crypto-volatility or BTC confirmation. HVRVTE is BTC-internal turnover-to-return information. "
                "HVVIXTE instead estimates a causal categorical macro-to-BTC transition law from historical "
                "VIX and BTC signs, conditions on BTC's own prior sign, and trades the resulting forecast rather "
                "than the VIX sign. It reuses no prior event set or diagnostic control."
            ),
            "why_suited_to_volatile_regimes": "the current completed BTC decision-to-decision variation must occupy its causal upper thirty-five percent",
            "why_low_gross9_overlap_is_plausible": "daily external-state conditional forecasts at next-session 09:40 ET are absent from Gross9 primitives",
        },
        external_basis={
            "paper": "Palazzi, Raimundo Júnior, and Klotzle (2026), From Network Fundamentals to Macro-Financial Integration: The Evolving Predictability of Bitcoin Returns",
            "ssrn": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6199098",
            "reported_facts": [
                "the study evaluates nonlinear transfer entropy from macro-finance variables to Bitcoin returns",
                "macro-finance information flows become strongest during stress regimes",
                "equity volatility is included among the macro-finance predictor classes",
            ],
            "untested_adaptation": "a binary VIX-to-BTC conditional transition probability is tested as a tradable daily direction forecast under the requested high-variation regime",
        },
        features={
            "source_sessions": "ordered common official Cboe source dates with finite positive VIX closes",
            "decision": "for source date D_i, the next common Cboe source date D_(i+1) at 09:35 America/New_York, when VIX(D_i) is certainly public",
            "vix_state": "strict sign log(VIX(D_i)/VIX(D_(i-1))); zero invalid",
            "btc_state": "strict sign of BTCUSDT log open-to-open return from decision A_i to current decision A_(i+1); exact five-minute opens, zero invalid",
            "historical_transition": "for each earlier i, x=vix_state_i, z=btc_state through A_(i+1), y=BTC sign from A_(i+1) to A_(i+2); sample becomes available only at A_(i+2)",
            "causal_history": "at most 756 historical transitions whose y completion time is <= current decision, minimum 252; current transition excluded",
            "cell_support": "current (x,z) binary conditioning cell has at least 30 historical transitions and both y signs occur at least 5 times",
            "conditional_probability": "empirical unsmoothed fraction of y=+1 in the current (x,z) cell; finite and strictly unequal to 0.5",
            "forecast_strength": "abs(conditional_probability-0.5)",
            "strength_rank": "strict-prior midrank of forecast_strength over at most 756 earlier forecast-valid decisions, minimum 252, current excluded; rank>=0.75",
            "btc_variation": "sqrt(sum squared exact BTCUSDT five-minute open-to-close log returns)) over [previous decision,current decision)",
            "variation_rank": "strict-prior midrank over at most 756 earlier jointly source-valid decisions, minimum 252, current excluded; rank>=0.65",
            "eligibility": "all source/history/cell conditions, strength rank>=0.75, variation rank>=0.65, and strict forecast side",
            "onset": "eligible now and immediately previous forecast-valid decision ineligible; insufficient history counts as ineligible",
            "no_imputation": True,
        },
        clock={
            "feature_available": "current next-source-session 09:35 America/New_York decision",
            "entry": "exact BTCUSDT decision+5m open (09:40 America/New_York)",
            "side": "sign(conditional_probability-0.5)",
            "hold": "24 elapsed hours",
            "reservation": "global chronological half-open first-eligible reservation; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "transition_history": 756, "minimum_transitions": 252, "minimum_conditioning_cell": 30,
            "minimum_each_target_sign_in_cell": 5, "strength_rank_min": 0.75,
            "variation_rank_min": 0.65, "entry_delay_minutes": 5, "hold_hours": 24,
            "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": ["vix_sign_direct_fade", "unconditional_btc_transition", "no_strength_tail", "no_variation_gate", "one_session_stale_forecast", "direction_flip", "same_clock_forced_long"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "vix": {"path": "data/cboe_volatility_surface_2021_2026/cboe_volatility_surface_2021-01-01_2026-08-07.csv.gz", "sha256": "42eb1093f5167aec9c71a4733ab3451e40807c81dc7cb49568a6a0c634267ba0", "column": "VIX_close", "official_source": "Cboe daily index history"},
            "btc": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "window": ["2021-01-01T00:00:00Z", "2026-08-11T00:00:00Z"], "read_after_preregistration": True},
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "paper_abstract_read": True, "repository_exact_vix_conditional_transition_candidate_found": False,
            "adjacent_vix_and_transfer_entropy_candidates_known": True,
            "adjacent_outcomes_used_to_set_formula_side_hold_clock_or_threshold": False,
            "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False, "candidate_count": 1, "grid": False,
            "repair_of_prior_candidate": False, "promoted_prior_control": False,
            "selection_basis": "published nonlinear macro-to-BTC information-flow finding, official VIX history, and requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no state encoding, history, support, probability, rank, variation, onset, side, hold, clock, subset, threshold, or control repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core): raise RuntimeError("HVVIXTE preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    payload = build(); validate(payload); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); print(args.output)
