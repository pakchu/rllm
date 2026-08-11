"""Outcome-blind preregistration for HVKRECLV-24."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVKRECLV-24"
DEFAULT_OUTPUT = Path("results/high_volatility_regional_bank_close_location_relay_preregistration_2026-08-11.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_regional_bank_close_location_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "A completed US cash-session close in the outer quarter of the regional-bank ETF range identifies accepted bank-credit risk pressure rather than a transient intraday excursion. During elevated BTC variation, follow that accepted pressure for twenty-four hours after the close is certainly public.",
            "side": "long when KRE close-location value is at least +0.5; short when at most -0.5",
            "why_distinct": "KRE close location uses one regional-bank cash auction's high-low-close geometry. It is not semiconductor/utilities or HYG/LQD relative return, broad financial ETF direction, an opening gap, crypto price geometry, OI, funding, prior event artifact, repair, ETF substitution inside a prior candidate, or promoted control.",
            "why_suited_to_volatile_regimes": "the preceding twenty-four-hour BTC realized variation must rank in its causal upper 35%, restricting bank-credit transmission to July-like crypto states",
            "why_low_gross9_overlap_is_plausible": "only outer-quarter regional-bank cash-auction closes create one external 23:05 UTC clock; this primitive is absent from Gross9",
        },
        external_basis={
            "official_source": "https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-sp-regional-banking-etf-kre",
            "definition": "KRE seeks exposure to the regional banks segment of the S&P Total Market Index through a modified equal-weighted regional-bank index",
            "fixed_geometry": "CLV=((close-low)-(high-close))/(high-low)=2*(close-low)/(high-low)-1; +/-0.5 is the boundary of the upper/lower quarter of the completed range",
            "selection_use": "official regional-bank exposure and algebraic outer-quarter auction acceptance only; no candidate incidence or economic outcomes",
        },
        features={
            "source": "candidate-specific Yahoo chart raw unadjusted KRE regular-session daily open/high/low/close snapshot downloaded only after preregistration push",
            "session": "US trading date D with finite positive raw OHLC, high>=max(open,close), low<=min(open,close), and high>low",
            "close_location": "((close-low)-(high-close))/(high-low), finite in [-1,1] up to numerical tolerance",
            "event": "close_location>=+0.5 or <=-0.5; interior values are ineligible",
            "availability": "D 23:00 UTC, safely after the 16:00 America/New_York regular close in EST and EDT",
            "btc_variation": "sqrt(sum squared exact BTCUSDT five-minute open-to-close log returns over the prior 24 elapsed hours ending at decision)",
            "variation_rank": "strict-prior midrank over at most 270 previous valid daily 23:00 UTC states, minimum 180, current excluded; rank>=0.65",
            "no_adjusted_close": True,
            "no_imputation": True,
        },
        clock={
            "decision": "each valid KRE regular-session date D at 23:00 UTC",
            "entry": "exact BTCUSDT 23:05 UTC open",
            "hold": "24 elapsed hours",
            "side": "strict outer-quarter close-location sign",
            "reservation": "chronological global half-open nonoverlap; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "absolute_close_location_min": 0.5,
            "variation_history_decisions": 270,
            "minimum_variation_history_decisions": 180,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": ["no_variation_gate", "any_nonzero_close_location", "cash_return_direction", "one_session_stale_location", "direction_flip", "forced_long"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "kre": "Yahoo chart query1=2022-09-01 query2=2026-08-02 exclusive interval=1d events=history; raw OHLC only",
            "destination": "data/high_volatility_regional_bank_close_location_relay_sources_2022_2026/kre_sessions.csv.gz",
            "btc": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "read_after_preregistration": True},
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "official_etf_definition_read": True,
            "repository_kre_close_location_candidate_found": False,
            "kre_values_opened": False,
            "prior_external_etf_economic_outcomes_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_economic_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent single-auction regional-bank credit-pressure channel plus requested high BTC variation",
        },
        stopping_rule="Terminal first failure; no ETF substitution, adjusted-price use, CLV boundary, variation, side, session, hold, clock, subset, threshold, or control repair.",
    )
    return {**core, "manifest_hash": canonical_hash(core)}

def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVKRECLV preregistration drift")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
