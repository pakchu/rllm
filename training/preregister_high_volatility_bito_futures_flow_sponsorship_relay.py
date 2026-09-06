"""Outcome-blind preregistration for HVBITOFL-24."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVBITOFL-24"
DEFAULT_OUTPUT = Path("results/high_volatility_bito_futures_flow_sponsorship_relay_preregistration_2026-08-11.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_bito_futures_flow_sponsorship_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "An unusually active completed BITO cash session transfers directional bitcoin-futures sponsorship into the continuously traded BTC market after US cash close. In elevated prior BTC variation, follow the sign of the completed BITO raw open-to-close return for twenty-four hours.",
            "side": "long for positive BITO raw cash-session return; short for negative return",
            "why_distinct": "BITO relative share-volume sponsorship is direct listed bitcoin-futures participation. It is not ETF range close-location, an equity-sector pair, a generic BTC price-volume indicator, opening gap, funding, OI, basis, prior event artifact, repair, or promoted control.",
            "why_suited_to_volatile_regimes": "both BITO relative share volume and preceding twenty-four-hour BTC realized variation must rank in their causal upper 35%, selecting July-like high-participation stress or expansion states",
            "why_low_gross9_overlap_is_plausible": "a sparse external US-listed bitcoin-futures fund cash-close clock and its causal relative-volume state are absent from Gross9",
        },
        external_basis={
            "official_source": "https://www.proshares.com/our-etfs/strategic/bito",
            "definition": "ProShares states BITO targets bitcoin performance through futures and swaps and does not invest directly in bitcoin",
            "selection_use": "official futures-linked exposure only; no BITO candidate incidence or economic outcomes",
        },
        features={
            "source": "candidate-specific Yahoo chart raw unadjusted BITO regular-session daily open/close/volume snapshot downloaded only after preregistration push",
            "session": "US trading date D with finite positive raw open and close and finite positive raw share volume",
            "cash_return": "log(raw close/raw open); zero is ineligible",
            "availability": "D 23:00 UTC, safely after the 16:00 America/New_York regular close in EST and EDT",
            "relative_volume_rank": "strict-prior midrank of raw BITO share volume over at most 270 prior valid sessions, minimum 180, current excluded; rank>=0.65",
            "btc_variation": "sqrt(sum squared exact BTCUSDT five-minute open-to-close log returns over the prior 24 elapsed hours ending at decision)",
            "variation_rank": "strict-prior midrank over at most 270 previous valid daily 23:00 UTC states, minimum 180, current excluded; rank>=0.65",
            "no_adjusted_close": True, "no_imputation": True,
        },
        clock={
            "decision": "each valid BITO regular-session date D at 23:00 UTC",
            "entry": "exact BTCUSDT 23:05 UTC open", "hold": "24 elapsed hours",
            "side": "sign of completed raw BITO cash-session return",
            "reservation": "chronological global half-open nonoverlap; exit first on equal open",
            "gross_exposure": 0.5, "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "relative_volume_rank_min": 0.65, "relative_volume_history_sessions": 270,
            "minimum_relative_volume_history_sessions": 180,
            "variation_history_decisions": 270, "minimum_variation_history_decisions": 180,
            "variation_rank_min": 0.65, "entry_delay_minutes": 5, "hold_hours": 24,
            "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": ["no_variation_gate", "no_relative_volume_gate", "one_session_stale_flow", "direction_flip", "forced_long"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bito": "Yahoo chart query1=2022-09-01 query2=2026-08-02 exclusive interval=1d events=history; raw open/close/volume only",
            "destination": "data/high_volatility_bito_futures_flow_sponsorship_relay_sources_2022_2026/bito_sessions.csv.gz",
            "btc": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "read_after_preregistration": True},
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "official_etf_definition_read": True, "repository_bito_flow_candidate_found": False,
            "bito_values_opened": False, "prior_external_etf_economic_outcomes_known": True,
            "prior_event_sets_reused": False, "prior_candidate_economic_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False, "candidate_count": 1, "grid": False,
            "repair_of_prior_candidate": False, "promoted_prior_control": False,
            "selection_basis": "independent direct bitcoin-futures ETF participation-transfer channel plus requested high BTC variation",
        },
        stopping_rule="Terminal first failure; no listed-fund substitution, adjusted-price use, volume-rank, variation, side, session, hold, clock, subset, threshold, or control repair.",
    )
    return {**core, "manifest_hash": canonical_hash(core)}

def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVBITOFL preregistration drift")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); payload = build(); validate(payload); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); print(args.output)
