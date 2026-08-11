"""Outcome-blind preregistration for HVLFX-12."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any
from training import preregister_sterling_euro_risk_beta_relay as template

POLICY_ID = "HVLFX-12"
DEFAULT_OUTPUT = Path("results/high_volatility_london_fix_dollar_impulse_relay_preregistration_2026-08-11.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build()); core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_london_fix_dollar_impulse_relay_v1",
        policy_id=POLICY_ID, as_of_date="2026-08-11",
        mechanism={
            "claim": "The WMR London 4pm benchmark window concentrates institutional FX execution. A same-sign EURUSD and GBPUSD move across the completed fixing interval identifies a common US-dollar demand impulse; in elevated prior BTC variation, common dollar selling maps long BTC and common dollar buying maps short BTC for twelve hours.",
            "side": "long when both completed EURUSD and GBPUSD fixing-window log returns are positive; short when both are negative; disagreement or zero is ineligible",
            "external_support": {
                "official_source": "https://www.lseg.com/en/ftse-russell/benchmarks/wmr-fx-benchmarks",
                "official_methodology": "https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/wmr-fx-methodology.pdf",
                "economic_channel": "LSEG administers WMR London 4pm Closing Spot Rates as FX benchmarks; the rule tests an unverified post-fix transmission to BTC rather than reproducing the benchmark",
            },
            "why_distinct": "This uses common-sign price pressure only inside the completed London benchmark-fixing interval. It is not the full-session GBP-minus-EUR relative-beta candidate, broad daily dollar breadth, an ETF substitution, crypto price geometry, funding, OI, repair, or promoted control.",
            "why_suited_to_volatile_regimes": "BTC prior-24h realized variation must rank at least 0.65 causally",
            "why_low_gross9_overlap_is_plausible": "weekday DST-aware 16:10 Europe/London entries conditioned on an external ten-minute benchmark interval are absent from Gross9 primitives",
        },
        features={
            "fixing_interval": "exact EURUSD and GBPUSD Polygon 1m bars [15:55,16:05) Europe/London each Monday-Friday; all ten timestamps required for both symbols",
            "pair_return": "for each pair, log(close at 16:04/open at 15:55); both signs must agree and be nonzero",
            "availability": "16:05 Europe/London after both completed source paths",
            "btc_variation": "sqrt(sum squared log(close/open)) over exact BTCUSDT 1m bars in the prior 24 elapsed hours ending at decision",
            "btc_variation_rank": "strict-prior midrank against at most 252 previous valid weekday decision variations; minimum 126; current excluded; rank>=0.65",
            "missing_duplicate_or_nonpositive": "ineligible or source failure; no imputation",
        },
        clock={
            "decision": "each Monday-Friday 16:05 Europe/London with complete fixing and BTC source paths",
            "entry": "exact 16:10 Europe/London BTCUSDT 5m open", "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open", "split_crossing_action": "skip",
            "gross_exposure": 0.5, "funding_oi_premium": "not signal inputs; exact funding only after novelty passes", "no_imputation": True,
        },
        source_plan={
            "fx": {"table":"bars_polygon","symbols":["GBPUSD","EURUSD"],"interval":"1m","columns":["ts","open","close"],"read_only":True},
            "btc": {"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","close"],"read_only":True},
            "execution_price":"sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names":["no_volatility_gate","eurusd_only","one_session_stale_fix_impulse","direction_flip"],
            "diagnostic_controls_cannot_be_promoted":True,
        },
        research_boundary={
            "official_benchmark_definition_read":True, "database_metadata_only_opened_before_preregistration":True,
            "fx_values_used_to_select_rule":False, "candidate_incidence_opened":False,
            "postentry_return_or_pnl_opened":False, "gross9_rows_opened":False,
            "candidate_count":1, "grid":False, "repair_of_prior_candidate":False, "promoted_prior_control":False,
            "selection_basis":"independent scheduled institutional FX benchmark-flow transmission channel plus user-required high volatility",
        },
        stopping_rule="terminal first-failure sequence: source support, Gross9 novelty, strict economics; no pair, interval, consensus, threshold, side, hold, timing, volatility, subset, or control repair",
    )
    return {**core, "manifest_hash": canonical_hash(core)}

def validate(registration: dict[str, Any]) -> None:
    core={k:v for k,v in registration.items() if k!="manifest_hash"}
    if registration.get("manifest_hash") != canonical_hash(core) or registration != build():
        raise RuntimeError("HVLFX preregistration drift")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args()
    result=build(); validate(result); args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n"); print(args.output)
