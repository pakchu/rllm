"""Outcome-blind preregistration for HVDHSM-12."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID="HVDHSM-12"
DEFAULT_OUTPUT=Path("results/high_volatility_daily_half_session_momentum_relay_preregistration_2026-08-11.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def build() -> dict[str,Any]:
    core=copy.deepcopy(template.build());core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_daily_half_session_momentum_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
        mechanism={
            "claim":"Published cryptocurrency evidence reports intraday return predictability, including momentum, while broader intraday-momentum evidence finds stronger effects on volatile days. On each UTC day, follow the sign of the completed 00:00-12:00 BTC return for twelve hours only when the causally prior 24-hour realized variation ranks in its upper 35%.",
            "side":"long when the completed UTC first-half return is positive; short when negative; zero is ineligible",
            "external_support":{
                "crypto_intraday_predictability":"https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4080253",
                "volatile_day_intraday_momentum":"https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866",
                "adaptation":"the fixed UTC half-day BTC relay is an untested adaptation, not a claimed replication",
            },
            "why_distinct":"This is a single daily information-diffusion handoff from the completed UTC first half to the second half. It is not a weekly momentum rule, one-hour reversal, oscillator crossing, jump threshold, external-source relay, funding/OI/basis rule, repair, or promoted control.",
            "why_suited_to_volatile_regimes":"the exact trailing 24-hour BTC realized variation at 12:00 UTC must rank at least 0.65 using strict-prior states",
            "why_low_gross9_overlap_is_plausible":"one causal 12:05 UTC entry per selected day and a daily half-session primitive are absent from Gross9",
        },
        features={
            "source":"BTCUSDT bars_binance interval=1m",
            "first_half":"720 exact distinct coherent finite positive minute rows [D 00:00,D 12:00) UTC; return=log(close at 11:59/open at 00:00)",
            "btc_variation":"sqrt(sum squared exact minute log(close/open)) over [D-1 12:00,D 12:00)",
            "variation_rank":"strict-prior midrank over at most 270 previous valid daily 12:00 states, minimum 180, current excluded; rank>=0.65",
            "availability":"D 12:00 UTC after both completed paths", "no_imputation":True,
        },
        clock={
            "decision":"each valid UTC day D at 12:00","entry":"exact BTCUSDT D 12:05 UTC open","hold":"12 elapsed hours",
            "side":"sign of completed first-half return","reservation":"chronological global half-open; exit first on equal open",
            "gross_exposure":0.5,"funding":"not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "variation_history_decisions":270,"minimum_variation_history_decisions":180,"variation_rank_min":0.65,
            "entry_delay_minutes":5,"hold_hours":12,"leverage":0.5,"base_cost_per_notional_side":0.0006,"stress_cost_per_notional_side":0.001,
        },
        diagnostic_controls={
            "names":["no_variation_gate","late_six_hour_direction","one_day_stale_first_half","direction_flip"],"cannot_be_promoted":True,
        },
        source_plan={
            "btc":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"read_after_preregistration":True},
            "window":["2022-12-01T00:00:00Z","2026-08-01T00:00:00Z"],"execution_prices":"sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "published_abstracts_read":True,"repository_daily_half_session_candidate_found":False,"candidate_source_values_opened":False,
            "prior_event_sets_reused":False,"prior_candidate_economic_outcomes_used_to_set_formula_side_hold_or_clock":False,
            "candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,
            "candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,
            "selection_basis":"published intraday information diffusion plus user-required high realized volatility",
        },
        stopping_rule="Terminal first failure; no UTC boundary, half-session definition, variation, side, hold, clock, subset, threshold, or control repair.",
    )
    return {**core,"manifest_hash":canonical_hash(core)}

def validate(value:dict[str,Any])->None:
    core={k:v for k,v in value.items() if k!="manifest_hash"}
    if value.get("manifest_hash")!=canonical_hash(core) or value!=build(): raise RuntimeError("HVDHSM preregistration drift")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args();value=build();validate(value)
    args.output.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(args.output)
