"""Outcome-blind preregistration for HVKCLL-12."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVKCLL-12"
DEFAULT_OUTPUT = Path("results/high_volatility_korean_cash_lead_lag_relay_preregistration_2026-08-11.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build()); contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_korean_cash_lead_lag_relay_v1", policy_id=POLICY_ID, as_of_date="2026-08-11",
        mechanism={
            "claim":"During an already volatile BTC block, an unusually strong five-minute Upbit-to-Binance lead relative to the reverse direction identifies regional cash price discovery awaiting global perpetual absorption. When completed venue returns agree, follow their common direction for twelve hours at the fresh leadership onset.",
            "side":"common strict nonzero sign of completed four-hour Upbit KRW-BTC and Binance BTCUSDT perpetual returns",
            "why_distinct":"Korean cash leadership previously compared return magnitudes; Korean variance leadership compared quadratic variation; kimchi work used FX-adjusted premium changes. HVKCLL uses neither level, FX, premium, flow, volume nor magnitude dominance: it compares opposite one-step cross-venue return correlations on exact aligned paths.",
            "why_suited_to_volatile_regimes":"completed Binance four-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible":"fresh Korean-cash lead-lag onsets on four-hour boundaries are absent from Gross9 primitives",
        },
        external_basis={
            "paper":"https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003018218",
            "paper_title":"Correlation Analysis of Time Differences between Domestic and International Bitcoin Prices",
            "support":"the study uses minute-level domestic and international Bitcoin prices and reports that lagged relationships can emerge in shorter samples and differ across sharp-move regimes",
            "selection_use":"fixed one-five-minute lead asymmetry only; rank, confirmation, onset, side and holding application are untested; no incidence or outcomes",
        },
        features={
            "decision_grid":"every completed four-hour UTC boundary",
            "sources":"240 exact aligned coherent one-minute rows from bars_upbit KRW-BTC and bars_binance BTCUSDT over [T-4h,T)",
            "five_minute_returns":"48 exact nonoverlapping groups per venue; log of each group fifth-minute close divided by first-minute open",
            "upbit_leads_binance":"Pearson corr(Upbit_return[t], Binance_return[t+1]) over 47 pairs",
            "binance_leads_upbit":"Pearson corr(Binance_return[t], Upbit_return[t+1]) over 47 pairs",
            "leadership_advantage":"upbit_leads_binance minus binance_leads_upbit; both pair variances strict positive; finite strict positive gate",
            "leadership_rank":"strict-prior midrank over at most 270 earlier source-valid decisions, minimum 180, current excluded; rank>=0.80",
            "directions":"log(last five-minute close/first five-minute open) independently by venue; both strict nonzero and same sign",
            "binance_variation":"sqrt(sum squared 48 Binance five-minute log open-to-close returns), finite strict positive",
            "variation_rank":"strict-prior 270/180 midrank, current excluded; rank>=0.65",
            "eligible_state":"positive leadership advantage, leadership and variation ranks pass, and venue return directions agree",
            "onset":"eligible now and immediately prior source-valid decision ineligible; missing prior cannot trigger",
            "no_imputation":True,
        },
        clock={"feature_available":"completed four-hour boundary","entry":"exact BTCUSDT perpetual boundary+5m open","hold":"12 elapsed hours","reservation":"global half-open; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
        policy={"five_minute_bars":48,"leadership_history_decisions":270,"minimum_leadership_history_decisions":180,"leadership_rank_min":.80,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":12,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
        diagnostic_controls={"names":["no_leadership_gate","no_variation_gate","binance_leads_upbit","one_bar_stale_onset","direction_flip","forced_long"],"cannot_be_promoted":True},
        source_plan={"upbit":{"table":"bars_upbit","symbol":"KRW-BTC","interval":"1m","columns":["ts","open","high","low","close"]},"binance":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"]},"window":["2023-04-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True,"execution_price":"sealed until source support and Gross9 novelty pass"},
        research_boundary={"domestic_international_lag_study_read":True,"repository_exact_upbit_binance_lead_lag_candidate_found":False,"prior_korean_return_variance_premium_outcomes_known":True,"prior_event_sets_reused":False,"prior_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_korean_cash_or_variance_leadership":False,"promoted_prior_control":False,"selection_basis":"published regime-varying domestic/international Bitcoin lag relationship applied as a regional-cash price-discovery mechanism"},
        stopping_rule="Terminal first failure; no venue, block, bar aggregation, lag, correlation, rank, direction agreement, onset, side, hold, clock, subset, control, or other repair.",
    )
    return {**contract, "manifest_hash":canonical_hash(contract)}

def validate(value: dict[str, Any]) -> None:
    core={k:v for k,v in value.items() if k!="manifest_hash"}
    if value.get("manifest_hash")!=canonical_hash(core) or value!=build(): raise RuntimeError("HVKCLL-12 preregistration drift")

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args()
    value=build(); validate(value); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n"); print(args.output)
