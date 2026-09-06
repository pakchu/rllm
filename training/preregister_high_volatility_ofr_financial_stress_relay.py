"""Outcome-blind preregistration for HVOFSR-24."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any

if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_high_volatility_energy_technology_spillover_relay as template

DEFAULT_OUTPUT=Path("results/high_volatility_ofr_financial_stress_relay_preregistration_2026-08-12.json")
SOURCE_URL="https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(protocol_version="high_volatility_ofr_financial_stress_relay_v1",policy_id="HVOFSR-24",as_of_date="2026-08-12",gross9_rows_opened=False,
 mechanism={"claim":"The U.S. Treasury Office of Financial Research publishes a daily market-based global Financial Stress Index built from 33 variables spanning credit, equity valuation, funding, safe assets and volatility. An unusually large completed daily stress change is a broad systemic risk repricing; during elevated completed BTC variation, rising stress maps short BTC and falling stress maps long BTC for twenty-four hours.","side":"negative strict sign of the completed OFR FSI observation-to-observation change","external_support":{"official_page":"https://www.financialresearch.gov/financial-stress-index/","official_methodology":"https://www.financialresearch.gov/working-papers/2023/06/27/transition-alternative-reference-rates-in-ofr-financial-stress-index/","reported_definition":"daily market-based snapshot constructed from 33 variables; positive levels denote above-average stress","official_latency":"published data are current from two business days prior","inference_disclosure":"change-tail event, BTC variation gate, directional map and 24-hour hold are preregistered adaptations"},"why_distinct":"HVNFCI uses weekly Chicago Fed real-time-vintage financial conditions and a 72-hour weekend clock. HVOFSR uses the Treasury OFR daily global stress composite, its official two-business-day delay, a daily change shock and a 24-hour clock. Existing Cboe, Treasury-yield, dollar, equity and crypto candidates use narrower objects.","why_suited_to_volatile_regimes":"both the systemic-stress change and completed BTC variation must lie in causal upper tails","why_low_gross9_overlap_is_plausible":"a delayed federal daily systemic-stress release clock is absent from Gross9"},
 features={"source":"official OFR downloadable FSI CSV","observation":"finite total OFR FSI value on source observation date D","change":"FSI[D]-FSI[immediately previous exact source row], strict nonzero","stress_change_rank":"strict-prior midrank of abs(change) over at most 252 valid changes, minimum 126, current excluded; rank>=0.70","availability":"22:00 UTC on the date of the second subsequent exact OFR source row, conservatively after the official two-business-day publication lag and US cash close","btc_variation":"sqrt(sum squared exact BTCUSDT 1m log(close/open)) over 24 elapsed hours ending at availability","btc_variation_rank":"strict-prior midrank over at most 270 valid decisions, minimum 180, current excluded; rank>=0.65","missing_revision_or_duplicate":"duplicate date, schema drift, nonfinite value, missing BTC minute or nonmonotone dates reject; no imputation"},
 clock={"decision":"22:00 UTC on second subsequent OFR source-row date","entry":"exact BTCUSDT 22:05 UTC five-minute open","hold":"24 elapsed hours","reservation":"global half-open; exit first on equal open","split_crossing_action":"skip","gross_exposure":.5,"funding_oi_premium":"not signal inputs; exact funding only after novelty passes","no_imputation":True},
 policy={"stress_prior_observations":252,"stress_prior_minimum":126,"stress_midrank_min":.70,"variation_prior_observations":270,"variation_prior_minimum":180,"variation_midrank_min":.65,"publication_lag_source_rows":2,"decision_hour_utc":22,"entry_delay_minutes":5,"hold_hours":24,"gross_exposure":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
 source_plan={"ofr":{"provider":"U.S. Treasury Office of Financial Research","url":SOURCE_URL,"download_after_preregistration":True,"expected_frequency":"daily business observations","total_index_only":True},"btc_1m":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","close"],"read_only":True},"execution_price":"sealed until source support and Gross9 novelty pass"},
 diagnostic_controls={"names":["no_btc_volatility_gate","no_stress_change_tail","one_observation_stale_change","fsi_level_sign","direction_flip","same_clock_forced_long"],"diagnostic_controls_cannot_be_promoted":True},
 research_boundary={"official_definition_methodology_latency_and_transport_opened":True,"ofr_fsi_values_dates_changes_ranks_or_incidence_opened":False,"repository_exact_ofr_fsi_candidate_found":False,"prior_nfci_and_macro_outcomes_known":True,"prior_event_sets_or_controls_reused":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"independent official daily systemic-stress object with stated causal publication lag"},
 stopping_rule="terminal first-failure sequence: source contract and byte reproduction, source support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90; no source, latency, threshold, side, hold, clock, subset or control repair")
 return {**c,"manifest_hash":canonical_hash(c)}

def validate(x:dict[str,Any])->None:
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x!=build() or x.get("manifest_hash")!=canonical_hash(core):raise RuntimeError("HVOFSR preregistration drift")
 if x["outcomes_opened"] is not False or x["source_incidence_opened"] is not False:raise RuntimeError("HVOFSR boundary drift")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();r=build();validate(r);a.output.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(a.output)
