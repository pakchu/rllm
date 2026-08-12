"""Outcome-blind preregistration for HVOID-8."""
from __future__ import annotations

import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID="HVOID-8"
DEFAULT_OUTPUT=Path("results/high_volatility_oi_unwind_directional_dominance_relay_preregistration_2026-08-13.json")

def canonical_hash(value:Any)->str:return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(
  protocol_version="high_volatility_oi_unwind_directional_dominance_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-13",
  mechanism={"claim":"During elevated BTC variation, if completed five-minute price displacement occurring specifically while open interest contracts is directionally coherent and dominates displacement occurring while open interest expands, forced inventory reduction rather than new leverage is driving price discovery; follow the unwind direction for eight elapsed hours.","side":"strict sign of the sum of five-minute BTC returns paired with negative open-interest changes","why_distinct":"HVOID attributes every completed price return to contemporaneous OI expansion or contraction and compares their directional sums. Endpoint OI purge uses only net stock change; OI-price sign concordance counts agreement without separating unwind displacement; coactivity discards direction; OI-lead sponsorship shifts OI changes forward. HVOID uses no endpoint OI sign gate, onset, funding, premium, flow, fitted outcome, prior event, or control.","why_suited_to_volatile_regimes":"both BTC variation and gross OI activity occupy causal upper regimes while unwind displacement must dominate leverage-building displacement","why_low_gross9_overlap_is_plausible":"offset eight-hour bar-level OI unwind attribution clocks are absent from Gross9 primitives"},
  features={"decision_grid":"exact 02:00, 10:00, and 18:00 UTC boundaries","price_path":"96 exact coherent BTCUSDT five-minute bars aggregated from 480 one-minute rows [D-8h,D)","oi_path":"97 exact positive BTCUSDT period=5m observations from D-8h through D, each available no later than D","price_return":"r_i=log(close_i/open_i) for each completed five-minute bar","oi_change":"d_i=log(OI_i/OI_(i-1)) aligned to the same five-minute interval","unwind_return":"U=sum r_i where d_i<0, finite strict nonzero","build_return":"B=sum r_i where d_i>0","unwind_directional_dominance":"abs(U)/(abs(U)+abs(B)), finite with strict-positive denominator; must be >=2/3","direction_confirmation":"completed eight-hour return has the same strict sign as U","gross_oi_activity":"sum abs(d_i), finite strict positive","realized_variation":"sqrt(sum r_i^2), finite strict positive","causal_ranks":"strict-prior midranks of gross OI activity and realized variation over at most 270 earlier source-valid decisions, minimum 180, current excluded","eligibility":"dominance>=2/3, direction confirmation, gross-OI-activity rank>=0.60, and variation rank>=0.65","no_imputation":True},
  clock={"feature_available":"offset eight-hour boundary after both completed paths","entry":"exact BTCUSDT open five elapsed minutes later","side":"strict sign of unwind_return","hold":"8 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":0.5,"funding":"not a signal input; exact settlements only after novelty passes"},
  policy={"bar_minutes":5,"path_bars":96,"decision_hours":8,"dominance_min":2/3,"history_decisions":270,"minimum_history_decisions":180,"oi_activity_rank_min":0.60,"variation_rank_min":0.65,"entry_delay_minutes":5,"hold_hours":8,"leverage":0.5,"base_cost_per_notional_side":0.0006,"stress_cost_per_notional_side":0.001},
  diagnostic_controls={"names":["no_oi_activity_gate","no_variation_gate","no_dominance_gate","direction_flip","forced_long"],"cannot_be_promoted":True},
  source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2023-05-25T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"open_interest":{"table":"open_interest_binance","symbol":"BTCUSDT","period":"5m","columns":["timestamp","sum_open_interest"],"same_window":True,"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
  research_boundary={"bar_level_oi_unwind_attribution_selected":True,"repository_exact_oi_unwind_directional_dominance_candidate_found":False,"adjacent_endpoint_purge_sign_concordance_coactivity_and_lead_candidates_known":True,"adjacent_candidate_outcomes_used_to_set_formula_side_hold_clock_or_threshold":False,"prior_event_sets_reused":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"fixed contemporaneous OI-contraction return attribution and two-thirds directional dominance under elevated variation"},
  stopping_rule="Terminal first failure; no path, OI alignment, attribution, dominance, ranks, side, hold, clock, subset, threshold, or control repair."
 )
 return {**c,"manifest_hash":canonical_hash(c)}

def validate(v:dict[str,Any])->None:
 core={k:x for k,x in v.items() if k!="manifest_hash"}
 if v.get("manifest_hash")!=canonical_hash(core) or v!=build():raise RuntimeError("HVOID preregistration drift")

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();v=build();validate(v);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(a.output)
