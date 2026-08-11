"""Outcome-blind preregistration for HVWSI-Z20-24."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID="HVWSI-Z20-24"
DEFAULT_OUTPUT=Path("results/high_volatility_wilder_swing_index_zero_cross_relay_preregistration_2026-08-11.json")
def canonical_hash(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(
  protocol_version="high_volatility_wilder_swing_index_zero_cross_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
  mechanism={
   "claim":"During elevated realized variation, a completed four-hour Wilder Swing Index crossing zero identifies a change in comparative OHLC swing direction; follow the new sign for twenty-four hours.",
   "side":"long when Swing Index newly crosses above zero; short when it newly crosses below zero",
   "why_distinct":"Wilder Swing Index combines current versus prior close, current close versus open, prior close versus prior open, gap extremes, and current range through its N/R/K construction. It is not QStick, raw bar direction, close-only momentum, ATR or fixed channel, stochastic, RSI, volume, flow, OI, funding, fitted outcome, reused event set, or promoted control.",
   "why_suited_to_volatile_regimes":"the formula normalizes a signed two-bar OHLC impulse by the dominant gap/range geometry and completed trailing twenty-four-hour realized variation must rank in its causal upper 35%",
   "why_low_gross9_overlap_is_plausible":"four-hour sign crossings of Wilder's two-bar OHLC swing geometry are absent from Gross9 primitives"},
  external_basis={
   "origin":"J. Welles Wilder Jr., New Concepts in Technical Trading Systems; QuantConnect LEAN canonical implementation",
   "definition_source":"https://github.com/QuantConnect/Lean/blob/master/Indicators/WilderSwingIndex.cs",
   "documentation_source":"https://www.quantconnect.com/docs/v2/writing-algorithms/indicators/supported-indicators/wilder-swing-index",
   "fixed_definition":"documented limitMove=20; N=C-Cprev+0.5*(C-O)+0.25*(Cprev-Oprev); R selected by the largest of abs(H-Cprev), abs(L-Cprev), abs(H-L) using the LEAN branch formulas; K=max(abs(H-Cprev),abs(L-Cprev)); SI=50*(N/R)*(K/20); warmup two bars",
   "scale_invariance":"for positive finite R and K, changing a positive limitMove rescales SI but cannot change its sign or zero-cross events; the documented value 20 is fixed nonetheless",
   "selection_use":"official formula, documented limitMove, and published positive/up versus negative/down interpretation only; no incidence or outcomes"},
  features={
   "decision_grid":"every exact four-hour UTC boundary","source_bar":"exact aggregation of 240 coherent BTCUSDT one-minute rows [T-4h,T)",
   "n":"close-current minus prior close plus 0.5*(current close-current open) plus 0.25*(prior close-prior open)",
   "r":"LEAN branch formula selected by the first maximum among abs(high-prior close), abs(low-prior close), and abs(high-low)",
   "k":"max(abs(high-prior close),abs(low-prior close))","swing_index":"50*(n/r)*(k/20), finite with r nonzero",
   "event":"SI newly strictly above zero from not above gives +1; newly strictly below zero from not below gives -1",
   "variation":"sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive",
   "variation_rank":"strict-prior midrank over at most 180 earlier valid decisions, minimum 120, current excluded; rank>=0.65","no_imputation":True},
  clock={"feature_available":"four-hour boundary after completed source bar","entry":"exact BTCUSDT open five elapsed minutes later","hold":"24 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
  policy={"limit_move":20.0,"variation_hours":24,"variation_history_decisions":180,"minimum_variation_history_decisions":120,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":24,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
  diagnostic_controls={"names":["no_variation_gate","raw_close_change","one_bar_stale_cross","direction_flip"],"cannot_be_promoted":True},
  source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2020-01-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
  research_boundary={"canonical_wilder_swing_index_definition_read":True,"repository_wilder_swing_index_candidate_found":False,"prior_ohlc_impulse_candidate_outcomes_known":True,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"official Wilder Swing Index(limitMove=20) zero crossing under the requested high-variation regime"},
  stopping_rule="Terminal first failure; no limit move, formula branch, crossing, variation, side, hold, clock, subset, control, or other repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x:dict[str,Any])->None:
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVWSI-Z20-24 preregistration drift")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();x=build();validate(x);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(a.output)
