"""Outcome-blind preregistration for HVMACD-24."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID="HVMACD-24"
DEFAULT_OUTPUT=Path("results/high_volatility_macd_crossover_relay_preregistration_2026-08-11.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 c=copy.deepcopy(template.build());c.pop("manifest_hash")
 c.update(
  protocol_version="high_volatility_macd_crossover_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
  mechanism={"claim":"A canonical daily 12/26 exponential moving-average convergence-divergence line crossing its 9-day signal line during elevated realized variation marks a newly resolved trend transition that can persist for the next twenty-four hours.","side":"long on a strict upward MACD/signal crossover; short on a strict downward crossover","why_distinct":"HVWADX requires an already-strong four-hour directional-movement state and HVWRSI fades daily oscillator extremes. HVMACD trades only the first daily sign transition of the difference between two exponential trend speeds and its signal smoother, with no ADX/RSI level, return-tail threshold, OI, flow, volume, funding, fitted outcome, prior event, or promoted control.","why_suited_to_volatile_regimes":"completed twenty-day daily realized variation must rank in its causal upper 35%","why_low_gross9_overlap_is_plausible":"sparse daily trend-speed crossover clocks are absent from Gross9 primitives"},
  external_basis={"origin":"Gerald Appel, canonical Moving Average Convergence-Divergence convention","fixed_definition":"daily close EMA(12) minus EMA(26), EMA(9) signal, crossover direction","selection_use":"standard published periods and direction only; no candidate incidence or stage outcomes"},
  features={"decision_grid":"every calendar day at exact 00:00 UTC","daily_close":"last close of 1,440 exact coherent BTCUSDT one-minute rows [D-24h,D)","ema_seed":"arithmetic mean of the first exact period closes, followed by alpha=2/(period+1) recursion without reset","macd":"EMA(12)-EMA(26)","signal":"EMA(9) of finite MACD values using the same seeded recursion","crossover":"current MACD-signal strict sign differs from the immediately prior finite nonzero sign; zero values are ineligible","variation":"sqrt(sum squared latest twenty completed daily close-to-close log returns), finite strict positive","variation_rank":"strict-prior midrank over at most 180 earlier source-valid decisions, minimum 120, current excluded; rank>=0.65","no_imputation":True},
  clock={"feature_available":"00:00 UTC after completed daily close","entry":"exact BTCUSDT 00:05 UTC open","hold":"24 elapsed hours","reservation":"global half-open; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
  policy={"fast_ema_days":12,"slow_ema_days":26,"signal_ema_days":9,"variation_days":20,"variation_history_days":180,"minimum_variation_history_days":120,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":24,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
  diagnostic_controls={"names":["no_variation_gate","one_day_stale_crossover","direction_flip","forced_long"],"cannot_be_promoted":True},
  source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2020-01-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
  research_boundary={"canonical_macd_definition_read":True,"repository_macd_candidate_found":False,"prior_wilder_rsi_and_adx_outcomes_known":True,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"canonical trend-speed crossover under the requested high-variation regime"},
  stopping_rule="Terminal first failure; no EMA period, seed, crossover, variation, side, hold, clock, subset, threshold, or control repair.")
 return {**c,"manifest_hash":canonical_hash(c)}
def validate(x):
 core={k:v for k,v in x.items() if k!="manifest_hash"}
 if x.get("manifest_hash")!=canonical_hash(core) or x!=build():raise RuntimeError("HVMACD preregistration drift")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);z=a.parse_args();x=build();validate(x);z.output.parent.mkdir(parents=True,exist_ok=True);z.output.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");print(z.output)
