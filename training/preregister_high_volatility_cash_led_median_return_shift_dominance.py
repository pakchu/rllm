"""Outcome-blind preregistration for HVCVMSD-8."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_cross_structure_action_vote as gates

POLICY_ID="HVCVMSD-8";DEFAULT_OUTPUT=Path("results/high_volatility_cash_led_median_return_shift_dominance_preregistration_2026-08-18.json")
PRIMARY={"preregistration":{"path":"results/high_volatility_median_return_shift_relay_preregistration_2026-08-10.json","sha256":"29c87e3b08c500c2ffa0371c94849a5d84dcbf99ae0eb3f9df25922c3393863b"},"support":{"path":"results/high_volatility_median_return_shift_relay_support_2026-08-10.json","sha256":"41f8754ad340d062cb550151d2017c3567b12661671573fc7f9307129e091b46"},"gross9":{"path":"results/high_volatility_median_return_shift_relay_gross9_novelty_2026-08-10.json","sha256":"83fb5e444b8d19519c833a3f13021bb7ad89189bd5ab1a97d8403ba96130bdcf"},"clock":{"path":"data/high_volatility_median_return_shift_relay_clocks_2023_2026.csv.gz","sha256":"71adf3d170a68a26928d6d4600d8fb46046b7adddb64f464948355df41f98d5c"}}
def sha256(p:str|Path)->str:return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 g=gates.build();core={"protocol_version":"high_volatility_cash_led_median_return_shift_dominance_v1","policy_id":POLICY_ID,"as_of_date":"2026-08-18","singleton":True,"candidate_family":[POLICY_ID],"candidate_family_size":1,"source_incidence_opened":False,"outcomes_opened":False,"gross9_rows_opened":False,"primary_artifacts":PRIMARY,
 "mechanism":{"claim":"When spot and perpetual exhibit the same robust median-return location shift but the spot shift has larger absolute magnitude, unlevered cash is leading broad price discovery; follow the common cash-led shift for eight hours during frozen high variation.","side":"immutable HVMRSR-8 side, emitted only when spot median shift has the same strict sign and strictly larger absolute magnitude","why_distinct":"HVMRSR is perpetual-only. Cash/perpetual candidates use endpoint returns, ranges, basis, flow, or lead-lag. HVCVMSD compares the signed median of 240 one-minute open-to-close returns in each half independently on both actual venues and uses no endpoint direction, volume, flow, OI, funding, premium, options, fitted outcome, or promoted control.","why_low_gross9_overlap_is_plausible":"cash-led robust return-location magnitude dominance at irregular HVMRSR events is absent from Gross9"},
 "construction":{"primary":"immutable HVMRSR-8 event, side, entry, and hold","spot_window":"480 exact coherent bars_binance_spot BTCUSDT one-minute rows [D-8h,D)","spot_shift":"median rows 240..479 log(close/open) minus median rows 0..239","confirmation":"spot_shift strict nonzero, sign equals immutable perpetual median-shift side, and abs(spot_shift)>abs(perpetual_shift)","entry":"immutable D+5m","hold":"8 elapsed hours","additional_or_tuned_thresholds":"none","weights":"none"},
 "stages":g["stages"],"source_support_gates":g["source_support_gates"],"gross9_novelty_gates":g["gross9_novelty_gates"],"economic_gates":g["economic_gates"],
 "source_plan":{"table":"bars_binance_spot","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"read_after_preregistration":True,"execution_prices":"sealed until source and Gross9 pass"},
 "research_boundary":{"HVMRSR_train_single_half_failure_known":True,"prior_HVCVMRS_train_single_half_failure_known":True,"HVMRSR_test_eval_final_outcomes_known":False,"exact_cash_led_dominance_incidence_or_outcomes_known":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"classification":"exploratory discovery; not fresh confirmatory evidence","selection_basis":"cash-led magnitude dominance in robust within-auction location shift"},
 "stopping_rule":"source, Gross9, train/test/eval/final; terminal first failure; no primary, spot source, half partition, median, cash dominance, side, clock, hold, subset, threshold, or control repair"};return {**core,"manifest_hash":canonical_hash(core)}
def validate(x:dict[str,Any])->None:
 if x!=build():raise RuntimeError("HVCVMSD-8 preregistration drift")
 for artifact in PRIMARY.values():
  if sha256(artifact["path"])!=artifact["sha256"]:raise RuntimeError("HVCVMSD-8 primary drift")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();x=build();validate(x);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(a.output)
