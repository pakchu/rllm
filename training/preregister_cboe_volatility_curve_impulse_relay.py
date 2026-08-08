"""Outcome-blind singleton preregistration for CVCIR-24."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT=Path("results/cboe_volatility_curve_impulse_relay_preregistration_2026-08-08.json")
SOURCE=Path("data/cboe_volatility_surface_2021_2026/cboe_volatility_surface_2021-01-01_2026-08-07.csv.gz")
SOURCE_MANIFEST=Path("data/cboe_volatility_surface_2021_2026/manifest.json")
SOURCE_SHA256="42eb1093f5167aec9c71a4733ab3451e40807c81dc7cb49568a6a0c634267ba0"
SOURCE_MANIFEST_SHA256="ec1dd33efcee29b75c80294fb594969cd1b12a9343fc40f888525db4400bc936"
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 core={"protocol_version":"cboe_volatility_curve_impulse_relay_v1","policy_id":"CVCIR-24","as_of_date":"2026-08-08","outcomes_opened":False,"source_incidence_opened":False,"singleton":True,
 "mechanism":{"claim":"Under elevated VIX, same-sign one-session changes in both the 9D/VIX and VIX/3M log slopes identify a coherent volatility-curve impulse: joint steepening transmits risk-off pressure to BTC (short), while joint cooling transmits relief (long) for 24 elapsed hours.","long":"delta log(VIX9D/VIX)<0 and delta log(VIX/VIX3M)<0","short":"both deltas >0","why_distinct":"CVCIR is a first-difference curve-coherence impulse with no tail or option-flow input. CVSRC was a rare joint tail-plus-term level crossing; CVTR and related clocks were static ranked level states.","why_low_gross9_overlap_is_plausible":"an official Cboe next-session curve-impulse clock is external to BTC activity and absent from Gross9"},
 "features":{"vix_strict_prior_midrank":"252 valid observations, minimum 126; current appended after rank","front":"log(VIX9D/VIX)","broad":"log(VIX/VIX3M)","deltas":"current minus immediately previous exact common source observation; no calendar fill","vix_rank_min":.60,"zero_or_nonfinite":"ineligible","no_imputation":True},
 "clock":{"trigger":"false-to-true onset of each same-sign impulse state; direct switch from opposite state is a new onset","entry":"first later exact common Cboe source date at 09:35 America/New_York","hold":"24 elapsed hours","reservation":"global half-open, entry equal to prior exit accepted","split_crossing_action":"skip","gross_exposure":.5,"btc_price_oi_funding_forbidden_as_signal_inputs":True},
 "stages":{"train":["2023-07-01T00:00:00Z","2024-01-01T00:00:00Z"],"test":["2024-01-01T00:00:00Z","2025-01-01T00:00:00Z"],"eval":["2025-01-01T00:00:00Z","2026-01-01T00:00:00Z"],"final":["2026-01-01T00:00:00Z","2026-08-01T00:00:00Z"]},
 "source_support_gates":{"minimum_events":{"train":8,"test":12,"eval":12,"final":8},"minority_side_share_min":.20,"max_month_share":.45},
 "novelty_gates":{"exact_entry_jaccard_max":.10,"candidate_near_6h_share_max":.35,"occupied_5m_jaccard_max":.25,"absolute_signed_exposure_pearson_max":.35,"must_pass_before_economics":True},
 "economic_gates":{"absolute_return_positive":True,"cagr_to_strict_mdd_min":3.,"strict_mdd_max_pct":15.,"mean_gross_underlying_min_bp":20.,"weekly_signflip_one_sided_p_max":.10,"stress_absolute_return_positive":True,"stress_cagr_to_strict_mdd_min":2.5,"each_calendar_half_positive":True,"accounting":"fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
 "controls":["no_vix_high","front_only","broad_only","no_onset","direction_flip"],"source_binding":{"panel":{"path":str(SOURCE),"sha256":SOURCE_SHA256},"manifest":{"path":str(SOURCE_MANIFEST),"sha256":SOURCE_MANIFEST_SHA256}},
 "research_boundary":{"candidate_incidence_opened":False,"candidate_post_entry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_cvsrc":False,"future_can_rank_repair_or_reselect":False},"stopping_rule":"Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no threshold, sign, hold, or subset repair."}
 return {**core,"manifest_hash":canonical_hash(core)}
def validate(p:dict[str,Any])->None:
 core={k:v for k,v in p.items() if k!="manifest_hash"}
 if p.get("manifest_hash")!=canonical_hash(core):raise RuntimeError("CVCIR preregistration canonical hash mismatch")
 for path,want in ((SOURCE,SOURCE_SHA256),(SOURCE_MANIFEST,SOURCE_MANIFEST_SHA256)):
  if hashlib.sha256(path.read_bytes()).hexdigest()!=want:raise RuntimeError(f"CVCIR frozen source hash mismatch: {path}")
def write(output:Path=DEFAULT_OUTPUT)->dict[str,Any]:
 p=build();validate(p);output.parent.mkdir(parents=True,exist_ok=True);b=json.dumps(p,indent=2,ensure_ascii=False,allow_nan=False)+"\n"
 if output.exists() and output.read_text()!=b:raise RuntimeError("CVCIR preregistration byte drift")
 output.write_text(b);return p
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);x=a.parse_args();print(json.dumps({"output":str(x.output),"manifest_hash":write(x.output)["manifest_hash"]},indent=2))
