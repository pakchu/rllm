"""Outcome-blind preregistration for OIDCR-8."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT=Path("results/oi_divergence_contradiction_relay_preregistration_2026-08-08.json")
SOURCE=Path("data/oi_premium_asymmetric_volatility_relay_sources_2023_2026/signal_features.csv.gz")
PARENT=Path("configs/live/oi_divergence_pullback_range_rsi_h96_s6_candidate.json")
HASHES={str(SOURCE):"4e3bc8ee099470068ef738852e6a11c3778ba2cf6c8739e1947efcd902e04233",str(PARENT):"6533650bb6800308762dc02f310dbfe7dbd59c8a217d55305f6c5388eb2a480b"}
THRESHOLDS={"oi_abs":.8954018630586817,"return_z_abs":.7389570664259131,"range_vol":.04008415457867338,"rsi_abs":.04507656773717145}
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 core={"protocol_version":"oi_divergence_contradiction_relay_v1","policy_id":"OIDCR-8","as_of_date":"2026-08-08","outcomes_opened":False,"source_incidence_opened":False,"singleton":True,
 "mechanism":{"claim":"In a high-range-volatility state, positive OI-minus-price divergence during a negative return shock is inventory absorption and maps long; the exact sign mirror is fragile price appreciation unsupported by OI and maps short.","side_rule":"sign(oi_minus_px_4h_z) when return_zscore_48 has the opposite sign and all symmetric absolute gates pass","why_distinct":"one symmetric OI-price contradiction law, not the mixed OI/premium state machine OIPAR-ASYM and not a terminal control or direction flip","why_july_like":"explicit high range-volatility gate targets July-like shock and pullback conditions"},
 "features":{"oi_minus_px_4h_z":"completed 4h OI log change minus BTC close return, rolling 288-bar z-score","return_zscore_48":"completed 4h price return z-score","range_vol":"completed-bar range-volatility state","rsi_norm":"completed-bar normalized RSI","availability":"completed 5m bar t; enter t+1 open","no_imputation":True},
 "symmetric_thresholds":THRESHOLDS,
 "states":{"long":"oi_z>=+oi_abs AND return_z<=-return_z_abs AND range_vol>=gate AND rsi<=-rsi_abs","short":"oi_z<=-oi_abs AND return_z>=+return_z_abs AND range_vol>=gate AND rsi>=+rsi_abs","conflict":"impossible under strict mirrored signs; skip defensively"},
 "clock":{"entry":"next 5m open","hold":"8 elapsed hours","stride":"30 elapsed minutes at frozen stride-1 UTC offset","reservation":"global half-open","split_crossing_action":"skip","gross_exposure":.5},
 "stages":{"train":["2023-07-01T00:00:00Z","2024-01-01T00:00:00Z"],"test":["2024-01-01T00:00:00Z","2025-01-01T00:00:00Z"],"eval":["2025-01-01T00:00:00Z","2026-01-01T00:00:00Z"],"final":["2026-01-01T00:00:00Z","2026-08-01T00:00:00Z"]},
 "source_support_gates":{"minimum_events":{"train":8,"test":12,"eval":12,"final":8},"minority_side_share_min":.2,"max_month_share":.45},
 "novelty_gates":{"exact_entry_jaccard_max":.1,"candidate_near_6h_share_max":.35,"occupied_5m_jaccard_max":.25,"absolute_signed_exposure_pearson_max":.35,"must_pass_before_economics":True},
 "economic_gates":{"absolute_return_positive":True,"cagr_to_strict_mdd_min":3.,"strict_mdd_max_pct":15.,"mean_gross_underlying_min_bp":20.,"weekly_signflip_one_sided_p_max":.1,"stress_absolute_return_positive":True,"stress_cagr_to_strict_mdd_min":2.5,"each_calendar_half_positive":True,"stop_on_first_failure":True,"accounting":"fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
 "diagnostic_controls":{"names":["no_range_vol_gate","no_rsi_gate","no_return_contradiction","one_bar_stale_features","direction_flip"],"diagnostic_controls_cannot_be_promoted":True},"source_bindings":HASHES,
 "research_boundary":{"prior_long_component_outcomes_known":True,"symmetric_candidate_outcomes_known":False,"candidate_incidence_opened":False,"post_entry_outcomes_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False},"stopping_rule":"terminal first failure; no repair"}
 return {**core,"manifest_hash":canonical_hash(core)}
def validate(x:dict[str,Any])->None:
 if x["manifest_hash"]!=canonical_hash({k:v for k,v in x.items() if k!="manifest_hash"}):raise RuntimeError("OIDCR hash mismatch")
 for p,h in HASHES.items():
  if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError(f"OIDCR source drift: {p}")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();x=build();validate(x);a.output.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(a.output)
