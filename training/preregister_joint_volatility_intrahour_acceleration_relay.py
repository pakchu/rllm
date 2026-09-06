"""Outcome-blind preregistration for JVIAR-6."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT=Path("results/joint_volatility_intrahour_acceleration_relay_preregistration_2026-08-08.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 core={"protocol_version":"joint_volatility_intrahour_acceleration_relay_v1","policy_id":"JVIAR-6","as_of_date":"2026-08-08","outcomes_opened":False,
 "mechanism":{"claim":"When BVOL and DVOL both expand during a completed hour, a large first-half BTC move followed by a larger same-direction second-half move identifies accelerating cross-venue price discovery; the direction should relay for six hours.","side":"common sign of the two completed half-hour BTC returns","why_distinct":"JVIAR requires same-direction intrahour acceleration under joint volatility expansion. OLIAH required opposite-direction absorption plus OI contraction; OIFAR used a full-hour shock and OI flush and traded reversal; CVVIB used fixed four-hour compression blocks.","why_low_gross9_overlap_is_plausible":"joint options-volatility ignition plus exact within-hour acceleration geometry is absent from Gross9"},
 "clock":{"decision":"T after a completed UTC hour","volatility_ignition":"normalized completed-hour BVOL and DVOL bodies are both strictly positive","price_source":"60 exact BTCUSDT 1m bars in [T-1h,T), split into minute offsets 0..29 and 30..59","first_half_shock":"nonzero absolute first-half return >= strictly-prior 720h q60 with 672 observations","directional_acceleration":"second-half return has the same nonzero sign and absolute magnitude >= first-half absolute magnitude","trigger":"false-to-true onset with prior source-valid consecutive hour","entry":"exact BTCUSDT T+5m open","side":"common half-hour direction","hold":"6 elapsed hours","reservation":"global half-open, exit first on equal open","funding":"not a signal input; exact settlements only for later PnL","oi":"not a signal input","no_imputation":True},
 "policy":{"prior_hours":720,"prior_min_hours":672,"first_half_absolute_return_quantile":.60,"minimum_acceleration_ratio":1.,"entry_delay_minutes":5,"hold_hours":6,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
 "stages":{"train":["2023-07-01T00:00:00Z","2024-01-01T00:00:00Z"],"test":["2024-01-01T00:00:00Z","2025-01-01T00:00:00Z"],"eval":["2025-01-01T00:00:00Z","2026-01-01T00:00:00Z"],"final":["2026-01-01T00:00:00Z","2026-08-01T00:00:00Z"]},
 "source_support_gates":{"minimum_events":{"train":8,"test":12,"eval":12,"final":8},"minority_side_share_min":.20,"max_month_share":.45},
 "novelty_gates":{"exact_entry_jaccard_max":.10,"candidate_near_6h_share_max":.35,"occupied_5m_jaccard_max":.25,"absolute_signed_exposure_pearson_max":.35,"must_pass_before_economics":True},
 "economic_gates":{"absolute_return_positive":True,"cagr_to_strict_mdd_min":3.,"strict_mdd_max_pct":15.,"mean_gross_underlying_min_bp":20.,"weekly_signflip_one_sided_p_max":.10,"stress_absolute_return_positive":True,"stress_cagr_to_strict_mdd_min":2.5,"each_calendar_half_positive":True,"stop_on_first_failure":True,"future_can_rank_repair_or_reselect":False,"accounting":"fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
 "source_plan":{"volatility":"reuse hash-bound OCDR-12C BVOL/DVOL snapshot through 2026-08-01","intrahour_price":"reuse hash-bound OLIAH completed-hour snapshot through 2026-08-01","execution_price":"sealed until source-support and Gross9 novelty pass"},
 "diagnostic_controls":{"names":["no_joint_expansion","no_first_half_tail","no_same_direction","no_acceleration","direction_flip"],"diagnostic_controls_cannot_be_promoted":True},
 "research_boundary":{"related_intrahour_candidate_incidence_known":True,"related_candidate_outcomes_used_to_define_jviar":False,"jviar_candidate_incidence_opened":False,"jviar_post_entry_return_or_pnl_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False}}
 return {**core,"manifest_hash":canonical_hash(core)}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();a.output.write_text(json.dumps(build(),indent=2,ensure_ascii=False)+"\n");print(a.output)
