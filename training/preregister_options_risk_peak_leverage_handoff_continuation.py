"""Outcome-blind preregistration for ORPLHC-6."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any

POLICY_ID="ORPLHC-6";DEFAULT_OUTPUT=Path("results/options_risk_peak_leverage_handoff_continuation_preregistration_2026-08-11.json")
def canonical_hash(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def build()->dict[str,Any]:
 core={
  "protocol_version":"options_risk_peak_leverage_handoff_continuation_v1","policy_id":POLICY_ID,"as_of_date":"2026-08-11","singleton":True,"outcomes_opened":False,"source_incidence_opened":False,"gross9_rows_opened":False,
  "public_source_basis":{"dvol":"https://insights.deribit.com/exchange-updates/dvol-deribit-implied-volatility-index/","perpetual_theory":"https://arxiv.org/html/2212.06888v5","open_interest":"https://arxiv.org/abs/2310.14973","claims_used":["DVOL is forward-looking 30-day implied volatility and an action gauge","premium is the high-frequency perpetual basis input","OI is outstanding leveraged inventory rather than direction"],"implementation_is_unpublished_adaptation":True},
  "mechanism":{"claim":"After an unusually large positive completed-hour DVOL repricing, the first immediately following negative DVOL body while DVOL remains elevated marks an options-risk peak. If the cooling hour still has an upper-tail premium displacement and increasing perpetual OI, directional risk discovery has handed off from options repricing to fresh leveraged basis demand; follow the premium direction for six hours.","side":"strict sign of the cooling hour premium close-minus-open displacement","why_distinct":"PVIAR uses same-hour premium acceleration during joint BVOL/DVOL expansion. FSVCCR uses actual funding settlement, BTC pre/post price continuation, and joint BVOL/DVOL cooling. CVTR requires two price-trend hours, stable OI, neutral funding, and simultaneous BVOL/DVOL contraction. ORPLHC uses a two-hour DVOL positive-to-negative peak transition, current premium direction, and current OI increase; it uses no BTC price, BVOL, funding, prior event set, or prior control.","why_suited_to_volatile_regimes":"the prior DVOL body magnitude is in its causal upper quartile and current DVOL level remains in its upper 40%","why_low_gross9_overlap_is_plausible":"a sparse two-hour options-to-perpetual handoff event is absent from Gross9 clocks"},
  "features":{"decision_grid":"each exact UTC hour D","dvol_peak":"previous completed hour normalized DVOL body>0, current completed hour normalized body<0, abs(previous body) strict-prior 720h midrank>=0.75, current close strict-prior rank>=0.60; minimum 672 and current excluded","premium_hour":"60 exact unique one-minute rows in [D-1h,D); close-minus-open strict nonzero; absolute displacement strict-prior rank>=0.60","oi_hour":"12 exact unique five-minute OI rows in [D-1h,D); log(last/first)>0","availability":"D after all constituent rows complete","no_imputation":True,"grid":False},
  "clock":{"decision":"exact completed-hour boundary D","entry":"exact BTCUSDT D+5m open","side":"cooling-hour premium displacement sign","hold":"6 elapsed hours","reservation":"global half-open; earliest event wins; exit first on equal open","split_crossing_action":"skip","gross_exposure":.5,"funding":"not a signal input; exact held settlements only after novelty","rv20":"q90 audit only after all economics pass"},
  "policy":{"history_hours":720,"minimum_history_hours":672,"prior_dvol_body_rank_min":.75,"current_dvol_level_rank_min":.60,"premium_displacement_rank_min":.60,"entry_delay_minutes":5,"hold_hours":6,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
  "stages":{"train":["2023-07-01T00:00:00Z","2024-01-01T00:00:00Z"],"test":["2024-01-01T00:00:00Z","2025-01-01T00:00:00Z"],"eval":["2025-01-01T00:00:00Z","2026-01-01T00:00:00Z"],"final":["2026-01-01T00:00:00Z","2026-08-01T00:00:00Z"]},
  "source_support_gates":{"minimum_events":{"train":8,"test":12,"eval":12,"final":8},"minority_side_share_min":.20,"max_month_share":.45},
  "novelty_gates":{"exact_entry_jaccard_max":.10,"candidate_near_6h_share_max":.35,"occupied_5m_bar_jaccard_max":.25,"absolute_signed_exposure_pearson_max":.35,"must_pass_before_economics":True},
  "economic_gates":{"absolute_return_positive":True,"cagr_to_strict_mdd_min":3.,"strict_mdd_max_pct":15.,"mean_gross_underlying_min_bp":20.,"weekly_signflip_one_sided_p_max":.10,"stress_absolute_return_positive":True,"stress_cagr_to_strict_mdd_min":2.5,"each_calendar_half_positive":True,"stop_on_first_failure":True,"accounting":"fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
  "post_stage_volatility_audit":{"prerequisite":"unchanged all-stage pass","rv20_q90_entry_filter":False,"minimum_q90_trades":8,"candidate_q90_absolute_return_positive":True,"identical_clock_forced_long_residual_positive":True},
  "diagnostic_controls":{"names":["no_prior_dvol_shock","no_current_dvol_cooling","no_dvol_level","no_premium_tail","no_oi_increase","one_hour_stale_features","direction_flip","forced_long"],"cannot_be_promoted":True},
  "source_plan":{"premium":{"table":"bars_binance_premium","symbol":"BTCUSDT","interval":"1m","read_only":True},"oi":{"table":"open_interest_binance","symbol":"BTCUSDT","period":"5m","read_only":True},"dvol":{"path":"data/options_crowding_deleveraging_relay_sources_v4_2023_2026/dvol_hourly.csv.gz","read_only":True},"execution_prices":"sealed until source and novelty pass"},
  "research_boundary":{"prior_options_and_leverage_outcomes_known":True,"exact_candidate_incidence_or_outcomes_known":False,"prior_event_sets_or_controls_reused":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"independent options-risk-peak to fresh-leverage handoff transition"},
  "stopping_rule":"Terminal first failure; no threshold, side, hold, clock, subset, source, or control repair."
 }
 return {**core,"manifest_hash":canonical_hash(core)}


def validate(v:dict[str,Any])->None:
 core={k:x for k,x in v.items() if k!="manifest_hash"}
 if v.get("manifest_hash")!=canonical_hash(core):raise RuntimeError("ORPLHC preregistration hash mismatch")


if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();v=build();validate(v);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(a.output)
