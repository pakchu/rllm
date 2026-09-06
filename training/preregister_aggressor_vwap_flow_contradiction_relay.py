"""Outcome-blind preregistration for AVFCR-12."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

POLICY_ID="AVFCR-12"
DEFAULT_OUTPUT=Path("results/aggressor_vwap_flow_contradiction_relay_preregistration_2026-08-09.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def build()->dict[str,Any]:
 core={
  "protocol_version":"aggressor_vwap_flow_contradiction_relay_v1","policy_id":POLICY_ID,"as_of_date":"2026-08-09","outcomes_opened":False,"source_incidence_opened":False,"gross9_rows_opened":False,"singleton":True,
  "mechanism":{"claim":"During the completed final six hours of a UTC day, the average execution price of aggressive buys versus aggressive sells reveals which side moved through the price path. If that VWAP-separation sign opposes aggregate signed taker notional, the lower-notional side exercised disproportionate price control; follow the execution-price-dominant side for twelve hours.","side":"sign(log(aggressive_buy_vwap/aggressive_sell_vwap))","why_distinct":"AVFCR uses separate buy and residual-sell base/quote VWAPs, not close VWAP, aggregate imbalance confirmation, impact per volume, return magnitude, run persistence, funding, volatility gating, or a fitted model.","volatile_market_target":"disproportionate price control by the minority aggressor side is a latent impact mechanism; RV20 q90 is a later stress audit only","why_low_gross9_overlap_is_plausible":"one source-conditioned twelve-hour UTC-day position"},
  "features":{"window":"exact 360 BTCUSDT one-minute bars [D-6h,D) before daily 00:00 UTC D","buy_base":"sum(taker_buy_base)","buy_quote":"sum(taker_buy_quote)","sell_base":"sum(volume-taker_buy_base)","sell_quote":"sum(quote_asset_volume-taker_buy_quote)","buy_vwap":"buy_quote/buy_base","sell_vwap":"sell_quote/sell_base","separation":"log(buy_vwap/sell_vwap), strict nonzero","flow":"(2*buy_quote-total_quote)/total_quote, strict nonzero","contradiction":"sign(separation)=-sign(flow)","source_valid":"exact unique minute grid; finite coherent positive OHLC; nonnegative volumes/counts; 0<=taker fields<=totals; all four buy/sell totals positive; no imputation"},
  "rv20_stress_slice":{"rv20":"sqrt(365*mean(r_d^2)) over exact daily returns t-20 through t-1","threshold":"numpy linear q90 over 756 strictly prior available RV20 observations; current excluded","entry_filter":False,"future_use":"only after all sequential full-calendar stages pass"},
  "clock":{"decision":"daily 00:00 UTC after [D-6h,D) completes","entry":"exact BTCUSDT D+5m open","hold":"12 elapsed hours","reservation":"global half-open; exit first on equal open","split_crossing_action":"skip","gross_exposure":.5,"funding":"not an input; exact realized funding only after novelty"},
  "stages":{"train":["2023-07-01T00:00:00Z","2024-01-01T00:00:00Z"],"test":["2024-01-01T00:00:00Z","2025-01-01T00:00:00Z"],"eval":["2025-01-01T00:00:00Z","2026-01-01T00:00:00Z"],"final":["2026-01-01T00:00:00Z","2026-08-01T00:00:00Z"]},
  "source_support_gates":{"minimum_events":{"train":8,"test":12,"eval":12,"final":8},"minority_side_share_min":.2,"max_month_share":.45},
  "novelty_gates":{"exact_entry_jaccard_max":.1,"candidate_near_6h_share_max":.35,"occupied_5m_bar_jaccard_max":.25,"absolute_signed_exposure_pearson_max":.35,"must_pass_before_economics":True},
  "economic_gates":{"absolute_return_positive":True,"cagr_to_strict_mdd_min":3.,"strict_mdd_max_pct":15.,"mean_gross_underlying_min_bp":20.,"weekly_signflip_one_sided_p_max":.1,"stress_absolute_return_positive":True,"stress_cagr_to_strict_mdd_min":2.5,"each_calendar_half_positive":True,"stop_on_first_failure":True,"accounting":"fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
  "post_stage_volatility_audit":{"prerequisite":"unchanged candidate passes train, test, eval, final","persistent_long_vol_comparator":"same accepted clock and 0.5 gross, side forced long","full_calendar_decomposition":"candidate minus comparator net return","rv20_q90_decomposition":"same decomposition on causal RV20 q90 decisions","minimum_q90_trades":8,"candidate_q90_absolute_return_positive":True,"candidate_specific_q90_residual_positive":True,"comparator_cannot_satisfy_candidate_claim":True},
  "diagnostic_controls":{"definitions":{"no_flow_contradiction":"separation side without contradiction requirement","flow_side":"sign aggregate signed taker notional on the primary contradiction clock","window_return_side":"sign completed six-hour open-to-close return on the primary contradiction clock","direction_flip":"negative primary side"},"clock_rule":"controls abstain on their own zero statistic and cannot be promoted","cannot_be_promoted":True},
  "source_plan":{"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
  "research_boundary":{"exact_avfcr_outcomes_known":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False},
  "stopping_rule":"terminal first failure; no window, VWAP formula, contradiction, side, clock, hold, RV20, subset, comparator, control, or gate repair"
 };return {**core,"manifest_hash":canonical_hash(core)}

def validate(x:dict[str,Any])->None:
 if x.get("manifest_hash")!=canonical_hash({k:v for k,v in x.items() if k!="manifest_hash"}):raise RuntimeError("AVFCR preregistration drift")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();r=build();validate(r);a.output.write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n");print(a.output)
