"""Outcome-blind preregistration for DSQFCR-12."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT=Path("results/daily_stablecoin_quote_flow_consensus_relay_preregistration_2026-08-08.json")
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def build()->dict[str,Any]:
 core={
  "protocol_version":"daily_stablecoin_quote_flow_consensus_relay_v1","policy_id":"DSQFCR-12","as_of_date":"2026-08-08","outcomes_opened":False,"source_incidence_opened":False,"singleton":True,
  "mechanism":{"claim":"After a source-complete UTC day, same-direction normalized aggressive BTC taker flow in both Binance USDT and USDC quote books during an already elevated BVOL/DVOL regime identifies broad stablecoin-denominated price discovery; the consensus direction should relay for twelve hours.","side":"common sign of daily USDT and USDC normalized signed taker flow","why_distinct":"DSQFCR aggregates two complete quote books once per UTC day. VGSQF was an hourly USDC+FDUSD consensus with USDT lag; VGSFR required ordered propagation; SQFD used an unrestricted hourly diffusion clock.","why_suited_to_volatile_regimes":"both BVOL and DVOL at the day boundary must exceed their causal 720-hour q60 levels","why_low_gross9_overlap_is_plausible":"one daily 00:05 UTC stablecoin-flow consensus clock is absent from Gross9"},
  "clock":{"source_day":"exact prior UTC day [D-24h,D), requiring all 24 hourly rows for BTCUSDT and BTCUSDC source_complete","normalized_flow":"sum(signed_taker_flow_btc)/sum(base_volume_btc), separately by quote book; denominator positive","consensus":"both normalized flows nonzero, same sign, and each absolute value>=0.05","participation":"USDC base-volume share of combined USDT+USDC base volume>=0.01","volatile_regime":"BVOL and DVOL closes available at D are each >= own strictly-prior 720h q60, requiring 672 valid hours","decision":"exact D 00:00 UTC after all source features complete","entry":"exact BTCUSDT D+5m open","side":"common flow sign","hold":"12 elapsed hours","reservation":"fixed daily decisions; global half-open, exit first on equal open","funding_oi":"not signal inputs; exact funding only for later PnL","no_imputation":True},
  "policy":{"absolute_normalized_flow_min":0.05,"usdc_volume_share_min":0.01,"volatility_prior_hours":720,"volatility_prior_min_hours":672,"volatility_level_quantile":0.60,"entry_delay_minutes":5,"hold_hours":12,"leverage":0.5,"base_cost_per_notional_side":0.0006,"stress_cost_per_notional_side":0.001},
  "stages":{"train":["2023-07-01T00:00:00Z","2024-01-01T00:00:00Z"],"test":["2024-01-01T00:00:00Z","2025-01-01T00:00:00Z"],"eval":["2025-01-01T00:00:00Z","2026-01-01T00:00:00Z"],"final":["2026-01-01T00:00:00Z","2026-08-01T00:00:00Z"]},
  "source_support_gates":{"minimum_events":{"train":8,"test":12,"eval":12,"final":8},"minority_side_share_min":0.20,"max_month_share":0.45},
  "novelty_gates":{"exact_entry_jaccard_max":0.10,"candidate_near_6h_share_max":0.35,"occupied_5m_jaccard_max":0.25,"absolute_signed_exposure_pearson_max":0.35,"must_pass_before_economics":True},
  "economic_gates":{"absolute_return_positive":True,"cagr_to_strict_mdd_min":3.0,"strict_mdd_max_pct":15.0,"mean_gross_underlying_min_bp":20.0,"weekly_signflip_one_sided_p_max":0.10,"stress_absolute_return_positive":True,"stress_cagr_to_strict_mdd_min":2.5,"each_calendar_half_positive":True,"stop_on_first_failure":True,"future_can_rank_repair_or_reselect":False,"accounting":"fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
  "source_bindings":{"spot_flow":"hash-bound Binance stablecoin quote-flow panel through 2026-07-31","volatility":"hash-bound completed-hour BVOL/DVOL panel through 2026-08-01","execution_price":"sealed until source-support and Gross9 novelty pass"},
  "diagnostic_controls":{"names":["no_volatility_gate","usdt_only","usdc_only","no_participation_gate","direction_flip"],"diagnostic_controls_cannot_be_promoted":True},
  "research_boundary":{"prior_hourly_stablecoin_candidate_incidence_known":True,"prior_candidate_outcomes_used_to_define_dsqfcr":False,"dsqfcr_candidate_incidence_opened":False,"dsqfcr_post_entry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"independent daily cross-denominator consensus mechanism with complete pre-entry sources"},
  "stopping_rule":"Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no threshold, side, hold, clock, or subset repair."
 }
 return {**core,"manifest_hash":canonical_hash(core)}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();a.output.write_text(json.dumps(build(),indent=2)+"\n");print(a.output)
