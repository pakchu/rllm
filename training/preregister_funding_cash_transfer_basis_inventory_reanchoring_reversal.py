"""Outcome-blind preregistration for FCBIRR-8."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

POLICY_ID="FCBIRR-8"
DEFAULT_OUTPUT=Path("results/funding_cash_transfer_basis_inventory_reanchoring_reversal_preregistration_2026-08-11.json")


def canonical_hash(value:Any)->str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def build()->dict[str,Any]:
    core={
      "protocol_version":"funding_cash_transfer_basis_inventory_reanchoring_reversal_v1","policy_id":POLICY_ID,"as_of_date":"2026-08-11","singleton":True,
      "outcomes_opened":False,"source_incidence_opened":False,"gross9_rows_opened":False,
      "public_source_basis":{"binance_funding":"https://www.binance.com/en/support/faq/detail/360033525031","perpetual_theory":"https://arxiv.org/html/2212.06888v5","dvol":"https://insights.deribit.com/exchange-updates/dvol-deribit-implied-volatility-index/","claims_used":["funding periodically transfers cash between long and short holders to tether perpetual price to spot","the preceding premium/basis path is the funding input and positive funding incentivizes futures short arbitrage","DVOL is a forward-looking implied-volatility action regime"],"implementation_is_unpublished_adaptation":True},
      "mechanism":{"claim":"At an actual BTCUSDT funding cash transfer, a same-sign settled funding rate and upper-tail preceding eight-hour mean premium identify one-sided basis pressure. If open interest increased during that window while DVOL is elevated, the cash transfer and arbitrage incentive should re-anchor the crowded perpetual; trade opposite the common funding/premium sign for one inventory cycle.","side":"negative strict common sign of the actual funding rate and preceding eight-hour mean premium index","why_distinct":"OCDR requires joint BVOL/DVOL body expansion, DVOL body leadership, an extreme funding tail, and rising OI. FSVUR/FSVCCR require pre/post BTC price paths and post-settlement volatility cooling. HVEFR fades a causal funding residual under realized-price variation. FCBIRR uses no BTC price, BVOL, funding magnitude tail, post-settlement confirmation, prior event set, or prior control; it uses the exact cash-transfer event, preceding mean premium, net OI expansion, and a DVOL level regime.","why_suited_to_volatile_regimes":"completed DVOL close must occupy its causal upper 40%","why_low_gross9_overlap_is_plausible":"actual millisecond funding events mapped to conservative +10m entries form a sparse derivatives-only clock"},
      "features":{"event":"each actual Binance BTCUSDT funding settlement S","source_end":"floor S to exact UTC hour H","premium_window":"480 exact unique one-minute premium rows in [H-8h,H); arithmetic mean close is finite, strict nonzero, and same sign as settled funding","premium_rank":"strict-prior midrank of absolute eight-hour mean premium over at most 270 prior valid settlements, minimum 180; current excluded; rank>=0.60","oi_window":"96 exact unique five-minute OI rows in [H-8h,H); log(last/first)>0","dvol_regime":"exact completed Deribit DVOL hour ending H, positive close; strict-prior 270-settlement midrank with 180 minimum; rank>=0.60","no_imputation":True,"grid":False},
      "clock":{"decision":"ceil actual funding timestamp S to the next exact five-minute boundary after all event/source fields exist","entry":"one full five-minute bar after decision; standard .001ms settlement therefore enters S-hour+10m","side":"opposite common funding and premium sign","hold":"8 elapsed hours","reservation":"global half-open; exit first on equal open","split_crossing_action":"skip","gross_exposure":.5,"signal_settlement":"excluded from held funding cash because entry is later","later_funding":"exact entry<=time<exit after novelty only","rv20":"q90 audit only after all economic stages pass"},
      "policy":{"premium_minutes":480,"oi_points":96,"history_settlements":270,"minimum_history_settlements":180,"premium_mean_rank_min":.60,"dvol_level_rank_min":.60,"entry_delay_bars_5m":1,"hold_hours":8,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
      "stages":{"train":["2023-07-01T00:00:00Z","2024-01-01T00:00:00Z"],"test":["2024-01-01T00:00:00Z","2025-01-01T00:00:00Z"],"eval":["2025-01-01T00:00:00Z","2026-01-01T00:00:00Z"],"final":["2026-01-01T00:00:00Z","2026-08-01T00:00:00Z"]},
      "source_support_gates":{"minimum_events":{"train":8,"test":12,"eval":12,"final":8},"minority_side_share_min":.20,"max_month_share":.45},
      "novelty_gates":{"exact_entry_jaccard_max":.10,"candidate_near_6h_share_max":.35,"occupied_5m_bar_jaccard_max":.25,"absolute_signed_exposure_pearson_max":.35,"must_pass_before_economics":True},
      "economic_gates":{"absolute_return_positive":True,"cagr_to_strict_mdd_min":3.,"strict_mdd_max_pct":15.,"mean_gross_underlying_min_bp":20.,"weekly_signflip_one_sided_p_max":.10,"stress_absolute_return_positive":True,"stress_cagr_to_strict_mdd_min":2.5,"each_calendar_half_positive":True,"stop_on_first_failure":True,"accounting":"fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
      "post_stage_volatility_audit":{"prerequisite":"unchanged all-stage pass","rv20_q90_entry_filter":False,"minimum_q90_trades":8,"candidate_q90_absolute_return_positive":True,"identical_clock_forced_long_residual_positive":True},
      "diagnostic_controls":{"names":["no_dvol_level","no_premium_tail","no_oi_increase","no_funding_premium_sign_agreement","one_settlement_stale_features","direction_flip","forced_long"],"cannot_be_promoted":True},
      "source_plan":{"funding":{"table":"funding_rates_binance","symbol":"BTCUSDT","actual_timestamps":True,"read_only":True},"premium":{"table":"bars_binance_premium","symbol":"BTCUSDT","interval":"1m","read_only":True},"oi":{"table":"open_interest_binance","symbol":"BTCUSDT","period":"5m","read_only":True},"dvol":{"path":"data/options_crowding_deleveraging_relay_sources_v4_2023_2026/dvol_hourly.csv.gz","read_only":True},"execution_prices":"sealed until source and novelty pass"},
      "research_boundary":{"prior_funding_family_outcomes_known":True,"exact_candidate_incidence_or_outcomes_known":False,"prior_event_sets_or_controls_reused":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"public no-arbitrage cash-transfer reanchoring with independent premium/OI/DVOL state"},
      "stopping_rule":"Terminal first failure; no threshold, side, hold, clock, subset, source, or control repair."
    }
    return {**core,"manifest_hash":canonical_hash(core)}


def validate(value:dict[str,Any])->None:
    core={k:v for k,v in value.items() if k!="manifest_hash"}
    if value.get("manifest_hash")!=canonical_hash(core):raise RuntimeError("FCBIRR preregistration hash mismatch")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args();value=build();validate(value);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(args.output)
