"""Outcome-blind preregistration for HVCOUNTER-R10-E5-8."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID="HVCOUNTER-R10-E5-8"
DEFAULT_OUTPUT=Path("results/high_volatility_counterattack_reversal_relay_preregistration_2026-08-11.json")

def canonical_hash(value:Any)->str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def build()->dict[str,Any]:
    core=copy.deepcopy(template.build());core.pop("manifest_hash")
    core.update(
      protocol_version="high_volatility_counterattack_reversal_relay_v1",policy_id=POLICY_ID,as_of_date="2026-08-11",
      mechanism={"claim":"During elevated completed realized variation, two consecutive hourly long real bodies of opposite color that finish at equal closes within the official causal tolerance identify a counterattack: the second initiative has fully opposed the first without extending the terminal price. Follow the official second-candle direction for eight hours.","side":"long for official bullish Counterattack output +1 and short for official bearish output -1","why_distinct":"Counterattack uses two separately lagged ten-body averages plus a five-range equal-close tolerance. It requires opposing long bodies but neither body/range containment, a multi-bar false breakout, a channel, nor a wide-range continuation. No volume, flow, OI, funding, fitted outcome, reused event set, or promoted control is used.","why_suited_to_volatile_regimes":"opposing initiative bodies around the same terminal price can resolve sharply when trailing twenty-four-hour realized variation ranks in its causal upper 35%","why_low_gross9_overlap_is_plausible":"asynchronous two-hour equal-close opposing-body clocks are absent from Gross9 primitives"},
      external_basis={"origin":"Counterattack candlestick pattern; QuantConnect LEAN canonical implementation","definition_source":"https://github.com/QuantConnect/Lean/blob/master/Indicators/CandlestickPatterns/Counterattack.cs","definition_source_sha256":"c1f58d4a53b931885765cbcbb0a75a3c7e9947d6bd01439739a861570b95818b","candle_settings_source":"https://github.com/QuantConnect/Lean/blob/master/Indicators/CandlestickPatterns/CandleSettings.cs","fixed_definition":"BodyLong is factor 1 times mean real body over 10 prior applicable candles; Equal is factor .05 times mean high-low range over 5 candles preceding the first pattern candle; first and second real bodies must each be strictly above their position-specific BodyLong averages, colors must oppose, and second close must lie inclusively within first close plus or minus Equal tolerance; output equals second color","warmup":"max(Equal 5, BodyLong 10)+1+1 = 12 completed hourly bars","selection_use":"official defaults, inequalities, output direction, and readiness only; no incidence or outcomes"},
      features={"decision_grid":"every exact hourly UTC boundary","source_bar":"exact aggregation of 60 coherent BTCUSDT one-minute rows [T-1h,T)","candle_color":"white when close>=open and black otherwise, matching LEAN","body_long_averages":"first body compared with mean real body of its prior 10 hours; second body compared with mean real body of its prior 10 hours, including the first pattern body","equal_tolerance":"0.05 times mean high-low range of the 5 hours preceding the first pattern body","pattern":"official ready Counterattack output exactly +1 or -1","variation":"sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive","variation_rank":"strict-prior midrank over at most 2160 earlier source-valid hourly decisions, minimum 720, current excluded; rank>=0.65","recursive_reset":"hour aggregation, all candle-setting totals, readiness, and variation history reset after any invalid completed source hour","no_imputation":True},
      clock={"feature_available":"hour boundary after completed second pattern body","entry":"exact BTCUSDT open five elapsed minutes later","hold":"8 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
      policy={"body_long_average_period":10,"body_long_factor":1.0,"equal_average_period":5,"equal_factor":.05,"indicator_period_hours":12,"variation_hours":24,"variation_history_decisions":2160,"minimum_variation_history_decisions":720,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":8,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
      diagnostic_controls={"names":["no_variation_gate","no_long_body_requirements","opposing_bodies_without_equal_close","one_hour_stale_pattern","direction_flip","forced_long"],"cannot_be_promoted":True},
      source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2023-04-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
      research_boundary={"canonical_counterattack_definition_read":True,"repository_counterattack_candidate_found":False,"prior_wide_body_hikkake_three_inside_and_absorption_candidates_known":True,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"official LEAN Counterattack under the requested high-variation regime"},
      stopping_rule="Terminal first failure; no timeframe, candle settings, inequalities, readiness, variation, side, hold, clock, subset, threshold, control, or other repair.")
    return {**core,"manifest_hash":canonical_hash(core)}

def validate(value:dict[str,Any])->None:
    core={key:item for key,item in value.items() if key!="manifest_hash"}
    if value.get("manifest_hash")!=canonical_hash(core) or value!=build():raise RuntimeError("HVCOUNTER-R10-E5-8 preregistration drift")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args();value=build();validate(value)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(args.output)
