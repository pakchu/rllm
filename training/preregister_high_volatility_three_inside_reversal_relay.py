"""Outcome-blind preregistration for HV3INSIDE-R10-8."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HV3INSIDE-R10-8"
DEFAULT_OUTPUT = Path("results/high_volatility_three_inside_reversal_relay_preregistration_2026-08-11.json")

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build()); core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_three_inside_reversal_relay_v1", policy_id=POLICY_ID, as_of_date="2026-08-11",
        mechanism={
            "claim":"During elevated completed realized variation, the official hourly Three Inside Up/Down pattern confirms that a long directional real body was absorbed by a strictly contained short body and then reversed beyond the first candle's open. Enter in the confirmed reversal direction for eight hours.",
            "side":"long for official Three Inside Up output +1 and short for official Three Inside Down output -1",
            "why_distinct":"Three Inside uses real-body size relative to two separately lagged ten-body averages, strict real-body containment, and a third opposite-color close beyond the first open. Hikkake uses high-low containment, a same-direction false displacement, and a later three-bar confirmation; inside-auction resolution uses disjoint multi-hour range partitions and direct outside acceptance. No volume, flow, OI, funding, fitted outcome, reused event set, or promoted control is used.",
            "why_suited_to_volatile_regimes":"a completed origin-body reversal can propagate when trailing twenty-four-hour realized variation ranks in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible":"asynchronous three-hour body-containment reversal confirmations are absent from Gross9 primitives"},
        external_basis={
            "origin":"Three Inside Up/Down candlestick pattern; QuantConnect LEAN canonical implementation",
            "definition_source":"https://github.com/QuantConnect/Lean/blob/master/Indicators/CandlestickPatterns/ThreeInside.cs",
            "definition_source_sha256":"a8fc5894a73df1b56942238ed37c484bb4ed68d47558a955141663aad46440e2",
            "candle_settings_source":"https://github.com/QuantConnect/Lean/blob/master/Indicators/CandlestickPatterns/CandleSettings.cs",
            "fixed_definition":"BodyLong and BodyShort each use factor 1 times the mean real body of the prior 10 applicable hourly candles; first body is strictly above its lagged BodyLong average; second body is at most its lagged BodyShort average and both body endpoints are strictly inside the first body; third color opposes the first and closes strictly beyond the first open; output is the opposite of first color",
            "warmup":"max(BodyLong 10, BodyShort 10)+2+1 = 13 completed hourly bars",
            "selection_use":"official defaults, inequalities, output direction, and readiness only; no incidence or outcomes"},
        features={
            "decision_grid":"every exact hourly UTC boundary",
            "source_bar":"exact aggregation of 60 coherent BTCUSDT one-minute rows [T-1h,T)",
            "real_body":"absolute close minus open; white when close>=open and black otherwise, matching LEAN candle color",
            "body_averages":"separate rolling sums reproduce LEAN's prior-ten real-body averages for the first and second pattern positions, excluding each tested candle",
            "pattern":"official ready Three Inside output must be exactly +1 or -1",
            "variation":"sqrt(sum squared completed five-minute open-to-close log returns over [T-24h,T)), finite strict positive",
            "variation_rank":"strict-prior midrank over at most 2160 earlier source-valid hourly decisions, minimum 720, current excluded; rank>=0.65",
            "recursive_reset":"hour aggregation, both candle-setting totals, readiness, and variation history reset after any invalid completed source hour",
            "no_imputation":True},
        clock={"feature_available":"hour boundary after completed third pattern bar","entry":"exact BTCUSDT open five elapsed minutes later","hold":"8 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":.5,"funding":"not a signal input; exact settlements only after novelty passes"},
        policy={"body_long_average_period":10,"body_short_average_period":10,"body_long_factor":1.0,"body_short_factor":1.0,"indicator_period_hours":13,"variation_hours":24,"variation_history_decisions":2160,"minimum_variation_history_decisions":720,"variation_rank_min":.65,"entry_delay_minutes":5,"hold_hours":8,"leverage":.5,"base_cost_per_notional_side":.0006,"stress_cost_per_notional_side":.001},
        diagnostic_controls={"names":["no_variation_gate","no_body_size_requirements","containment_without_third_confirmation","one_hour_stale_pattern","direction_flip","forced_long"],"cannot_be_promoted":True},
        source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2023-04-01T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
        research_boundary={"canonical_three_inside_definition_read":True,"repository_three_inside_candidate_found":False,"related_hikkake_inside_auction_and_engulfing_candidates_known":True,"prior_event_sets_reused":False,"prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"official LEAN Three Inside confirmation under the requested high-variation regime"},
        stopping_rule="Terminal first failure; no timeframe, candle settings, inequalities, readiness, variation, side, hold, clock, subset, threshold, control, or other repair.")
    return {**core, "manifest_hash":canonical_hash(core)}

def validate(value: dict[str, Any]) -> None:
    core={key:item for key,item in value.items() if key!="manifest_hash"}
    if value.get("manifest_hash")!=canonical_hash(core) or value!=build(): raise RuntimeError("HV3INSIDE-R10-8 preregistration drift")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args()
    registration=build(); validate(registration); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(registration,indent=2,ensure_ascii=False,allow_nan=False)+"\n"); print(args.output)
