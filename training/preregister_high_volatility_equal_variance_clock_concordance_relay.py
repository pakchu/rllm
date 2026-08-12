"""Outcome-blind preregistration for HVEVC-8."""
from __future__ import annotations

import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVEVC-8"
DEFAULT_OUTPUT = Path("results/high_volatility_equal_variance_clock_concordance_relay_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build()); contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_equal_variance_clock_concordance_relay_v1", policy_id=POLICY_ID, as_of_date="2026-08-13",
        mechanism={
            "claim": "In a completed high-variation eight-hour BTC path, unanimous direction across four sequential equal-realized-variance intrinsic-time segments means price discovery persisted through every risk quartile rather than being created by one physical-time burst; follow that direction for eight elapsed hours.",
            "side": "common strict sign of all four equal-variance-clock segment returns",
            "why_distinct": "HVEVC partitions the path at cumulative squared-return quartiles and requires unanimity across those endogenous risk-time segments. Fixed temporal halves, equal-duration trend persistence, one dominant shock, variance concentration, path efficiency, directional-change state machines, and volatility dispersion do not construct or sign equal-variance intrinsic-time returns. HVEVC uses no fitted outcome, volume, flow, OI, funding, prior event, or control.",
            "why_suited_to_volatile_regimes": "completed eight-hour variation must rank in its causal upper thirty-five percent and directional persistence must survive all four risk-time quartiles",
            "why_low_gross9_overlap_is_plausible": "three fixed UTC clocks filtered by endogenous equal-variance segment unanimity are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "realized-variance intrinsic-time transformation",
            "fixed_definition": "use cumulative squared five-minute close returns as the clock and split ordered returns at 25%, 50%, and 75% of completed total variation",
            "selection_use": "the equal-variance time change only; no directional return, incidence, or outcome claim is imported",
        },
        features={
            "decision_grid": "exact 00:00, 08:00, and 16:00 UTC boundaries",
            "path": "96 exact completed five-minute closes over [T-8h,T), yielding 95 ordered close-to-close log returns",
            "realized_variance": "sum squared 95 returns, finite strict positive",
            "variance_clock_assignment": "return i belongs to quartile min(3,floor(4*cumulative_variance_before_i/total_variance)); bars are indivisible and assignment is fixed before adding return i",
            "segment_validity": "each of four ordered quartiles contains at least one return",
            "segment_return": "sum of close-to-close log returns assigned to each intrinsic-time quartile, each finite strict nonzero",
            "concordance": "all four segment returns have one common strict sign",
            "variation_rank": "strict-prior midrank of sqrt(realized_variance) over at most 270 earlier source-valid decisions, minimum 180, current excluded; rank>=0.65",
            "eligibility": "source-valid, four nonempty strict-nonzero segments, concordance, and variation rank gate",
            "no_imputation": True,
        },
        clock={"feature_available":"eight-hour boundary after all path closes complete","entry":"exact BTCUSDT open five elapsed minutes later","side":"common segment-return sign","hold":"8 elapsed hours","reservation":"global half-open first-eligible reservation; exit first on equal open","gross_exposure":0.5,"funding":"not a signal input; exact settlements only after novelty passes"},
        policy={"bar_minutes":5,"path_bars":96,"variance_segments":4,"decision_hours":8,"variation_history_decisions":270,"minimum_history_decisions":180,"variation_rank_min":0.65,"entry_delay_minutes":5,"hold_hours":8,"leverage":0.5,"base_cost_per_notional_side":0.0006,"stress_cost_per_notional_side":0.001},
        diagnostic_controls={"names":["no_variation_gate","three_of_four_concordance","equal_physical_time","direction_flip","forced_long"],"cannot_be_promoted":True},
        source_plan={"bars":{"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","columns":["ts","open","high","low","close"],"window":["2023-05-25T00:00:00Z","2026-08-01T00:00:00Z"],"read_after_preregistration":True},"execution_prices":"sealed until source support and Gross9 novelty pass"},
        research_boundary={"realized_variance_intrinsic_time_definition_read":True,"repository_exact_equal_variance_clock_concordance_candidate_found":False,"adjacent_temporal_persistence_variance_concentration_dispersion_and_path_efficiency_candidates_known":True,"adjacent_candidate_outcomes_used_to_set_formula_side_hold_clock_or_threshold":False,"prior_event_sets_reused":False,"candidate_incidence_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"candidate_count":1,"grid":False,"repair_of_prior_candidate":False,"promoted_prior_control":False,"selection_basis":"fixed four-segment realized-variance intrinsic clock concordance under the requested high-variation regime"},
        stopping_rule="Terminal first failure; no path, variance clock, segment count, unanimity, variation rank, side, hold, clock, subset, threshold, or control repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(value: dict[str, Any]) -> None:
    core={k:v for k,v in value.items() if k!="manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build(): raise RuntimeError("HVEVC preregistration drift")


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args();payload=build();validate(payload);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+"\n");print(args.output)
