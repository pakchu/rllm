"""Outcome-blind preregistration for VSPCR-8."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "VSPCR-8"
DEFAULT_OUTPUT = Path(
    "results/volume_stratified_price_control_relay_preregistration_2026-08-09.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "volume_stratified_price_control_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "When the highest-volume quarter of a completed six-hour window moves strongly in one direction while the lowest-volume quarter moves oppositely, and the high-volume return dominates, price discovery is volume-sponsored rather than ambient drift. Follow the high-volume direction for eight elapsed hours at the onset of that state.",
            "side": "strict sign of R_H",
            "why_distinct": "VSPCR stratifies the same 72 completed five-minute returns by contemporaneous quote volume, then requires opposite signed high- and low-volume cohort returns plus high-volume directional dominance. It is not an unweighted return tail, a high-volume move without disagreement, a single dominant bar, a temporal-half split, funding, flow classification, or a fitted model.",
            "volatile_market_target": "volume-sponsored price control should persist through volatile auctions; causal RV20 q90 remains only a later post-stage audit",
            "why_low_gross9_overlap_is_plausible": "strict-prior top-quintile cross-stratum disagreement onsets with an eight-hour global reservation",
        },
        "features": {
            "decision_grid": "every exact 4-hour UTC boundary",
            "window": "exact 72 completed coherent five-minute bars [decision-6h,decision)",
            "bar_quote_volume": "q_i=sum quote_asset_volume across the five constituent one-minute rows",
            "bar_return": "r_i=log(close/open) for each completed five-minute aggregate",
            "volume_rank": "rank all 72 bars lexicographically by (q_i,timestamp); H is the top 18 and L is the bottom 18",
            "high_volume_return": "R_H=sum r_i over H",
            "low_volume_return": "R_L=sum r_i over L",
            "normalizer": "V=sqrt(sum r_i^2 over all 72 bars), strict positive",
            "stratified_disagreement": "S=abs(R_H-R_L)/V, finite",
            "high_volume_dominance": "D_H=abs(R_H)/(abs(R_H)+abs(R_L)), with strict positive denominator",
            "stratified_disagreement_rank": "strict-prior midrank of S over at most 540 prior valid exact four-hour decisions; minimum 360; current excluded",
            "eligible_state": "R_H*R_L<0, D_H>=2/3, and stratified disagreement rank>=0.80",
            "onset": "current eligible state true and immediately preceding exact four-hour decision eligible state false",
            "source_valid": "each five-minute aggregate has exactly five unique coherent finite positive-OHLC one-minute rows with finite nonnegative quote_asset_volume; all 72 aggregates must be complete; no imputation",
        },
        "rv20_stress_slice": {
            "rv20": "sqrt(365*mean exact daily returns^2 over t-20 through t-1)",
            "threshold": "numpy linear q90 over 756 strictly prior available RV20 observations",
            "entry_filter": False,
            "future_use": "only after all sequential full-calendar stages pass",
        },
        "clock": {
            "decision": "exact 4-hour UTC boundary after all 72 five-minute bars complete",
            "entry": "exact BTCUSDT decision+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not an input; exact realized funding only after novelty",
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.2,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.1,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_bar_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.1,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR",
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, final",
            "persistent_long_vol_comparator": "same accepted clock and 0.5 gross, side forced long",
            "full_calendar_decomposition": "candidate minus comparator net return",
            "rv20_q90_decomposition": "same decomposition on causal RV20 q90 decisions",
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "candidate_specific_q90_residual_positive": True,
            "comparator_cannot_satisfy_candidate_claim": True,
        },
        "diagnostic_controls": {
            "definitions": {
                "unweighted_6h_return": "U=sum all 72 r_i; active when strict-prior midrank(abs(U))>=0.80 with the primary 540/360 history and false-to-true onset; side=sign(U)",
                "high_volume_without_disagreement": "same H/L, S rank>=0.80, D_H>=2/3, R_H!=0, and false-to-true onset, but omit R_H*R_L<0; side=sign(R_H)",
                "single_dominant_volume_bar": "M is r_i of the last bar in ascending lexicographic (q_i,timestamp) order; active when strict-prior midrank(abs(M))>=0.80 with the primary 540/360 history and false-to-true onset; side=sign(M)",
                "temporal_half_partition": "R_F=sum final 36 r_i and R_E=sum first 36 r_i; S_T=abs(R_F-R_E)/V; require R_F*R_E<0, abs(R_F)/(abs(R_F)+abs(R_E))>=2/3, strict-prior midrank(S_T)>=0.80 with 540/360 history, and false-to-true onset; side=sign(R_F)",
                "one_decision_stale_strata": "primary eligible-state boolean and primary side are both shifted by one exact four-hour decision; apply false-to-true onset to the shifted boolean",
                "direction_flip": "negative primary side",
            },
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "exact_vspcr_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": "terminal first failure; no volume rank, cohort, formula, threshold, history, onset, side, clock, hold, RV20, subset, comparator, control, or gate repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    payload = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(payload):
        raise RuntimeError("VSPCR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
