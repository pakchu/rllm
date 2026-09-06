"""Outcome-blind preregistration for PPCSR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "PPCSR-6"
DEFAULT_OUTPUT = Path(
    "results/premium_polarity_cross_settlement_relay_preregistration_2026-08-09.json"
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "premium_polarity_cross_settlement_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "A premium-index sign change after sixty consecutive completed minutes of the opposite strict sign marks a derivatives-pressure regime transfer; follow the newly established premium polarity for six elapsed hours.",
            "side": "strict sign of the current completed premium-index minute after the polarity cross",
            "why_distinct": "PPCSR uses premium polarity persistence and a zero crossing only. It uses no BTC price path, premium magnitude/rank, funding level, OI, flow, range breakout, block clock, fitted threshold, or prior candidate control.",
            "volatile_market_target": "a persistent one-sided derivatives basis that crosses through zero identifies leverage rotation during volatile repricing; causal RV20 q90 remains only a later audit",
            "why_low_gross9_overlap_is_plausible": "premium-only zero-cross events, exact five-minute sampling, a complete availability buffer, and six-hour reservation create a sparse independent clock",
        },
        "features": {
            "decision_schedule": "every exact five-minute UTC boundary D",
            "current_minute": "premium-index one-minute row [D-1m,D), with finite strict-nonzero close P0",
            "persistence_window": "the 60 exact premium-index minute closes in [D-61m,D-1m), all finite, strict nonzero, and one common sign S",
            "polarity_cross": "sign(P0)=-S with strict signs; no magnitude threshold or fitted rank",
            "raw_availability": "the current minute is conservatively available at D+1 second",
            "decision": "raw availability ceiled to the next exact five-minute boundary G=D+5m",
            "eligible_state": "polarity cross true; isolated event needs no fitted onset rule",
            "side": "sign(P0)",
            "source_valid": "61 exact unique coherent premium-index one-minute rows, no imputation",
        },
        "rv20_stress_slice": {
            "rv20": "sqrt(365*mean exact daily returns^2 over t-20 through t-1)",
            "threshold": "numpy linear q90 over 756 strictly prior available RV20 observations",
            "entry_filter": False,
            "future_use": "only after all sequential full-calendar stages pass",
        },
        "clock": {
            "entry": "exact BTCUSDT G+5m open (D+10m), leaving [D,D+5m) empty after source availability",
            "hold": "6 elapsed hours",
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
            "accounting": (
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional "
                "side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
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
                "thirty_minute_persistence": "same exact polarity-cross rule with only the immediately prior 30 minutes persistent",
                "one_event_stale_cross": "primary accepted event side shifted to the next exact five-minute decision",
                "direction_flip": "negative primary side",
                "same_clock_forced_long": "same accepted primary clock with side forced long",
            },
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "premium": {
                "table": "bars_binance_premium",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_premium_family_outcomes_known": True,
            "exact_ppcsr_incidence_known": False,
            "exact_ppcsr_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": (
            "terminal first failure; no block predicate, threshold, confirmation, embargo, "
            "side, clock, hold, RV20, subset, comparator, control, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("PPCSR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
