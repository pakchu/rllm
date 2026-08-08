"""Outcome-sequenced preregistration for HVTCR-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_trough_candle_range_relay_preregistration_2026-08-09.json"
)
SOURCE_BINDINGS = {
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz": (
        "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
    ),
}


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_trough_candle_range_relay_v1",
        "policy_id": "HVTCR-24",
        "as_of_date": "2026-08-09",
        "oos_outcomes_opened": False,
        "oos_source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "In a high-volatility state, the full range of the particular five-minute candle "
                "that established the trailing six-hour low measures whether forced selling was "
                "absorbed or support remained fragile. An upper-tail trough-candle range relays long, "
                "while a lower-tail range relays short for twenty-four hours."
            ),
            "side": "upper calibrated trough-candle-range tail long; lower tail short",
            "why_distinct": (
                "HVTCR attributes the rolling low to one causal candle and uses that candle's full "
                "range relative to current close. It is not HVPHR peak geometry, a Donchian reclaim, "
                "current wick, range "
                "position, realized-variance allocation, model output, flow, OI, funding, calendar, "
                "terminal candidate repair, or diagnostic-control promotion."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "tail onsets depend on the identity and geometry of an endogenous trough-setting candle"
            ),
        },
        "causal_features": {
            "decision": "each completed xx:55 UTC 5m bar",
            "lookback": "72 completed five-minute bars including the decision bar",
            "trough_row": (
                "last row attaining the minimum low in the lookback; deterministic last occurrence "
                "tie break"
            ),
            "trough_candle_range": "(high[trough_row]-low[trough_row])/close[decision]",
            "range_vol": (
                "(maximum high-minimum low)/midpoint over 144 completed five-minute bars including "
                "the decision bar"
            ),
            "calibration": (
                "2023H1 source-only q15/q85 of trough_candle_range and q60 of range_vol across complete "
                "hourly anchors; no calibration labels or post-entry prices"
            ),
            "eligibility": (
                "range_vol>=q60 and trough_candle_range<=q15 or >=q85; first row of a consecutive same-tail "
                "run only"
            ),
            "missing": "inactive; no interpolation or forward fill",
            "grid": False,
        },
        "oos_clock": {
            "domain_start": "2023-07-01T00:00:00Z",
            "entry": "next exact-hour 5m open",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
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
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate",
                "first_trough_tie_break",
                "current_candle_range",
                "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "source_plan": {
            "historical": "hash-bound 5m market through 2026-06-01",
            "live_extension": "read-only Postgres completed bars through 2026-08-01",
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "source_bindings": SOURCE_BINDINGS,
        "research_boundary": {
            "prior_univariate_trough_candle_forward_spread_diagnostics_known": True,
            "prior_diagnostics_are_not_a_strict_strategy_replay": True,
            "exact_hvtcr_policy_clock_and_strict_economics_known": False,
            "oos_candidate_incidence_opened": False,
            "oos_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": (
            "freeze singleton and calibration contract, then open source incidence, Gross9 novelty, "
            "and stage outcomes sequentially; terminal first failure without tail, tie-break, side, "
            "volatility, or hold repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVTCR preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"HVTCR source drift: {raw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
