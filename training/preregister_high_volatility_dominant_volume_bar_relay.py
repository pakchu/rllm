"""Outcome-sequenced preregistration for HVDVBR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_dominant_volume_bar_relay_preregistration_2026-08-09.json"
)
SOURCE_BINDINGS = {
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz": (
        "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
    ),
}


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_dominant_volume_bar_relay_v1",
        "policy_id": "HVDVBR-12",
        "as_of_date": "2026-08-09",
        "oos_outcomes_opened": False,
        "oos_source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "During high BTC range volatility, a six-hour block whose quote volume "
                "suddenly concentrates in one five-minute bar records a discrete inventory "
                "transfer. When that dominant bar and the complete block move in the same "
                "direction, the transfer is sponsored price discovery whose direction should "
                "relay for twelve elapsed hours."
            ),
            "side": "common strict sign of the dominant-volume bar and six-hour block return",
            "why_distinct": (
                "HVDVBR uses the endogenous identity and quote-volume share of the single "
                "largest-participation bar. It is not a price-extremum candle, taker-flow "
                "imbalance, block VWAP, late-session volume share, variance concentration, "
                "volume clock, terminal-candidate repair, or diagnostic-control promotion."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "tail onsets depend on a six-hour participation-concentration event rather "
                "than Gross9 calendar or state clocks"
            ),
        },
        "causal_features": {
            "decision": "each completed xx:55 UTC five-minute bar",
            "lookback": "72 completed five-minute bars including the decision bar",
            "validity": (
                "72 exact distinct consecutive rows; finite positive coherent OHLC; finite "
                "nonnegative quote volume; positive total quote volume; no imputation"
            ),
            "dominant_row": (
                "last row attaining maximum quote_asset_volume in the lookback; deterministic "
                "last-occurrence tie break"
            ),
            "dominant_share": "quote_asset_volume[dominant_row] / sum(quote_asset_volume)",
            "dominant_return": "log(close/open) of dominant_row",
            "block_return": "log(close[decision]/open[first lookback row])",
            "range_vol": (
                "(maximum high-minimum low)/midpoint over 144 completed five-minute bars "
                "including the decision bar"
            ),
            "calibration": (
                "2023H1 source-only q85 of dominant_share and q60 of range_vol across complete "
                "hourly anchors; no labels or post-entry prices"
            ),
            "eligibility": (
                "dominant_share>=q85, range_vol>=q60, and dominant_return and block_return "
                "have one common strict sign; first row of a consecutive eligible same-side "
                "run only"
            ),
            "grid": False,
        },
        "oos_clock": {
            "domain_start": "2023-07-01T00:00:00Z",
            "entry": "next exact-hour five-minute open",
            "hold": "12 elapsed hours",
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
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional "
                "side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate",
                "first_dominant_tie_break",
                "block_direction_only",
                "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "source_plan": {
            "historical": "hash-bound five-minute market through 2026-06-01",
            "live_extension": "read-only Postgres completed bars through 2026-08-01",
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "source_bindings": SOURCE_BINDINGS,
        "research_boundary": {
            "prior_volume_clock_and_daily_volume_candidate_outcomes_known": True,
            "those_clocks_or_controls_are_not_reused": True,
            "dominant_bar_concentration_policy_outcomes_known": False,
            "oos_candidate_incidence_opened": False,
            "oos_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": (
            "freeze singleton and source-only calibration, then open source incidence, "
            "Gross9 novelty, and stage outcomes sequentially; terminal first failure without "
            "threshold, side, tie-break, volatility, onset, or hold repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVDVBR preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"HVDVBR source drift: {raw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
