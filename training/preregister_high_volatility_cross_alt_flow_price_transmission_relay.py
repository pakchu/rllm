"""Outcome-blind preregistration for HVCAFPT-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVCAFPT-8"
SLUG = "high_volatility_cross_alt_flow_price_transmission_relay"
ALTS = ("ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
DEFAULT_OUTPUT = Path(f"results/{SLUG}_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": f"{SLUG}_v1",
        "policy_id": POLICY_ID,
        "slug": SLUG,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": (
                "During a completed volatile eight-hour crypto auction, aggressive quote flow in "
                "the first four hours that is followed by a same-direction price response in the "
                "second four hours across at least four liquid alt perpetuals identifies broad, "
                "still-propagating information absorption. Follow the common transmitted direction "
                "in BTC for eight elapsed hours."
            ),
            "side": "strict majority sign of the transmitting alts' second-half returns",
            "why_distinct": (
                "LFIC estimates BTC-only five-minute lagged flow/return coherence. HVCAQFCR and "
                "cross-alt flow candidates use contemporaneous or terminal flow signs. HVCAFRR fits "
                "a frozen outcome-response model. HVCAFPT instead uses a deterministic first-half "
                "alt aggressive-flow sign followed by a second-half same-alt return sign, then "
                "requires cross-sectional directional breadth; it uses no BTC return direction, "
                "fitted outcome, spot, premium, funding, OI, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "completed trailing twenty-four-hour BTC realized variation must occupy its causal "
                "upper 35 percent"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "offset 02/10/18 UTC ordered cross-alt flow-to-price transmission onsets are absent "
                "from Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "exact 02:00, 10:00, and 18:00 UTC boundaries D",
            "universe": list(ALTS),
            "block": (
                "480 exact aligned coherent native one-minute bars_binance rows [D-8h,D) per alt, "
                "split into two exact 240-minute halves"
            ),
            "first_half_flow": (
                "(2*sum(taker_buy_quote)-sum(quote_asset_volume))/sum(quote_asset_volume) over "
                "[D-8h,D-4h), finite strict nonzero with positive denominator"
            ),
            "second_half_return": (
                "log(last completed close/first completed open) over [D-4h,D), finite strict nonzero"
            ),
            "transmitting_alt": (
                "strict sign(first_half_flow) equals strict sign(second_half_return)"
            ),
            "directional_breadth": (
                "at least four of six alts transmit one common direction and strictly outnumber "
                "opposite-direction transmitters; non-transmitters abstain"
            ),
            "transmission_score": (
                "sum over alts transmitting in the majority direction of "
                "abs(first_half_flow)*abs(second_half_return)"
            ),
            "transmission_rank": (
                "strict-prior midrank over at most 270 source-valid positive-breadth decisions, "
                "minimum 180, current excluded; rank>=0.70"
            ),
            "btc_realized_variation": (
                "sqrt(sum squared exact BTC one-minute log(close/open) returns over [D-24h,D)), "
                "finite strict positive"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 270 source-valid decisions, minimum 180, current "
                "excluded; rank>=0.65"
            ),
            "onset": (
                "full eligible state true now and false at the immediately prior exact source-valid "
                "decision; missing prior cannot trigger"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "D after all completed alt and BTC source paths are available",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "majority transmitted direction",
            "hold": "8 elapsed hours",
            "reservation": "natural global half-open; exit first on equal-time entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty",
        },
        "policy": {
            "block_minutes": 480,
            "minimum_directional_breadth": 4,
            "history_decisions": 270,
            "minimum_history_decisions": 180,
            "transmission_rank_min": 0.70,
            "variation_rank_min": 0.65,
            "onset_required": True,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
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
                "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m "
                "favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes all four stages",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_transmission_tail",
                "no_variation_gate",
                "contemporaneous_flow_return",
                "one_block_stale_transmission",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "table": "bars_binance",
            "symbols": ["BTCUSDT", *ALTS],
            "interval": "1m",
            "columns": [
                "ts", "symbol", "open", "high", "low", "close",
                "quote_asset_volume", "taker_buy_quote",
            ],
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration_commit": True,
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_cross_alt_and_lagged_flow_outcomes_known": True,
            "repository_exact_cross_alt_ordered_flow_price_transmission_candidate_found": False,
            "prior_event_sets_or_controls_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent ordered cross-sectional flow-to-price propagation",
        },
        "stopping_rule": (
            "terminal first failure; no universe, block, half, flow, response, breadth, rank, "
            "variation, onset, side, clock, hold, subset, threshold, comparator, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVCAFPT preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != payload:
        raise RuntimeError(f"refusing overwrite {args.output}")
    args.output.write_bytes(payload)
    print(args.output)
