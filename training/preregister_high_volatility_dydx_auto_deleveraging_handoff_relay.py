"""Outcome-blind preregistration for HVDADH-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_dydx_auto_deleveraging_handoff_relay_preregistration_2026-08-10.json"
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_dydx_auto_deleveraging_handoff_relay_v1",
        "policy_id": "HVDADH-8",
        "as_of_date": "2026-08-10",
        "oos_outcomes_opened": False,
        "oos_source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "During volatile BTC trading, a large dYdX Chain forced-close episode is an "
                "exhaustion event when the protocol records only LIQUIDATED trades, but becomes a "
                "continuation event when the same completed window reaches the distinct "
                "DELEVERAGED risk-engine state. Fade the forced side in the first state and follow "
                "it in the second."
            ),
            "side": (
                "negative strict forced-flow sign when deleveraged notional is zero; positive "
                "strict forced-flow sign when deleveraged notional is positive"
            ),
            "why_distinct": (
                "The repository has no dYdX source or DELEVERAGED state. Prior liquidation work "
                "used Binance snapshots, OI/price proxies, or cross-asset liquidation flow; HVDADH "
                "uses the dYdX Chain indexer's explicit LIQUIDATED-versus-DELEVERAGED matching "
                "engine classification."
            ),
            "volatile_market_target": "strict-prior BTC six-hour realized-variation rank >=0.65",
            "why_low_gross9_overlap_is_plausible": (
                "entries require an hourly onset in a decentralized perpetual risk-engine state, "
                "not a fixed release, funding, daily, or exchange-calendar clock"
            ),
        },
        "source_contract": {
            "official_indexer": "https://indexer.dydx.trade/v4",
            "official_openapi": "https://indexer.dydx.trade/docs/",
            "official_repository": "https://github.com/dydxprotocol/v4-chain/tree/main/indexer",
            "endpoint": "/trades/perpetualMarket/BTC-USD",
            "source_window": ["2023-11-14T00:00:00Z", "2026-08-01T00:00:00Z"],
            "schema": [
                "id",
                "side",
                "size",
                "price",
                "type",
                "createdAt",
                "createdAtHeight",
            ],
            "accepted_types": ["LIMIT", "LIQUIDATED", "DELEVERAGED", "TWAP_SUBORDER"],
            "pagination": (
                "limit=1000 and descending createdBeforeOrAtHeight cursor; next cursor is one less "
                "than the minimum strictly-positive height returned; repeated id, nondecreasing "
                "cursor, malformed row, HTTP failure, or empty page before the lower boundary "
                "rejects the source"
            ),
            "trade_validity": (
                "unique nonempty id; side BUY or SELL; finite strictly-positive size and price; "
                "recognized type; timezone-aware createdAt; strictly-positive integer height; "
                "createdAt strictly before its decision time"
            ),
            "decision_grid": "every exact UTC hour D",
            "feature_window": "all valid BTC-USD trades with createdAt in [D-6h,D)",
            "causal_availability": (
                "createdAt and createdAtHeight are emitted by the public dYdX Chain indexer and "
                "the identical public WebSocket trade channel is the production counterpart; only "
                "rows strictly before D are used"
            ),
            "revision_boundary": (
                "capture every response-body SHA-256 and the OpenAPI initializer SHA-256; any "
                "historical reindexing after capture is outside this frozen run"
            ),
        },
        "feature_contract": {
            "trade_notional": "size*price in USD for every valid trade",
            "forced_types": ["LIQUIDATED", "DELEVERAGED"],
            "forced_buy_notional": "sum notional for forced rows whose side is BUY",
            "forced_sell_notional": "sum notional for forced rows whose side is SELL",
            "forced_notional": "forced_buy_notional+forced_sell_notional; strict positive",
            "forced_flow": "forced_buy_notional-forced_sell_notional; strict nonzero",
            "deleveraged_notional": "sum notional for DELEVERAGED rows",
            "deleveraged_state": "deleveraged_notional>0",
            "forced_notional_rank": (
                "strict-prior midrank of forced_notional among at most 720 valid hourly windows, "
                "current excluded, minimum 480; zero-forced windows are valid history"
            ),
            "btc_variation": (
                "sum squared close-to-close log returns from 360 exact coherent BTCUSDT "
                "bars_binance 1m rows [D-6h,D)"
            ),
            "btc_variation_rank": "same strict-prior 720/480 rule",
            "eligible_state": (
                "forced_notional>0, forced_flow strict nonzero, forced_notional_rank>=0.80, and "
                "btc_variation_rank>=0.65"
            ),
            "onset": "current eligible state true and immediately previous exact hourly state false",
            "no_imputation": True,
        },
        "oos_clock": {
            "start": "2023-11-14T00:00:00Z",
            "entry": "D+5m BTCUSDT perpetual open",
            "side": (
                "sign(forced_flow) when deleveraged_state is true, otherwise -sign(forced_flow)"
            ),
            "hold": "8 elapsed hours",
            "reservation": (
                "chronological first eligible onset while flat; intervals half-open and exit first "
                "on an equal-time entry"
            ),
            "funding": "not an input; exact settlements opened only after novelty passes",
        },
        "policy": {
            "window_hours": 6,
            "history_observations": 720,
            "minimum_history_observations": 480,
            "forced_notional_rank_min": 0.80,
            "btc_variation_rank_min": 0.65,
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
            "minority_side_share_min": 0.20,
            "max_month_share": 0.45,
            "complete_cursor_replay_required": True,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.10,
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
            "weekly_signflip_one_sided_p_max": 0.10,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding, 6bp/10bp per notional side, favorable-then-adverse "
                "held 5m path, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged passes every sequential economic stage",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate",
                "no_forced_notional_gate",
                "always_fade_forced_flow",
                "one_hour_stale_features",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "research_boundary": {
            "prior_liquidation_family_outcomes_known": True,
            "prior_dydx_or_deleveraged_state_outcomes_known": False,
            "candidate_specific_incidence_opened": False,
            "candidate_specific_postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "selection_basis": (
                "new official dYdX risk-engine state classification fixed before candidate "
                "incidence or outcomes"
            ),
        },
        "stopping_rule": (
            "freeze preregistration, source support, Gross9 novelty, and sequential economics; "
            "terminal first failure with no source cursor, type mapping, window, rank, onset, side, "
            "hold, clock, subset, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
