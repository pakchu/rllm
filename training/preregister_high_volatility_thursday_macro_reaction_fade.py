"""Outcome-blind preregistration for HVTMRF-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVTMRF-6"
DEFAULT_OUTPUT = Path("results/high_volatility_thursday_macro_reaction_fade_preregistration_2026-08-09.json")
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_thursday_macro_reaction_fade_v1",
        "policy_id": POLICY_ID, "as_of_date": "2026-08-09", "singleton": True,
        "outcomes_opened": False, "source_incidence_opened": False, "gross9_rows_opened": False,
        "mechanism": {
            "claim": "The recurring Thursday 08:30 America/New_York US initial-claims information window concentrates macro risk repricing. When BTC was already in a high-variation state, a one-hour directional reaction is more likely to overshoot in continuous speculative liquidity; fade that completed reaction for six hours.",
            "side": "negative strict sign of the completed first post-anchor hour BTC return",
            "why_distinct": "No prior repository policy uses the fixed weekly US initial-claims clock. HVOFR trades exchange opening-range failure, WHVOF trades weekend blocks, and macro-factor relays use external daily levels. HVTMRF uses only a predeclared weekly macro clock, completed BTC reaction, and causal prior variation.",
            "why_low_gross9_overlap_is_plausible": "one Thursday 09:35 America/New_York entry per week under a high-variation gate is absent from Gross9",
            "literature_context": {"paper": "Wen, Bouri, Xu and Zhao (2022), Intraday return predictability in the cryptocurrency markets: Momentum, reversal, or both", "doi": "10.1016/j.najef.2022.101733", "supported_scope": "Bitcoin exhibits both intraday momentum and crypto-specific reversal, with predictability changing around macro announcements", "implementation_is_not_a_replication": True},
        },
        "features": {
            "anchor": "every Thursday at 08:30 America/New_York, fixed independently of observed data; UTC conversion uses IANA timezone rules",
            "reaction_window": "exact 12 completed contiguous BTCUSDT 5m bars [08:30,09:30) America/New_York",
            "reaction_return": "log(09:25 close / 08:30 open), strict nonzero",
            "pre_anchor_variation": "sqrt(sum squared exact BTCUSDT 5m close-to-close log returns over [anchor-24h,anchor)); current reaction excluded",
            "variation_rank": "strict-prior midrank over at most 90 prior valid Thursday anchors, minimum 60, current excluded; rank>=0.65",
            "validity": "exact unique grid, finite positive coherent OHLC, no imputation",
        },
        "clock": {
            "decision": "Thursday 09:30 America/New_York after the reaction window completes",
            "entry": "exact Thursday 09:35 America/New_York BTCUSDT open", "hold": "6 elapsed hours",
            "side": "opposite reaction-return sign", "reservation": "global half-open; weekly clocks cannot overlap",
            "split_crossing_action": "skip", "gross_exposure": .5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty",
        },
        "policy": {"variation_prior_anchors": 90, "variation_minimum_anchors": 60, "variation_rank_min": .65, "reaction_hours": 1, "decision_delay_minutes": 60, "entry_delay_minutes": 5, "hold_hours": 6, "leverage": .5, "base_cost_per_notional_side": .0006, "stress_cost_per_notional_side": .001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": .2, "max_month_share": .45},
        "novelty_gates": {"exact_entry_jaccard_max": .1, "candidate_near_6h_share_max": .35, "occupied_5m_bar_jaccard_max": .25, "absolute_signed_exposure_pearson_max": .35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3., "strict_mdd_max_pct": 15., "mean_gross_underlying_min_bp": 20., "weekly_signflip_one_sided_p_max": .1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes train, test, eval, final", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"definitions": {"no_variation_gate": "fade every valid Thursday completed reaction", "reaction_continuation": "primary clock with reaction-return sign", "half_hour_reaction_fade": "same anchor and variation law using only [08:30,09:00), decision 09:00 and entry 09:05 America/New_York", "one_week_stale_reaction": "primary clock using the immediately prior valid Thursday reaction sign", "direction_flip": "negative primary side", "same_clock_forced_long": "side +1 on primary clock"}, "cannot_be_promoted": True},
        "source_plan": {"historical_market": {"path": str(MARKET), "sha256": MARKET_SHA}, "live_extension": "read-only Postgres BTCUSDT 1m through 2026-08-01", "calendar": "Python zoneinfo America/New_York recurring Thursday rule; no observed release outcome or calendar fitting", "execution_prices": "sealed until source and novelty pass"},
        "research_boundary": {"published_intraday_context_read": True, "prior_macro_candidate_event_sets_reused": False, "exact_hvtmrf_incidence_or_outcomes_known": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent recurring macro-window overreaction mechanism targeting user-requested volatility"},
        "stopping_rule": "terminal first failure; no weekday, timezone, reaction, rank, side, clock, hold, subset, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVTMRF preregistration drift")
    if hashlib.sha256(MARKET.read_bytes()).hexdigest() != MARKET_SHA:
        raise RuntimeError("HVTMRF market binding drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); result = build(); validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
