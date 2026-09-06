"""Outcome-blind preregistration for HVDQDR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVDQDR-8"
DEFAULT_OUTPUT = Path("results/high_volatility_dominant_quote_disagreement_resolution_relay_preregistration_2026-08-09.json")
FLOW = Path("data/binance_stablecoin_quote_flow_btc_2023_2026_aug/BTC_stablecoin_quote_flow_1h_2023-07-01_2026-07-31T23.csv.gz")
FLOW_SHA = "44374b9a2298ae4b64f0c1e7208665b1c08c8221045308694311123deae1c805"
FLOW_MANIFEST = FLOW.parent / "build_manifest.json"
FLOW_MANIFEST_SHA = "b9c64c3ce651934d9761a6d0731e814b2a92f5237b3040e11f794d7eb024a898"
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_dominant_quote_disagreement_resolution_relay_v1",
        "policy_id": POLICY_ID, "as_of_date": "2026-08-09", "singleton": True,
        "outcomes_opened": False, "source_incidence_opened": False, "gross9_rows_opened": False,
        "mechanism": {
            "claim": "During high BTC variation, a completed eight-hour block in which USDC and FDUSD aggressive-flow intensities agree but dominant USDT intensity has the opposite sign identifies unresolved cross-quote inventory. Because USDT carries dominant price discovery, follow its direction for eight hours.",
            "side": "strict sign of completed-block BTCUSDT normalized aggressive flow",
            "why_distinct": "VGSQF followed alternative-book consensus when USDT merely lagged under dual implied-volatility levels. DQDIR required a USDT impulse while alternative books were quiet plus OI contraction and BVOL/DVOL expansion. HVDQDR requires explicit opposite signs in all three active books, uses realized BTC variation only, and follows dominant-book resolution.",
            "why_low_gross9_overlap_is_plausible": "cross-quote flow-sign disagreement at three eight-hour boundaries is absent from Gross9",
        },
        "features": {
            "source": "hash-bound official Binance Spot hourly panel for BTCUSDT, BTCUSDC and BTCFDUSD",
            "block": "exact completed eight-hour UTC block ending at 00:00,08:00,16:00",
            "block_valid": "eight exact source_complete rows per symbol, finite nonnegative base volume, positive aggregate base volume; no imputation",
            "flow_intensity": "aggregate signed_taker_flow_btc divided by aggregate base_volume_btc independently per book",
            "alternative_consensus": "BTCUSDC and BTCFDUSD intensities have the same strict nonzero sign",
            "dominant_disagreement": "BTCUSDT intensity has the strict opposite sign",
            "active_books": "absolute intensity of every book is at least its own strict-prior 180-block median, minimum 90 valid blocks, current excluded",
            "btc_variation": "sqrt(sum squared exact completed 5m BTC log returns over 24 hours ending at decision)",
            "variation_rank": "strict-prior midrank over at most 270 prior valid decisions, minimum 180, current excluded; rank>=0.65",
        },
        "clock": {"decision": "exact block completion", "entry": "decision+5m BTCUSDT open", "hold": "8 elapsed hours", "side": "USDT flow-intensity sign", "reservation": "global half-open; exit first on equal open", "split_crossing_action": "skip", "gross_exposure": 0.5, "funding": "not a signal input; exact settlements after novelty"},
        "policy": {"block_hours": 8, "activity_prior_blocks": 180, "activity_minimum_blocks": 90, "activity_quantile": 0.5, "variation_prior_blocks": 270, "variation_minimum_blocks": 180, "variation_rank_min": 0.65, "entry_delay_minutes": 5, "hold_hours": 8, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes all stages", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"definitions": {"no_variation_gate": "active cross-quote disagreement without BTC variation rank", "no_activity_gate": "cross-quote sign disagreement and high variation without per-book activity medians", "alternative_direction": "alternative consensus side on primary clock", "one_block_stale_disagreement": "primary disagreement shifted one exact block while current USDT sets side", "direction_flip": "negative primary side", "same_clock_forced_long": "side +1 on primary clock"}, "cannot_be_promoted": True},
        "source_plan": {"flow": {"path": str(FLOW), "sha256": FLOW_SHA, "manifest": str(FLOW_MANIFEST), "manifest_sha256": FLOW_MANIFEST_SHA}, "historical_market": {"path": str(MARKET), "sha256": MARKET_SHA}, "live_extension": "read-only Postgres BTCUSDT 1m through 2026-08-01", "execution_prices": "sealed until source and novelty pass"},
        "research_boundary": {"prior_stablecoin_family_outcomes_known": True, "prior_event_sets_or_controls_reused": False, "exact_hvdqdr_incidence_or_outcomes_known": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent explicit cross-quote sign-disagreement resolution mechanism"},
        "stopping_rule": "terminal first failure; no symbols, block, normalization, activity, disagreement, variation, clock, hold, side, subset, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    if payload.get("manifest_hash") != canonical_hash({key: value for key, value in payload.items() if key != "manifest_hash"}):
        raise RuntimeError("HVDQDR preregistration drift")
    for path, expected in ((FLOW, FLOW_SHA), (FLOW_MANIFEST, FLOW_MANIFEST_SHA), (MARKET, MARKET_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"HVDQDR source drift: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    result = build(); validate(result); args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); print(args.output)
