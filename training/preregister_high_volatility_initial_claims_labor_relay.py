"""Outcome-blind preregistration for HVICLR-72."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVICLR-72"
DEFAULT_OUTPUT = Path("results/high_volatility_initial_claims_labor_relay_preregistration_2026-08-09.json")
SOURCE = Path("data/initial_claims_alfred_causal_vintages_2020_2026.csv")
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_initial_claims_labor_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": "A four-week rise in finalized US initial unemployment claims is a deterioration in high-frequency labor demand and maps short BTC; a decline maps long BTC. Trade only after the first revision is public and BTC variation is elevated.",
            "side": "negative strict sign of the causal four-week ICSA change",
            "why_distinct": "The source is the official weekly labor-claims level, not a market reaction, financial-conditions composite, policy-uncertainty index, price path, or previously opened diagnostic direction.",
            "why_suited_to_volatile_regimes": "the labor signal is admitted only when completed seven-day BTC variation is in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "a DST-aware Thursday 08:40 America/New_York revised-claims clock is absent from Gross9",
        },
        "features": {
            "source": "ALFRED real-time vintages of FRED ICSA",
            "reference": "seasonally adjusted initial claims for observation Saturday D",
            "causal_vintage": "the second Thursday after D, twelve calendar days later, after the normal 08:30 America/New_York release",
            "observation": "ICSA[D] and ICSA[D-28] read from only that decision-date ALFRED vintage; D has received its first scheduled revision",
            "change": "ICSA[D] - ICSA[D-28], strict nonzero",
            "direction": "positive means labor deterioration and maps short; negative means improvement and maps long",
            "btc_variation": "sqrt(sum squared exact completed 5m BTC log returns over seven elapsed days ending at the 08:35 America/New_York decision)",
            "variation_rank": "strict-prior midrank over at most 270 source-valid weekly decisions, minimum 180, current excluded; rank>=0.65",
            "no_imputation": True,
            "no_latest_vintage_backfill": True,
        },
        "clock": {
            "decision": "second Thursday after observation D at 08:35 America/New_York",
            "entry": "decision+5m exact BTCUSDT five-minute open",
            "hold": "72 elapsed hours",
            "side": "negative four-week claims-change sign",
            "reservation": "source-time order, global half-open nonoverlap; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not signal input; exact settlements after novelty",
        },
        "policy": {
            "change_weeks": 4,
            "first_revision_delay_days": 12,
            "variation_history_weeks": 270,
            "minimum_history_weeks": 180,
            "variation_rank_min": 0.65,
            "decision_minutes_after_release": 5,
            "entry_delay_minutes": 5,
            "hold_hours": 72,
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
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes all stages", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"definitions": {"no_variation_gate": "every valid finalized claims change", "one_release_stale_claims": "prior finalized four-week claims change with current variation", "current_unrevised_claims": "same-day first print rather than first-revised D", "direction_flip": "negative primary side", "same_clock_forced_long": "side +1 on primary clock"}, "cannot_be_promoted": True},
        "source_plan": {
            "claims": {"series": "ICSA", "vintage_provider": "ALFRED", "destination": str(SOURCE), "download_after_preregistration_commit": True, "one_decision_vintage_per_second_thursday": True},
            "historical_market": {"path": str(MARKET), "sha256": MARKET_SHA},
            "live_extension": "read-only Postgres BTCUSDT 1m through 2026-08-01",
            "execution_prices": "sealed until source and novelty pass",
        },
        "research_boundary": {
            "official_series": "https://fred.stlouisfed.org/series/ICSA",
            "official_release": "https://www.dol.gov/ui/data.pdf",
            "prior_macro_outcomes_known": True,
            "prior_event_sets_or_controls_reused": False,
            "exact_hviclr_incidence_or_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent official labor-demand mechanism with a first-revision causal embargo",
        },
        "stopping_rule": "terminal first failure; no source, vintage, delay, variation, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    if value.get("manifest_hash") != canonical_hash({k: v for k, v in value.items() if k != "manifest_hash"}):
        raise RuntimeError("HVICLR prereg drift")
    if SOURCE.exists():
        raise RuntimeError("HVICLR source must not exist before preregistration freeze")
    if hashlib.sha256(MARKET.read_bytes()).hexdigest() != MARKET_SHA:
        raise RuntimeError("HVICLR market drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
