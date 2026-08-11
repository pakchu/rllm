"""Outcome-blind preregistration for HVCPFR-72."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any

POLICY_ID = "HVCPFR-72"
DEFAULT_OUTPUT = Path("results/high_volatility_commercial_paper_funding_relay_preregistration_2026-08-11.json")
SOURCE = Path("data/commercial_paper_alfred_causal_vintages_2020_2026.csv")
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_commercial_paper_funding_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-11",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": "A four-week expansion in the seasonally adjusted US commercial-paper stock is a broadening of short-term corporate dollar funding and maps long BTC; contraction maps short BTC. Trade only from a first-revised causal vintage and when completed BTC variation is elevated.",
            "side": "strict sign of the causal four-week COMPOUT change",
            "why_distinct": "The source is the Federal Reserve's weekly outstanding commercial-paper funding stock, not an interest rate, Treasury balance, repo composition, labor claim, ETF return, crypto price path, OI, funding rate, prior event artifact, repair, or promoted control.",
            "why_suited_to_volatile_regimes": "the funding-stock impulse is admitted only when completed seven-day BTC variation ranks in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "one Friday 05:10 UTC first-revised weekly dollar-funding clock is absent from Gross9 primitives",
        },
        "external_basis": {
            "official_series": "https://fred.stlouisfed.org/series/COMPOUT",
            "official_release": "https://www.federalreserve.gov/releases/cp/default.htm",
            "definition": "Board of Governors Commercial Paper Outstanding, seasonally adjusted, billions of dollars, weekly ending Wednesday",
            "publication": "the Federal Reserve states the commercial-paper release is usually posted daily at 1:00 p.m.; the policy waits until Friday 05:05 UTC after the complete Thursday vintage date rather than assuming an intraday timestamp",
            "selection_use": "official funding-stock definition, weekly frequency, and conservative availability embargo only; no candidate incidence or outcomes",
        },
        "features": {
            "source": "ALFRED real-time vintages of FRED COMPOUT",
            "reference": "seasonally adjusted commercial paper outstanding for Wednesday D",
            "causal_vintage": "Thursday D+8 ALFRED vintage, after the following weekly release has supplied at least the first scheduled revision opportunity",
            "availability_embargo": "Friday D+9 at 05:05 UTC, later than every instant of the Thursday vintage date in America/New_York under EST or EDT",
            "observation": "COMPOUT[D] and COMPOUT[D-28] read only from the D+8 decision vintage",
            "change": "COMPOUT[D] - COMPOUT[D-28], finite strict nonzero",
            "direction": "positive funding-stock expansion maps long; negative contraction maps short",
            "btc_variation": "sqrt(sum squared exact completed five-minute BTC open-to-close log returns over seven elapsed days ending at decision)",
            "variation_rank": "strict-prior midrank over at most 270 source-valid weekly decisions, minimum 180, current excluded; rank>=0.65",
            "no_imputation": True,
            "no_latest_vintage_backfill": True,
        },
        "clock": {
            "decision": "Friday D+9 at 05:05 UTC after the first-revision Thursday vintage date",
            "entry": "decision+5m exact BTCUSDT five-minute open",
            "hold": "72 elapsed hours",
            "side": "four-week commercial-paper-stock change sign",
            "reservation": "source-time order, global half-open nonoverlap; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "change_weeks": 4,
            "first_revision_delay_days": 8,
            "variation_history_weeks": 270,
            "minimum_history_weeks": 180,
            "variation_rank_min": 0.65,
            "decision_utc_hour": 5,
            "decision_utc_minute": 5,
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
        "diagnostic_controls": {"definitions": {"no_variation_gate": "every valid first-revised four-week COMPOUT change", "one_release_stale_stock": "prior first-revised change with current variation", "current_first_print_stock": "D+1 first print instead of first-revised D", "direction_flip": "negative primary side", "same_clock_forced_long": "side +1 on primary clock"}, "cannot_be_promoted": True},
        "source_plan": {
            "commercial_paper": {"series": "COMPOUT", "vintage_provider": "ALFRED", "destination": str(SOURCE), "download_after_preregistration_commit": True, "one_decision_vintage_per_week": True},
            "historical_market": {"path": str(MARKET), "sha256": MARKET_SHA},
            "live_extension": "read-only Postgres BTCUSDT 1m through 2026-08-01",
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "official_definition_and_release_schedule_read": True,
            "repository_commercial_paper_candidate_found": False,
            "prior_macro_outcomes_known": True,
            "prior_event_artifacts_or_controls_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent official weekly corporate dollar-funding stock with conservative causal vintage embargo",
        },
        "stopping_rule": "Terminal first failure; no series, vintage, delay, change horizon, variation, side, clock, hold, subset, threshold, comparator, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    if value.get("manifest_hash") != canonical_hash({key: item for key, item in value.items() if key != "manifest_hash"}):
        raise RuntimeError("HVCPFR preregistration drift")
    if SOURCE.exists():
        raise RuntimeError("HVCPFR source must not exist before preregistration freeze")
    if hashlib.sha256(MARKET.read_bytes()).hexdigest() != MARKET_SHA:
        raise RuntimeError("HVCPFR market drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
