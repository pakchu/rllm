"""Outcome-blind preregistration for HVMNSD-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVMNSD-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_macro_news_sentiment_dispersion_relay_"
    "preregistration_2026-08-13.json"
)
SOURCE = Path("data/frbsf_daily_news_sentiment_1980_2026_aug.xlsx")
DERIVED = Path("data/frbsf_daily_news_sentiment_1980_2026_aug.csv")
SOURCE_MANIFEST = Path(
    "data/frbsf_daily_news_sentiment_1980_2026_aug_manifest.json"
)
MARKET = Path(
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
)
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
SOURCE_SHA = "f9b13dfa907d1cba3e411aac5564f773bae9c249e215c09e721137218e65d85d"
DERIVED_SHA = "4ed53d522d650028d6aaace39a5c800ab88295bcf5f170c70497c494ffcd809e"
SOURCE_MANIFEST_SHA = (
    "255a7fa7df461616dc34f3b9ce660cb582b80f1479bf60642ced7e1011ea86d1"
)
URL = "https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_macro_news_sentiment_dispersion_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": (
                "A week-over-week increase in the dispersion of the official SF Fed "
                "Daily News Sentiment Index marks fragmented macro narratives and a "
                "risk-off state, while declining dispersion marks narrative convergence. "
                "During elevated completed BTC variation, trade against the strict sign "
                "of the completed dispersion change for one day."
            ),
            "side": (
                "negative strict sign of log(current seven-day population standard "
                "deviation / prior seven-day population standard deviation)"
            ),
            "why_distinct": (
                "HVNSR uses an extreme signed seven-day change in the level of news tone. "
                "HVMNSD instead uses the change in within-week cross-day dispersion, a "
                "second-moment disagreement proxy, has no sentiment-shock tail, and does "
                "not reuse or promote an HVNSR control."
            ),
            "why_suited_to_volatile_regimes": (
                "macro-narrative fragmentation should matter most when BTC's independently "
                "measured completed variation is already elevated"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "a delayed official macro-news dispersion clock is absent from Gross9"
            ),
        },
        "features": {
            "source": "Federal Reserve Bank of San Francisco Daily News Sentiment Index XLSX",
            "source_url": URL,
            "observation": (
                "finite official daily news sentiment for every exact calendar day D-13 "
                "through D; no interpolation or imputation"
            ),
            "availability": (
                "D+8 00:00 UTC, deliberately later than the weekly update covering D; "
                "the frozen official history is used under this conservative embargo"
            ),
            "current_dispersion": (
                "population standard deviation of the seven exact observations D-6:D"
            ),
            "prior_dispersion": (
                "population standard deviation of the seven exact observations D-13:D-7"
            ),
            "dispersion_change": (
                "log(current_dispersion/prior_dispersion), requiring both dispersions "
                "strictly positive and the change strictly nonzero"
            ),
            "btc_variation": (
                "sqrt(sum squared exact completed 5m BTC log returns over UTC day D+7, "
                "ending at decision)"
            ),
            "btc_variation_rank": (
                "strict-prior midrank over at most 270 source-valid decisions, minimum "
                "180, current excluded; rank>=0.65"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "D+8 00:00 UTC",
            "entry": "decision+5m BTCUSDT open",
            "hold": "24 elapsed hours",
            "side": "negative dispersion-change sign; rising dispersion short",
            "reservation": "daily nonoverlapping; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not signal input; exact settlements after novelty",
        },
        "policy": {
            "dispersion_window_days": 7,
            "dispersion_estimator": "population_standard_deviation_ddof_0",
            "variation_history_days": 270,
            "minimum_history_days": 180,
            "btc_variation_rank_min": 0.65,
            "publication_delay_days": 8,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
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
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional "
                "side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes all stages",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "definitions": {
                "no_btc_variation_gate": "dispersion-change direction on every valid day",
                "one_week_stale_dispersion": (
                    "dispersion change ending D-7 with current BTC variation"
                ),
                "direction_flip": "positive primary dispersion-change sign",
                "same_clock_forced_long": "side +1 on the primary clock",
                "signed_tone_change": (
                    "sign of sentiment[D]-sentiment[D-7] on the primary clock; diagnostic "
                    "attribution only and not eligible for promotion"
                ),
            },
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "news_sentiment": {
                "url": URL,
                "raw_path": str(SOURCE),
                "raw_sha256": SOURCE_SHA,
                "derived_path": str(DERIVED),
                "derived_sha256": DERIVED_SHA,
                "manifest_path": str(SOURCE_MANIFEST),
                "manifest_sha256": SOURCE_MANIFEST_SHA,
                "read_only_snapshot": True,
                "already_frozen_before_candidate_selection": True,
            },
            "historical_market": {"path": str(MARKET), "sha256": MARKET_SHA},
            "live_extension": "read-only Postgres BTCUSDT 1m through 2026-08-01",
            "execution_prices": "sealed until source and novelty pass",
        },
        "research_boundary": {
            "official_page": (
                "https://www.frbsf.org/research-and-insights/data-and-indicators/"
                "daily-news-sentiment-index/"
            ),
            "paper": "Measuring News Sentiment, Journal of Econometrics 228(2), 2022",
            "prior_macro_outcomes_known": True,
            "hvnsr_signed_tone_outcome_known": True,
            "prior_event_sets_or_controls_reused": False,
            "exact_candidate_incidence_or_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "independent official macro-news second-moment mechanism chosen to provide "
                "dense daily high-variation clocks without altering a failed candidate"
            ),
            "revision_limit": (
                "no real-time vintage archive is claimed; causal use rests on the frozen "
                "official snapshot plus the preregistered conservative D+8 embargo"
            ),
        },
        "stopping_rule": (
            "terminal first failure; no source, publication delay, dispersion estimator, "
            "window, variation, side, clock, hold, subset, threshold, comparator, or "
            "control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVMNSD preregistration drift")
    bindings = {
        SOURCE: SOURCE_SHA,
        DERIVED: DERIVED_SHA,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA,
        MARKET: MARKET_SHA,
    }
    for path, expected in bindings.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"HVMNSD source drift: {path}")


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
