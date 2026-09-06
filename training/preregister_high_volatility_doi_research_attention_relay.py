"""Outcome-blind preregistration for HVDRA-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_energy_technology_spillover_relay as template


DEFAULT_OUTPUT = Path("results/high_volatility_doi_research_attention_relay_preregistration_2026-08-12.json")
CROSSREF_WORKS = "https://api.crossref.org/works"
TITLE_TERMS = ("bitcoin", "cryptocurrency", "cryptocurrencies", "cryptoasset", "cryptoassets", "crypto asset", "crypto assets")
WORK_TYPES = ("journal-article", "posted-content", "proceedings-article")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_doi_research_attention_relay_v1",
        policy_id="HVDRA-24",
        as_of_date="2026-08-12",
        mechanism={
            "claim": "New DOI registrations with explicit Bitcoin or cryptocurrency titles are an externally timestamped measure of formal research attention. Published Bitcoin evidence finds attention-return feedback lasting several days. During elevated causal BTC variation, follow the sign of the completed daily research-registration count relative to the same weekday one week earlier for one day.",
            "side": "strictly higher eligible DOI count than D-7 maps long; strictly lower maps short; equality is ineligible",
            "external_support": {
                "attention_study": "The link between Bitcoin and Google Trends attention (2021)",
                "attention_study_url": "https://arxiv.org/abs/2106.07104",
                "information_demand_study": "Information demand and cryptocurrency market activity, Economics Letters 2019",
                "information_demand_url": "https://doi.org/10.1016/j.econlet.2019.108714",
                "reported_fact": "The cited literature reports bidirectional Bitcoin-attention relationships over horizons up to several days and links information demand with cryptocurrency activity.",
                "inference_disclosure": "Transmitting formal scholarly-registration attention to broad BTC direction is a preregistered cross-domain attention inference, not a claimed direct estimate from DOI counts.",
            },
            "why_distinct": "Exact repository scans found no Crossref, DOI-registration, academic-publication, scholarly-attention, or research-submission trading clock. The signal uses no market direction, flow, funding, OI, premium, news article, regulatory action, physical hazard, sports result, lottery sale, prior event set, or promoted control.",
            "why_suited_to_volatile_regimes": "Only decisions whose completed prior-24-hour BTC variation ranks in the upper 35% are admitted.",
            "why_low_gross9_overlap_is_plausible": "External DOI creation dates and title-derived count signs are absent from Gross9 primitives.",
        },
        features={
            "authority": "Crossref REST API /works registry",
            "retrieval": "union of fixed query.title searches for bitcoin, cryptocurrency, and cryptoasset over the frozen created-date range; cursor pagination to exhaustion; local exact title grammar; duplicate DOI records must agree",
            "eligible_work": "lowercased nonempty DOI, exactly one nonempty title, type in the fixed three-type set, UTC created and latest deposited timestamps finite, created and deposited on the same UTC calendar day, and exact case-insensitive title token match",
            "title_grammar": "ASCII token boundaries around bitcoin, cryptocurrency, cryptocurrencies, cryptoasset, cryptoassets, crypto asset, or crypto assets; title only; abstract, subject, publisher, author, citation, score, and current index timestamp excluded",
            "causal_guard": "latest deposited UTC date must equal created UTC date; therefore no record whose metadata was redeposited on a later day can enter; decision is delayed two full calendar days",
            "daily_count": "number of unique eligible DOI identities created on UTC day D, including explicit zero-count days",
            "attention_change": "daily_count[D] - daily_count[D-7], strict nonzero; same-weekday comparison, no smoothing, normalization, tail threshold, or imputation",
            "side": "strict sign of attention_change",
            "btc_variation": "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars in [decision-24h,decision)",
            "btc_variation_rank": "strict-prior midrank versus at most 270 prior source-valid decisions; minimum 180; current excluded; rank >=0.65",
            "missing": "HTTP/cursor/schema drift, non-UTC timestamps, conflicting duplicate DOI, incomplete daily source range, or missing/duplicate/nonpositive BTC bars rejects as frozen; no imputation",
        },
        clock={
            "decision": "12:00 UTC on D+2 for each completed Crossref created UTC day D",
            "entry": "exact BTCUSDT five-minute open 5 minutes after decision",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "query_title_terms": ["bitcoin", "cryptocurrency", "cryptoasset"],
            "exact_title_terms": list(TITLE_TERMS),
            "work_types": list(WORK_TYPES),
            "publication_delay_days": 2,
            "decision_utc_hour": 12,
            "same_weekday_lag_days": 7,
            "variation_prior_days": 270,
            "variation_prior_minimum": 180,
            "variation_midrank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "crossref": {
                "url": CROSSREF_WORKS,
                "created_start": "2022-01-01T00:00:00Z",
                "created_end_exclusive": "2026-07-31T00:00:00Z",
                "rows_per_page": 1000,
                "cursor_to_exhaustion": True,
                "mailto_required": True,
                "download_after_preregistration": True,
                "read_only_snapshot": True,
            },
            "btc_1m": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": ["no_btc_volatility_gate", "attention_direction_flip", "one_day_stale_attention_change", "raw_day_over_day_change", "same_clock_forced_long"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "excluded_post_2024_feasibility_probe": "January 2025 Crossref query transport, selected fields, and broad query total only",
            "probe_opened_no_btc_price_return_funding_or_gross9_rows": True,
            "probe_not_used_to_choose_side_hold_rank_or_source_gate": True,
            "full_historical_candidate_incidence_opened": False,
            "source_values_used_to_fit_rule": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_doi_research_attention_candidate_found": False,
            "cross_domain_inference_disclosed": True,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "official DOI registry timestamps, later-day redeposit exclusion, published crypto attention evidence, dense daily source, and exact repository absence",
        },
        stopping_rule="terminal first-failure sequence: source contract/support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90 audit; no query, title grammar, work type, redeposit rule, delay, lag, variation threshold, side, hold, clock, subset, source, or control repair",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVDRA preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVDRA boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
