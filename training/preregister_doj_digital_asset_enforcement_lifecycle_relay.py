"""Outcome-blind preregistration for DDAELR-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/doj_digital_asset_enforcement_lifecycle_relay_preregistration_2026-08-09.json"
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "doj_digital_asset_enforcement_lifecycle_relay_v1",
        "policy_id": "DDAELR-24",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "Official DOJ digital-asset enforcement initiations expand legal and operating "
                "uncertainty around crypto rails, while completed convictions, sentences, seizures, "
                "forfeitures, recoveries, compensation, and infrastructure disruptions resolve an "
                "illicit-finance overhang. In an already volatile BTC regime, initiation maps short "
                "and resolution maps long for twenty-four elapsed hours."
            ),
            "side": "SHORT for unambiguous initiation; LONG for unambiguous resolution",
            "why_distinct": (
                "DDAELR uses point-in-time federal enforcement publications on irregular civil and "
                "criminal lifecycle clocks. It uses no market-derived direction, scheduled macro "
                "release, Gross9 clock, prior terminal event set, or diagnostic control."
            ),
            "why_suited_to_volatile_regimes": (
                "legal-rail shocks should transmit most strongly when BTC liquidity and risk appetite "
                "are already unstable; prior-24-hour realized variation must rank at least 0.65"
            ),
        },
        "authority": {
            "documentation": "https://www.justice.gov/developer/api-documentation/api_v1",
            "endpoint": "https://www.justice.gov/api/v1/press_releases.json",
            "resource": "DOJ News API press releases",
            "pagination": "sort=date&direction=ASC&pagesize=50&page=N",
            "fields": ["uuid", "date", "title", "body", "url", "updated"],
            "point_in_time_identity": (
                "UUID plus publication date and preserved raw response; later updated field is "
                "recorded but never moved backward into historical availability"
            ),
            "date_only_availability": "22:00:00 UTC on the displayed publication date",
        },
        "taxonomy": {
            "normalization": (
                "HTML-unescape; strip tags; Unicode NFKC; lowercase; collapse ASCII whitespace; "
                "match whole alphanumeric tokens or exact normalized phrases"
            ),
            "digital_asset_terms": [
                "bitcoin", "ethereum", "cryptocurrency", "crypto currency", "crypto asset",
                "virtual currency", "virtual asset", "digital currency", "digital asset",
                "stablecoin", "tether", "blockchain",
            ],
            "initiation_terms": [
                "charged", "charges", "indicted", "indictment", "arrested", "arrest",
                "complaint", "files civil action", "filed civil action", "unseals charges",
            ],
            "resolution_terms": [
                "convicted", "conviction", "sentenced", "sentencing", "pleads guilty",
                "pleaded guilty", "seizure", "seized", "forfeiture", "forfeited", "recovered",
                "returns funds", "compensation", "disrupts", "disrupted", "takedown",
            ],
            "classification": (
                "title plus body must contain a digital-asset term; initiation XOR resolution; "
                "both or neither is ineligible. Same publication day and side deduplicates to one "
                "event; opposite eligible sides on one day invalidate that day."
            ),
        },
        "volatility_gate": {
            "source": "BTCUSDT bars_binance 1m completed strictly before availability",
            "variation": (
                "sum squared one-minute close-to-close log returns over exact [availability-24h, "
                "availability); all 1440 timestamps required"
            ),
            "rank": (
                "strict-prior midrank at prior eligible DOJ publication days among at most 252 "
                "previous valid publication-day observations; minimum 126; current excluded; >=0.65"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "22:00 UTC on eligible DOJ publication date",
            "entry": "exact decision+5m BTCUSDT open",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; chronological events; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "exact funding only after Gross9 novelty passes",
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
            "names": ["no_volatility_gate", "title_only_taxonomy", "one_event_stale_side", "direction_flip"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "official_api_documentation_opened": True,
            "doj_press_release_values_or_incidence_opened": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "new immutable irregular federal enforcement lifecycle source",
        },
        "stopping_rule": (
            "terminal first failure: source transport/support, Gross9 novelty, then sequential "
            "economics; no agency, keyword, class, ambiguity, time, volatility, side, hold, or subset repair"
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
