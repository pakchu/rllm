"""Outcome-blind preregistration for HVWODP-24."""
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


DEFAULT_OUTPUT = Path("results/high_volatility_who_outbreak_disclosure_pressure_relay_preregistration_2026-08-12.json")
API = "https://www.who.int/api/news/diseaseoutbreaknews"
API_HELP = "https://www.who.int/api/news/diseaseoutbreaknews/sfhelp"
FIELDS = ("Id", "SystemSourceKey", "DonId", "UrlName", "Title", "PublicationDate", "PublicationDateAndTime", "DateCreated", "LastModified")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_who_outbreak_disclosure_pressure_relay_v1",
        policy_id="HVWODP-24",
        as_of_date="2026-08-12",
        mechanism={
            "claim": (
                "Official WHO Disease Outbreak News publication pressure is a timestamped global health-uncertainty "
                "shock. Published COVID-era evidence reports that significant uncertainty increases are associated "
                "with higher Bitcoin returns. During elevated causal BTC variation, go long on a WHO publication day "
                "when the trailing 28-day disclosure count exceeds the immediately preceding 28 days and short when "
                "it is lower."
            ),
            "side": "positive recent-versus-prior 28-day WHO disclosure pressure maps long; negative maps short; equality is ineligible",
            "external_support": {
                "paper": "Economic Policy Uncertainty in China and Bitcoin Returns: Evidence From the COVID-19 Period (2021)",
                "url": "https://pubmed.ncbi.nlm.nih.gov/33777889/",
                "reported_fact": "The paper concludes that significant rises in uncertainty lead to higher Bitcoin returns during the COVID-19 period.",
                "inference_disclosure": (
                    "WHO outbreak-publication pressure as a health-uncertainty proxy, the 28-day comparison, and "
                    "application outside the paper's China EPU estimator are preregistered adaptations."
                ),
            },
            "why_distinct": (
                "Exact repository scans found no WHO, disease-outbreak, epidemic, pandemic-disclosure, health-emergency, "
                "or infectious-disease uncertainty trading clock. It uses no prior event set, weather observation, "
                "market direction, flow, funding, OI, premium, or promoted control."
            ),
            "why_suited_to_volatile_regimes": "Only publication-day decisions whose completed prior-24-hour BTC variation ranks in the upper 35% are admitted.",
            "why_low_gross9_overlap_is_plausible": "Sparse WHO publication-day clocks and official global-health disclosure pressure are absent from Gross9 primitives.",
        },
        features={
            "authority": "WHO Sitefinity Disease Outbreak News collection API documented by WHO",
            "membership": "every collection item in the frozen publication interval with unique Id, SystemSourceKey, DonId and UrlName; title and body are not signal inputs",
            "availability": "the exact WHO PublicationDateAndTime; daily source day D is acted on only at 12:00 UTC on D+1 after the entire UTC day is complete",
            "effective_day": "UTC calendar day of PublicationDateAndTime; PublicationDate must normalize to the same instant or the source fails closed",
            "daily_count": "number of unique WHO Disease Outbreak News items on each explicit UTC effective day, including zeros",
            "pressure": "sum daily_count over [D-27,D] minus sum over [D-55,D-28]",
            "event": "eligible only when daily_count[D] is positive and pressure is strictly nonzero",
            "side": "sign of pressure",
            "btc_variation": "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars in [decision-24h,decision)",
            "btc_variation_rank": "strict-prior midrank versus at most 270 prior publication-day source-valid decisions; minimum 180; current excluded; rank >=0.65",
            "missing": "API identity, pagination, timestamp, schema, uniqueness, or BTC bar completeness failure rejects; no imputation",
        },
        clock={
            "decision": "12:00 UTC on WHO effective publication day D+1",
            "entry": "exact BTCUSDT five-minute open 5 minutes after decision",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "recent_window_days": 28,
            "prior_window_days": 28,
            "publication_day_required": True,
            "publication_delay_days": 1,
            "decision_utc_hour": 12,
            "variation_prior_events": 270,
            "variation_prior_minimum": 180,
            "variation_midrank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "who_disease_outbreak_news": {
                "collection_api": API,
                "schema_help": API_HELP,
                "fields": list(FIELDS),
                "publication_start": "2022-01-01T00:00:00Z",
                "publication_end_exclusive": "2026-07-30T00:00:00Z",
                "odata_order": ["PublicationDate asc", "Id asc"],
                "page_size": 50,
                "follow_only_same_origin_odata_next_link": True,
                "download_after_preregistration": True,
                "read_only_snapshot": True,
            },
            "btc_1m": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": ["no_btc_volatility_gate", "outbreak_direction_flip", "one_day_stale_pressure", "raw_publication_day_forced_long", "same_clock_forced_long"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "who_api_schema_help_opened": True,
            "who_collection_items_or_incidence_opened": False,
            "source_values_used_to_fit_rule": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_who_outbreak_candidate_found": False,
            "cross_proxy_inference_disclosed": True,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "official timestamped WHO disclosures, published uncertainty-to-Bitcoin sign, and exact repository absence",
        },
        stopping_rule=(
            "terminal first-failure sequence: WHO source contract/support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90 audit; "
            "no API family, membership, field, timestamp, window, publication-day requirement, variation threshold, side, hold, clock, subset, source, or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVWODP preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVWODP boundary drift")
    if value["research_boundary"]["who_collection_items_or_incidence_opened"] is not False:
        raise RuntimeError("HVWODP source boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
