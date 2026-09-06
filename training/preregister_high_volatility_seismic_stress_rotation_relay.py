"""Outcome-blind preregistration for HVSSR-24."""
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


DEFAULT_OUTPUT = Path("results/high_volatility_seismic_stress_rotation_relay_preregistration_2026-08-12.json")
USGS_EVENT_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_seismic_stress_rotation_relay_v1",
        policy_id="HVSSR-24",
        as_of_date="2026-08-12",
        mechanism={
            "claim": "Published event-study evidence reports that environmental crises negatively affect established cryptocurrencies after an initial response. During elevated causal BTC variation, short after an unusually large increase in completed global significant-earthquake stress and long after an unusually large decrease.",
            "side": "increase in global daily seismic-stress proxy maps short; decrease maps long",
            "external_support": {
                "directional_study": "Investigating the impact of global events on cryptocurrency performance: a big data event study approach, Journal of International Money and Finance 157 (2025), 103375",
                "directional_study_doi": "https://doi.org/10.1016/j.jimonfin.2025.103375",
                "reported_fact": "Environmental crises negatively affect established cryptocurrencies after an initial positive response.",
                "official_source": "USGS Advanced National Seismic System Comprehensive Earthquake Catalog",
                "official_api_documentation": "https://earthquake.usgs.gov/fdsnws/event/1/",
                "official_catalog_documentation": "https://earthquake.usgs.gov/data/comcat/index.php",
                "inference_disclosure": "Using global magnitude-five-or-greater seismic energy as the objective environmental-crisis intensity measure is a preregistered cross-literature inference, not a claimed replication of the event-study classification.",
                "implementation_choices_not_claimed_as_replication": [
                    "global earthquake events with magnitude at least 5.0",
                    "causally reconstructed event-product versions available by D+2 12:00 UTC",
                    "daily sum of 10^(1.5*magnitude) as a monotone seismic-stress proxy",
                    "strict-prior absolute stress-change rank of 0.65",
                    "causal prior-24-hour BTC variation rank of 0.65",
                    "24-hour BTC hold beginning five minutes after decision",
                ],
            },
            "why_distinct": "Exact repository scans found no earthquake, seismic, or USGS candidate. The source is a physical global hazard clock and uses no market direction, flow, funding, OI, premium, air pollution, weather, lunar phase, geomagnetic value, prior event set, or promoted control.",
            "why_suited_to_volatile_regimes": "The physical stress change is admitted only when completed prior-24-hour BTC realized variation ranks in its causal upper 35%, targeting July-like volatile conditions.",
            "why_low_gross9_overlap_is_plausible": "Sparse physical-hazard entries are absent from Gross9 primitives and independent of its market-state clocks.",
        },
        features={
            "event_universe": "global USGS ComCat earthquake events with magnitude >=5.0 and origin time in UTC day D",
            "causal_event_version": "for decision D+2 12:00 UTC, include only events first published by that time and use the latest origin/magnitude product version whose updateTime is <= decision; deleted and superseded product histories must be requested",
            "daily_seismic_stress": "sum 10^(1.5*magnitude) across the causal event versions for UTC day D",
            "stress_change": "daily seismic stress D minus D-1; zero is ineligible",
            "stress_change_rank": "strict-prior midrank of absolute stress change versus at most 180 prior complete changes; minimum 60; current excluded; rank >=0.65",
            "side": "positive stress change maps -1; negative maps +1",
            "btc_variation": "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars in [decision-24h,decision)",
            "btc_variation_rank": "strict-prior midrank versus at most 180 prior complete decisions; minimum 60; current excluded; rank >=0.65",
            "missing": "inaccessible event version history, ambiguous publication/update time, malformed catalog product, missing/duplicate/nonpositive BTC bars, or warmup makes the decision ineligible or rejects as frozen; no imputation",
        },
        clock={
            "decision": "12:00 UTC on D+2 for completed UTC seismic day D",
            "entry": "exact BTCUSDT 12:05 UTC five-minute open",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "minimum_magnitude": 5.0,
            "source_lag_hours_after_day_end": 36,
            "stress_prior_days": 180,
            "stress_prior_minimum": 60,
            "stress_change_midrank_min": 0.65,
            "variation_prior_days": 180,
            "variation_prior_minimum": 60,
            "variation_midrank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "usgs": {"url": USGS_EVENT_API, "format": "geojson", "eventtype": "earthquake", "include_deleted": True, "include_superseded_products": True, "download_after_preregistration": True, "read_only_snapshot": True},
            "btc_1m": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": ["no_btc_volatility_gate", "seismic_direction_flip", "one_day_stale_seismic_stress", "stress_rise_only", "stress_fall_only", "same_clock_forced_long"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "usgs_api_schema_read_before_preregistration": True,
            "earthquake_event_values_or_candidate_incidence_opened": False,
            "source_values_used_to_select_rule": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_seismic_candidate_found": False,
            "cross_literature_inference_disclosed": True,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "published environmental-crisis sign, official physical-event archive, causal product-version rule, and exact repository absence",
        },
        stopping_rule="terminal first-failure sequence: source-version support, incidence support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90 audit; no event type, magnitude, lag, stress transform, threshold, side, hold, clock, subset, source, or control repair",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVSSR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVSSR boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); result = build(); validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
