"""Outcome-blind preregistration for HVAPPR-24."""
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


DEFAULT_OUTPUT = Path("results/high_volatility_air_pollution_penalty_rotation_relay_preregistration_2026-08-12.json")
AIRNOW_HOURLY_URL = "https://files.airnowtech.org/airnow/{year}/{yyyymmdd}/HourlyData_{yyyymmddhh}.dat"


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_air_pollution_penalty_rotation_relay_v1",
        policy_id="HVAPPR-24",
        as_of_date="2026-08-12",
        mechanism={
            "claim": "Published cryptocurrency evidence reports that high air-quality-index levels negatively affect returns, especially for environmentally damaging proof-of-work assets. During elevated causal BTC variation, short after an unusually large rise in the completed New York City PM2.5 AQI and long after an unusually large fall.",
            "side": "daily New York City PM2.5 AQI increase maps short; decrease maps long",
            "external_support": {
                "directional_study": "Singh and Chakraborty (2025), Hazy returns: does air pollution affect cryptocurrencies?, Studies in Economics and Finance 42(4), 780-798",
                "directional_study_doi": "https://doi.org/10.1108/SEF-03-2024-0180",
                "reported_fact": "High AQI levels negatively affect cryptocurrency returns, with the penalty absent for lower-energy cryptocurrency categories.",
                "official_source": "U.S. EPA AirNow Data Management Center timestamped hourly public files",
                "official_source_documentation": "https://docs.airnowapi.org/docs/HourlyDataFactSheet.pdf",
                "official_availability_fact": "AirNow states that each hourly file contains publicly approved monitoring-site observations and is normally available about 45 minutes after collection.",
                "implementation_choices_not_claimed_as_replication": [
                    "New York City five-county PM2.5 monitors",
                    "EPA PM2.5 AQI breakpoint conversion from each complete monitor-day mean",
                    "strict-prior absolute AQI-change rank of 0.65",
                    "causal prior-24-hour BTC realized-variation rank of 0.65",
                    "24-hour BTC hold beginning five minutes after the 01:00 UTC decision",
                ],
            },
            "why_distinct": "Exact repository scans found no air-pollution, AQI, PM2.5, AirNow, or haze-return candidate. This environmental observation clock uses no prior event set, market direction, flow, funding, OI, premium, weekday, month phase, lunar phase, geomagnetic value, or promoted control.",
            "why_suited_to_volatile_regimes": "The pollution shock is admitted only when completed prior-24-hour BTC realized variation ranks in its causal upper 35%, targeting July-like volatile conditions.",
            "why_low_gross9_overlap_is_plausible": "Sparse EPA observation-driven entries are absent from Gross9 primitives and independent of its market-state clocks.",
        },
        features={
            "airnow_source": "one archived HourlyData file for every UTC hour of each completed day",
            "nyc_site_filter": "AirNow site identifiers with state FIPS 36 and county FIPS in 005,047,061,081,085",
            "pollutant": "PM2.5 in UG/M3 only",
            "monitor_day": "arithmetic mean of 18 or more distinct valid hourly observations for one site; duplicate site-hour or nonfinite/negative concentration rejects the day",
            "pm25_aqi": "truncate monitor-day mean to 0.1 UG/M3 and apply the frozen EPA PM2.5 AQI breakpoint table with integer rounding",
            "city_aqi": "maximum monitor AQI among all eligible New York City sites",
            "aqi_change": "city AQI for completed day D minus completed day D-1; zero is ineligible",
            "aqi_change_rank": "strict-prior midrank of absolute AQI change versus at most 180 prior complete changes; minimum 60; current excluded; rank >=0.65",
            "side": "positive AQI change maps -1; negative AQI change maps +1",
            "btc_variation": "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars in [decision-24h,decision)",
            "btc_variation_rank": "strict-prior midrank versus at most 180 prior complete daily-decision variations; minimum 60; current excluded; rank >=0.65",
            "missing": "missing hourly source file, malformed row, incomplete monitor coverage, missing/duplicate/nonpositive BTC bars, or rank warmup makes the decision ineligible or rejects as frozen; no imputation",
        },
        clock={
            "decision": "01:00 UTC on D+1 after all twenty-four AirNow hourly files for UTC day D should have been published",
            "entry": "exact BTCUSDT 01:05 UTC five-minute open",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "air_quality_prior_days": 180,
            "air_quality_prior_minimum": 60,
            "air_quality_change_midrank_min": 0.65,
            "variation_prior_days": 180,
            "variation_prior_minimum": 60,
            "variation_midrank_min": 0.65,
            "minimum_monitor_hours": 18,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "airnow": {"url_template": AIRNOW_HOURLY_URL, "utc_days": ["2023-04-30", "2026-07-31"], "download_after_preregistration": True, "read_only_snapshot": True},
            "btc_1m": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": ["no_btc_volatility_gate", "pollution_direction_flip", "one_day_stale_pollution", "aqi_rise_only", "aqi_fall_only", "same_clock_forced_long"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "airnow_schema_and_one_historical_file_read_before_preregistration": True,
            "multi_day_air_quality_values_or_candidate_incidence_opened": False,
            "source_values_used_to_select_rule": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_air_pollution_candidate_found": False,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "direct published cryptocurrency air-pollution sign, timestamped official hourly observations, and exact repository absence",
        },
        stopping_rule="terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90 audit; no pollutant, city, site set, AQI construction, threshold, side, hold, clock, subset, source, or control repair",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVAPPR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVAPPR boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
