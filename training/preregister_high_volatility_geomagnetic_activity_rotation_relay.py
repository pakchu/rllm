"""Outcome-blind preregistration for HVGMR-24."""
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


DEFAULT_OUTPUT = Path("results/high_volatility_geomagnetic_activity_rotation_relay_preregistration_2026-08-12.json")
GFZ_API = "https://kp.gfz.de/app/json/"


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_geomagnetic_activity_rotation_relay_v1",
        policy_id="HVGMR-24",
        as_of_date="2026-08-12",
        mechanism={
            "claim": "Geomagnetic disturbances impair digital infrastructure and are associated in behavioral-finance evidence with lower risky-asset returns. A large completed daily rise in the official planetary Kp index maps short BTC and a large fall maps long BTC, only when causal BTC variation is already elevated.",
            "side": "negative sign of the completed UTC-day mean Kp change from the immediately preceding complete UTC day; zero is ineligible",
            "external_support": {
                "bitcoin_study": "Michaelides, Prelorentzos, Scaillet, Topaloglou, and Tran (2025), Natural Hazards and Digital Finance: The Impact of Solar Storms on Bitcoin Activity",
                "bitcoin_study_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6005034",
                "bitcoin_reported_fact": "Geomagnetic disturbances significantly and time-varyingly suppress Bitcoin market activity.",
                "return_direction_study": "Krivelyova and Robotti, Playing the Field: Geomagnetic Storms and the Stock Market",
                "return_direction_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=375702",
                "return_reported_fact": "Elevated geomagnetic activity is associated with lower stock-market returns across countries.",
                "official_index": "GFZ Helmholtz Centre for Geosciences planetary Kp index",
                "official_data_page": "https://kp.gfz.de/en/data",
                "implementation_choices_not_claimed_as_replication": [
                    "arithmetic mean of eight completed three-hourly Kp nowcast-archive values per UTC day",
                    "strict-prior 270-day absolute-change midrank tail at 0.65",
                    "decision at 12:00 UTC on the next calendar day",
                    "independent causal prior-24-hour BTC variation-rank gate at 0.65",
                    "24-hour BTC hold beginning five minutes after decision",
                ],
            },
            "why_distinct": "Exact path, history, and mechanism scans found no geomagnetic, solar-storm, Kp-index, sunspot, or space-weather candidate. The signal uses an external physical index and no calendar phase, weather temperature, market direction, flow, funding, OI, premium, prior event set, or promoted control.",
            "why_suited_to_volatile_regimes": "Both the geomagnetic daily change and completed BTC variation must enter independently frozen causal upper-tail ranks, isolating large external disturbances during July-like volatile BTC conditions.",
            "why_low_gross9_overlap_is_plausible": "A delayed physical space-weather clock is absent from Gross9 market-state primitives.",
        },
        features={
            "kp_source": "GFZ archived three-hourly planetary Kp nowcast values, eight exact intervals per UTC day",
            "daily_kp": "arithmetic mean of the eight Kp values starting 00:00 through 21:00 UTC on source day D",
            "kp_change": "daily_kp[D]-daily_kp[D-1], requiring exact consecutive complete days and strict nonzero change",
            "kp_change_rank": "strict-prior midrank of absolute kp_change versus at most 270 prior valid changes; minimum 180; current excluded; rank >=0.65",
            "availability": "D+1 12:00 UTC, deliberately later than completion of all eight three-hour intervals and routine near-real-time provision",
            "btc_variation": "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars in [decision-24h,decision)",
            "btc_variation_rank": "strict-prior midrank versus at most 270 prior complete daily decisions; minimum 180; current excluded; rank >=0.65",
            "missing": "missing/duplicate/nonfinite Kp intervals, a non-nowcast archive status, or missing/duplicate/nonpositive BTC bars reject or make the day ineligible as frozen; no imputation",
        },
        clock={
            "source_day": "exact completed UTC day D",
            "decision": "D+1 12:00 UTC",
            "entry": "exact BTCUSDT D+1 12:05 UTC five-minute open",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "kp_intervals_per_day": 8,
            "kp_change_prior_days": 270,
            "kp_change_prior_minimum": 180,
            "kp_change_midrank_min": 0.65,
            "variation_prior_days": 270,
            "variation_prior_minimum": 180,
            "variation_midrank_min": 0.65,
            "publication_delay_hours_after_day": 12,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "kp": {"url": GFZ_API, "index": "Kp", "archive_status": "nowcast", "window": ["2022-12-01T00:00:00Z", "2026-08-01T00:00:00Z"], "download_after_preregistration": True, "read_only_snapshot": True},
            "btc_1m": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": ["no_kp_change_tail", "no_btc_volatility_gate", "one_day_stale_kp_change", "direction_flip", "same_clock_forced_long"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "gfz_api_schema_and_status_semantics_read_before_preregistration": True,
            "kp_values_or_candidate_incidence_opened": False,
            "source_values_used_to_select_rule": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_geomagnetic_candidate_found": False,
            "bitcoin_return_direction_is_cross_literature_inference": True,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent Bitcoin activity evidence, published risky-return direction, official near-real-time physical index, and exact repository absence",
        },
        stopping_rule="terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90 audit; no Kp aggregation, archive status, delay, tail, volatility threshold, side, hold, clock, subset, physical proxy, or control repair",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVGMR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVGMR boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
