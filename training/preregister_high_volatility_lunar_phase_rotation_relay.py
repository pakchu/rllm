"""Outcome-blind preregistration for HVLPR-24."""
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


DEFAULT_OUTPUT = Path("results/high_volatility_lunar_phase_rotation_relay_preregistration_2026-08-12.json")
USNO_YEAR_URL = "https://aa.usno.navy.mil/api/moon/phases/year?year={year}"


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_lunar_phase_rotation_relay_v1",
        policy_id="HVLPR-24",
        as_of_date="2026-08-12",
        mechanism={
            "claim": "Published Bitcoin evidence reports significantly lower returns in the three-day full-moon interval than in the corresponding new-moon interval. During elevated causal BTC variation, hold long inside the new-moon interval and short inside the full-moon interval for twenty-four hours.",
            "side": "long for a decision within 36 elapsed hours of a USNO New Moon timestamp; short within 36 elapsed hours of a USNO Full Moon timestamp",
            "external_support": {
                "directional_study": "Wei, Hua-Yen (2023), Lunar effect, investor sentiment and cryptocurrency returns, National Central University",
                "directional_study_url": "https://ir.lib.ncu.edu.tw/handle/987654321/92631?locale=en-US",
                "reported_fact": "Bitcoin returns are significantly lower around full moons than around new moons for the one-day-before through one-day-after interval.",
                "contrary_evidence": "Kovacs (2025), Are Investors Really Moonstruck? Lunar Phases, Returns, and Volatility in Global Equities and Cryptocurrencies, reports insignificant phase differences in wider plus/minus seven-day windows.",
                "contrary_evidence_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5867668",
                "astronomical_authority": "United States Naval Observatory Astronomical Applications Department",
                "astronomical_api_documentation": "https://aa.usno.navy.mil/data/api.html",
                "implementation_choices_not_claimed_as_replication": [
                    "exact USNO UTC phase timestamps",
                    "absolute 36-hour phase window sampled only at 00:00 UTC decisions",
                    "causal prior-24-hour BTC realized-variation rank gate of 0.65",
                    "24-hour BTC hold beginning five minutes after the decision",
                ],
            },
            "why_distinct": "Exact repository path, history, and mechanism scans found no lunar, moon-phase, synodic, new-moon, or full-moon candidate. The clock is deterministic astronomy and uses no weekday, day-of-month, fitted calendar slot, market direction, flow, funding, OI, premium, prior event set, or promoted control.",
            "why_suited_to_volatile_regimes": "The phase clock is admitted only when the completed prior-24-hour BTC realized variation ranks in its causal upper 35%, targeting July-like volatile conditions.",
            "why_low_gross9_overlap_is_plausible": "Sparse astronomical phase-window entries are absent from Gross9 primitives and are independent of its market-state clocks.",
        },
        features={
            "phase_source": "USNO calendar-year Phases of the Moon JSON for 2023 through 2026",
            "eligible_phases": ["New Moon", "Full Moon"],
            "phase_window": "at exact daily 00:00 UTC decision, absolute elapsed distance to exactly one eligible phase timestamp is <=36 hours",
            "phase_tie_or_overlap": "ineligible; no nearest-phase fallback",
            "side": "New Moon maps +1 and Full Moon maps -1",
            "btc_variation": "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars in [decision-24h,decision)",
            "btc_variation_rank": "strict-prior midrank versus at most 270 prior complete daily-decision variations; minimum 180; current excluded; rank >=0.65",
            "missing": "missing/duplicate/nonpositive BTC bars, malformed USNO rows, duplicate eligible phase timestamps, or non-UTC phase data reject or make the decision ineligible as frozen; no imputation",
        },
        clock={
            "decision": "each exact 00:00 UTC boundary from 2023-07-01 through 2026-07-31",
            "entry": "exact BTCUSDT 00:05 UTC five-minute open",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "phase_window_hours": 36,
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
            "moon_phases": {"url_template": USNO_YEAR_URL, "years": [2023, 2024, 2025, 2026], "download_after_preregistration": True, "read_only_snapshot": True},
            "btc_1m": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": ["no_btc_volatility_gate", "phase_direction_flip", "one_day_stale_phase_window", "new_moon_only", "full_moon_only", "same_clock_forced_long"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "astronomical_api_schema_read_before_preregistration": True,
            "phase_timestamps_or_candidate_incidence_opened": False,
            "source_values_used_to_select_rule": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_lunar_candidate_found": False,
            "contrary_lunar_evidence_disclosed": True,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "direct published Bitcoin new-moon versus full-moon sign, deterministic official astronomy, and exact repository absence",
        },
        stopping_rule="terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90 audit; no phase set, window, volatility threshold, side, hold, clock, subset, astronomical proxy, or control repair",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVLPR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVLPR boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
