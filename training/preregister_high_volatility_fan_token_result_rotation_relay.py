"""Outcome-blind preregistration for HVFTRR-12."""
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


DEFAULT_OUTPUT = Path("results/high_volatility_fan_token_result_rotation_relay_preregistration_2026-08-12.json")
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard?dates={year}&limit=1000"
TEAM_IDS = {"83": "Barcelona", "1068": "Atletico Madrid", "243": "Sevilla", "94": "Valencia", "89": "Real Sociedad"}


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build()); core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_fan_token_result_rotation_relay_v1", policy_id="HVFTRR-12", as_of_date="2026-08-12",
        mechanism={
            "claim": "Published fan-token event studies report negative abnormal returns after football losses, while separate Bitcoin evidence reports post-match local demand from underground sportsbook settlement. During elevated causal BTC variation, map completed wins by a fixed set of Spanish fan-token clubs long and losses short for twelve hours.",
            "side": "tracked-club win maps long and loss maps short; draws and tracked-club head-to-head matches are ineligible",
            "external_support": {
                "directional_study": "Do match surprises move fan token prices? Evidence from UEFA competitions, Finance Research Letters (2025)",
                "directional_study_url": "https://www.sciencedirect.com/science/article/pii/S154461232502598X",
                "reported_fact": "Losses, especially unexpected losses, produce significantly negative fan-token abnormal returns.",
                "bitcoin_demand_study": "Hiding in plain sight: Detecting underground sportsbooks through local Bitcoin demand, Journal of International Money and Finance (2026)",
                "bitcoin_demand_study_url": "https://www.sciencedirect.com/science/article/abs/pii/S0261560625002487",
                "reported_bitcoin_fact": "Immediately after national-team matches, restrictive-gambling countries show a temporary increase in local Bitcoin demand where shadow economies are large.",
                "official_token_support": "Socios identifies Barcelona, Atletico Madrid, Sevilla, Valencia, and Real Sociedad as Spanish fan-token clubs.",
                "official_token_url": "https://www.socios.com/es/fan-tokens-del-fc-barcelona/",
                "inference_disclosure": "Transmitting club-result fan-token mood to broad BTC direction is a preregistered cross-asset behavioral inference, not a claimed direct Bitcoin result-sign estimate.",
            },
            "why_distinct": "Exact repository scans found no football, soccer, match-result, sports-sentiment, ESPN, or fan-token result clock. The signal uses no market direction, flow, funding, OI, premium, physical hazard, environmental measure, calendar phase, prior event set, or promoted control.",
            "why_suited_to_volatile_regimes": "Only match-result decisions whose completed prior-24-hour BTC variation ranks in the upper 35% are admitted.",
            "why_low_gross9_overlap_is_plausible": "Sparse external match-result timestamps and signs are absent from Gross9 primitives.",
        },
        features={
            "league": "ESPN Spanish LALIGA scoreboard namespace esp.1",
            "tracked_team_ids": TEAM_IDS,
            "eligible_match": "status.type.completed is true, exactly one tracked team participates, and that team has an explicit winner boolean; draw and tracked-team head-to-head are ineligible",
            "decision_group": "scheduled kickoff UTC plus exactly 3 elapsed hours; simultaneous eligible matches are retained only when every tracked result maps to the same side, then represented once",
            "side": "winner true maps +1; winner false maps -1",
            "btc_variation": "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars in [decision-24h,decision)",
            "btc_variation_rank": "strict-prior midrank versus at most 180 prior complete eligible decision groups; minimum 60; current excluded; rank >=0.65",
            "missing": "schema drift, non-final status, missing winner flag, duplicate event/team identity, missing/duplicate/nonpositive BTC bars, or warmup makes the decision ineligible or rejects as frozen; no imputation",
        },
        clock={
            "decision": "scheduled La Liga kickoff UTC plus 3 elapsed hours",
            "entry": "exact BTCUSDT five-minute open 5 minutes after decision", "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open", "split_crossing_action": "skip", "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes", "no_imputation": True,
        },
        policy={
            "tracked_team_ids": sorted(TEAM_IDS), "post_kickoff_decision_hours": 3,
            "variation_prior_events": 180, "variation_prior_minimum": 60, "variation_midrank_min": 0.65,
            "entry_delay_minutes": 5, "hold_hours": 12, "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "match_results": {"url_template": ESPN_SCOREBOARD, "years": [2023, 2024, 2025, 2026], "download_after_preregistration": True, "read_only_snapshot": True},
            "btc_1m": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={"names": ["no_btc_volatility_gate", "result_direction_flip", "one_match_stale_result", "wins_only", "losses_only", "same_clock_forced_long"], "diagnostic_controls_cannot_be_promoted": True},
        research_boundary={
            "espn_schema_and_team_identity_read_before_preregistration": True, "multi_year_match_results_or_candidate_incidence_opened": False,
            "source_values_used_to_select_rule": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False,
            "repository_sports_result_candidate_found": False, "cross_asset_inference_disclosed": True, "prior_event_sets_reused": False,
            "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False,
            "selection_basis": "published fan-token loss sign, direct post-match Bitcoin-demand evidence, fixed official-token club set, and exact repository absence",
        },
        stopping_rule="terminal first-failure sequence: source contract/support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90 audit; no league, club set, result class, decision lag, threshold, side, hold, clock, subset, source, or control repair",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build(): raise RuntimeError("HVFTRR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False: raise RuntimeError("HVFTRR boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    result = build(); validate(result); args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); print(args.output)
