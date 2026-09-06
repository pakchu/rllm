"""Outcome-blind preregistration for HVEXRP-24."""
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


POLICY_ID = "HVEXRP-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_exchange_reserve_pressure_relay_preregistration_2026-08-13.json"
)
ENDPOINT = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ("SplyExNtv", "AssetEODCompletionTime")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_exchange_reserve_pressure_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-13",
        mechanism={
            "claim": (
                "A completed daily increase in Bitcoin held at tagged exchanges raises immediately available "
                "sell inventory and maps short, while a decrease relieves exchange-side sell pressure and maps "
                "long. Apply the signed reserve change only during elevated causal BTC variation."
            ),
            "side": "log exchange-reserve change>0 maps short; log exchange-reserve change<0 maps long; zero rejects",
            "external_support": {
                "paper": (
                    "Hoang and Baur (2023), Loaded for bear: Bitcoin private wallets, exchange reserves and prices, "
                    "International Review of Financial Analysis 86, 102495"
                ),
                "doi": "10.1016/j.irfa.2022.102495",
                "reported_fact": (
                    "The peer-reviewed paper reports a negative relation between daily Bitcoin exchange-reserve "
                    "changes and current and future returns, finds a significant effect on one-day-ahead returns, "
                    "and evaluates out-of-sample forecast accuracy."
                ),
                "official_metric_definition": (
                    "Coin Metrics aggregate exchange supply estimates native units controlled by addresses tagged "
                    "to its curated exchange universe; its aggregate exchange metrics use address-clustering "
                    "heuristics and are exposed as daily asset metrics."
                ),
                "official_docs": "https://docs.coinmetrics.io/resources/faqs",
                "inference_disclosure": (
                    "The log-change representation, completion-time-derived clock, high-variation filter, Binance "
                    "execution venue, and fixed twenty-four-hour hold are preregistered adaptations rather than a "
                    "replication of the paper's forecasting regression."
                ),
            },
            "why_distinct": (
                "Repository-wide exact and semantic scans found no exchange-reserve or SplyEx candidate. One older "
                "blockspace preregistration explicitly excluded exchange-tag metrics and did not define or test this "
                "event set. Flow, address-activity, MVRV, miner, and market-microstructure candidates do not measure "
                "the stock of BTC immediately held at tagged exchanges."
            ),
            "why_suited_to_volatile_regimes": (
                "Reserve-pressure direction is admitted only when completed trailing BTC variation ranks in its "
                "causal upper thirty-five percent, directly targeting July-like volatile states."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "The decision follows each row's causally recorded end-of-day completion time rather than a fixed "
                "market-session clock, so exact entries should be structurally irregular relative to Gross9."
            ),
        },
        features={
            "source_day": "Coin Metrics BTC UTC daily observation D at the frozen current download vintage",
            "exchange_reserve": "exact finite positive SplyExNtv on D and D-1; no USD conversion or reconstruction",
            "reserve_log_change": "log(SplyExNtv_D / SplyExNtv_D-1); exact zero rejects",
            "availability": (
                "D is eligible only if AssetEODCompletionTime is finite, strictly after D+1 00:00 UTC, and no later "
                "than D+1 12:00 UTC; D-1 must be a consecutive valid row; late rows reject"
            ),
            "btc_variation": (
                "sqrt(sum squared one-minute log(close/open) returns) over 1,440 exact BTCUSDT rows in "
                "[decision-24h,decision)"
            ),
            "btc_variation_rank": (
                "strict-prior midrank versus at most 180 previous source-valid daily decisions; minimum 120; "
                "current excluded; rank>=0.65"
            ),
            "eligibility": "valid nonzero reserve log change and BTC variation rank>=0.65",
            "missing": (
                "missing, duplicate, nonpositive, nonconsecutive, late-completion, completion-before-boundary, or "
                "BTC-grid drift rejects; no imputation"
            ),
        },
        clock={
            "decision": (
                "ceiling to the next exact five-minute boundary at or after AssetEODCompletionTime for source day D"
            ),
            "entry": "exact BTCUSDT five-minute open one five-minute bar after decision",
            "hold": "24 elapsed hours",
            "reservation": (
                "global half-open; exit first on equal open; if a later daily entry occurs before the reserved exit, "
                "the later event skips"
            ),
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "reserve_change_lag_days": 1,
            "variation_prior_days": 180,
            "variation_prior_minimum": 120,
            "variation_midrank_min": 0.65,
            "completion_earliest_next_day_utc": "00:00",
            "completion_latest_next_day_utc": "12:00",
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "coin_metrics": {
                "endpoint": ENDPOINT,
                "asset": "btc",
                "metrics": list(METRICS),
                "frequency": "1d",
                "start_time": "2022-01-01",
                "end_time_inclusive": "2026-07-29",
                "page_size": 10000,
                "current_vintage_not_historical_revision_archive": True,
                "read_after_preregistration": True,
                "read_only": True,
            },
            "btc_1m": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "close"],
                "read_after_preregistration": True,
                "read_only": True,
            },
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": [
                "no_btc_variation_gate",
                "exchange_reserve_direction_flip",
                "one_day_stale_reserve_change",
                "exchange_reserve_level_rank",
                "same_clock_forced_long",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "paper_abstract_and_section_summary_opened": True,
            "official_metric_definition_opened": True,
            "community_catalog_metadata_opened": True,
            "community_catalog_reported_history_start": "2011-04-24",
            "community_catalog_reported_coverage_through": "2026-08-12",
            "source_value_rows_opened": False,
            "candidate_source_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_exact_exchange_reserve_candidate_found": False,
            "btc_exchange_netflow_nonpredictability_paper_known": True,
            "btc_exchange_netflow_candidate_rejected_before_preregistration": True,
            "adjacent_candidate_outcomes_used_to_set_formula_side_hold_clock_or_threshold": False,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "peer-reviewed one-day-ahead reserve-change direction, free causally timestamped daily source, "
                "requested high-variation regime, irregular availability clock, and repository absence"
            ),
        },
        stopping_rule=(
            "terminal first-failure sequence: source contract/support, Gross9 novelty, train/test/eval/final strict "
            "economics, then RV20 q90 audit; no metric, vintage, change transform, variation history, threshold, "
            "side, hold, clock, subset, source, or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVEXRP preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVEXRP boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
