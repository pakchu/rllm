"""Outcome-blind preregistration for HVEXDP-24."""
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


POLICY_ID = "HVEXDP-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_exchange_deposit_pressure_relay_preregistration_2026-08-13.json"
)
ENDPOINT = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ("FlowInExNtv", "AssetEODCompletionTime")


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
        protocol_version="high_volatility_exchange_deposit_pressure_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-13",
        mechanism={
            "claim": (
                "An increase in completed daily Bitcoin deposits to tagged exchanges raises newly delivered sell "
                "inventory and maps short, while a decrease in deposit flow relieves marginal exchange-side sell "
                "pressure and maps long. Apply the signed deposit-flow change only during elevated causal BTC variation."
            ),
            "side": "log exchange-deposit change>0 maps short; log exchange-deposit change<0 maps long; zero rejects",
            "external_support": {
                "paper": (
                    "Hoang and Baur (2023), Loaded for bear: Bitcoin private wallets, exchange reserves and prices, "
                    "International Review of Financial Analysis 86, 102495"
                ),
                "doi": "10.1016/j.irfa.2022.102495",
                "reported_fact": (
                    "The peer-reviewed paper reports that transfers of Bitcoin onto exchanges create immediate "
                    "selling pressure and that increased exchange reserves predict short-horizon price falls."
                ),
                "official_metric_definition": (
                    "Coin Metrics defines exchange deposits as the sum of assets sent to tagged exchange addresses "
                    "during the interval; for Bitcoin it excludes change-output effects. Aggregate exchange flows "
                    "use address-clustering heuristics over a curated exchange universe."
                ),
                "official_docs": "https://docs.coinmetrics.io/asset-metrics/exchange/flowinexusd",
                "inference_disclosure": (
                    "Mapping the day-over-day log change of deposit flow symmetrically, plus the completion-time "
                    "clock, high-variation filter, Binance execution, and twenty-four-hour hold, is a preregistered "
                    "adaptation rather than a replication of the paper's reserve-change regression."
                ),
            },
            "why_distinct": (
                "Repository-wide exact and semantic scans found no exchange-deposit or FlowInEx candidate. One older "
                "blockspace preregistration explicitly excluded exchange-tag metrics and did not define or test this "
                "event set. The terminal HVEXRP candidate measured the stock held at exchanges; HVEXDP instead "
                "measures acceleration or deceleration in the gross daily BTC flow sent into tagged exchanges and "
                "reuses neither its signal nor its event set."
            ),
            "why_suited_to_volatile_regimes": (
                "Deposit-pressure direction is admitted only when completed trailing BTC variation ranks in its "
                "causal upper thirty-five percent, directly targeting July-like volatile states."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "The decision follows each row's causally recorded end-of-day completion time rather than a fixed "
                "market-session clock, so exact entries should be structurally irregular relative to Gross9."
            ),
        },
        features={
            "source_day": "Coin Metrics BTC UTC daily observation D at the frozen current download vintage",
            "exchange_deposit": "exact finite positive FlowInExNtv on D and D-1; no USD conversion or reconstruction",
            "deposit_log_change": "log(FlowInExNtv_D / FlowInExNtv_D-1); exact zero rejects",
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
            "eligibility": "valid nonzero deposit log change and BTC variation rank>=0.65",
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
            "deposit_change_lag_days": 1,
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
                "exchange_deposit_direction_flip",
                "one_day_stale_deposit_change",
                "exchange_deposit_level_rank",
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
            "shared_completion_and_btc_variation_rows_previously_opened_for_hvexrp": True,
            "shared_rows_used_to_set_formula_side_hold_clock_or_threshold": False,
            "FlowInExNtv_value_rows_previously_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_exact_exchange_deposit_candidate_found": False,
            "btc_exchange_netflow_nonpredictability_paper_known": True,
            "btc_exchange_netflow_candidate_rejected_before_preregistration": True,
            "adjacent_candidate_outcomes_used_to_set_formula_side_hold_clock_or_threshold": False,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "peer-reviewed one-day-ahead deposit-change direction, free causally timestamped daily source, "
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
        raise RuntimeError("HVEXDP preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVEXDP boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
