"""Outcome-blind preregistration for HVMVRV-24."""
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


POLICY_ID = "HVMVRV-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_mvrv_valuation_dislocation_relay_preregistration_2026-08-13.json"
)
ENDPOINT = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ("CapMVRVCur", "AssetEODCompletionTime")


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
        protocol_version="high_volatility_mvrv_valuation_dislocation_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-13",
        mechanism={
            "claim": (
                "Bitcoin market value far above its strictly prior local relationship to realized cost basis "
                "represents valuation overextension and maps short, while a displacement below that relationship "
                "represents undervaluation and maps long. Trade only dislocations of at least one-half prior "
                "standard deviation during elevated causal BTC variation."
            ),
            "side": "MVRV local z-score>=+0.5 maps short; z-score<=-0.5 maps long; the interior is ineligible",
            "external_support": {
                "paper": (
                    "Palazzi, Raimundo Junior, and Klotzle (2026), From Network Fundamentals to "
                    "Macro-Financial Integration: The Evolving Predictability of Bitcoin Returns"
                ),
                "ssrn": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6199098",
                "reported_fact": (
                    "Using daily Bitcoin data through 2025, the paper reports that market-to-realized value "
                    "predicts Bitcoin returns in every identified regime."
                ),
                "official_metric_definition": (
                    "Coin Metrics defines CapMVRVCur as current market capitalization divided by realized market "
                    "capitalization and interprets unusually high values as potential overvaluation/local maxima "
                    "and unusually low values as potential undervaluation/local minima."
                ),
                "official_docs": "https://docs.coinmetrics.io/asset-metrics/market/capact1yrusd",
                "inference_disclosure": (
                    "The causal thirty-day local z-score, half-standard-deviation threshold, Binance execution, "
                    "high-variation gate, and twenty-four-hour hold are preregistered adaptations, not a published "
                    "trading-rule replication."
                ),
            },
            "why_distinct": (
                "Repository-wide exact scans found no CapMVRVCur, MVRV, market-to-realized-value, or realized-cost-"
                "basis candidate. HVAANV divides activity by market value; miner and blockspace candidates use "
                "hashrate, fees, counts, or topology. HVMVRV instead compares market value with the on-chain price "
                "at which the current coin supply last moved and reuses no prior event set or control."
            ),
            "why_suited_to_volatile_regimes": (
                "A valuation dislocation is admitted only when completed prior-day BTC variation occupies its "
                "causal upper thirty-five percent, directly targeting July-like volatile states."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "A conservative once-daily on-chain realized-cost-basis clock at 12:05 UTC is absent from Gross9 "
                "structural market clocks."
            ),
        },
        features={
            "source_day": "Coin Metrics BTC UTC daily observation D at the frozen current download vintage",
            "mvrv": "exact finite positive CapMVRVCur on D; no reconstruction or alternate capitalization metric",
            "availability": (
                "D is eligible only if AssetEODCompletionTime is finite, falls after D+1 00:00 UTC and no later "
                "than the fixed D+1 12:00 UTC decision; late rows reject"
            ),
            "local_reference": (
                "arithmetic mean and sample standard deviation of log(CapMVRVCur) over exactly the thirty source-"
                "valid observations D-30 through D-1; current excluded; missing dates reject"
            ),
            "local_z_score": (
                "(log(CapMVRVCur_D)-strict-prior 30-day mean)/strict-prior 30-day sample standard deviation; "
                "prior standard deviation must be finite and strict positive"
            ),
            "valuation_gate": "local z-score>=+0.5 or <=-0.5",
            "btc_variation": (
                "sqrt(sum squared one-minute log(close/open) returns) over 1,440 exact BTCUSDT rows in "
                "[decision-24h,decision)"
            ),
            "btc_variation_rank": (
                "strict-prior midrank versus at most 180 previous jointly source-valid daily decisions; minimum "
                "120; current excluded; rank>=0.65"
            ),
            "eligibility": "valid available source, valuation gate, and BTC variation rank>=0.65",
            "missing": "missing, duplicate, nonpositive, late-completion, nonconsecutive, or BTC-grid drift rejects; no imputation",
        },
        clock={
            "decision": "12:00 UTC on source day D+1 after frozen completion-time validation",
            "entry": "exact BTCUSDT five-minute open at 12:05 UTC",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open; consecutive daily positions are allowed",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "mvrv_reference_days": 30,
            "mvrv_reference_minimum": 30,
            "mvrv_absolute_z_min": 0.5,
            "variation_prior_days": 180,
            "variation_prior_minimum": 120,
            "variation_midrank_min": 0.65,
            "decision_utc_hour": 12,
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
                "mvrv_direction_flip",
                "one_day_stale_mvrv",
                "market_cap_only_local_z",
                "same_clock_forced_long",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "paper_abstract_and_official_metric_definition_opened": True,
            "community_catalog_metadata_opened": True,
            "community_catalog_reported_history_start": "2010-07-18",
            "community_catalog_reported_coverage_through": "2026-08-12",
            "source_value_rows_opened": False,
            "candidate_source_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_exact_mvrv_candidate_found": False,
            "adjacent_vix_transfer_entropy_candidate_known": True,
            "adjacent_candidate_outcomes_used_to_set_formula_side_hold_clock_or_threshold": False,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "paper-reported all-regime MVRV predictability, official valuation interpretation, causally "
                "timestamped free daily source, requested high-variation regime, and repository absence"
            ),
        },
        stopping_rule=(
            "terminal first-failure sequence: source contract/support, Gross9 novelty, train/test/eval/final strict "
            "economics, then RV20 q90 audit; no metric, vintage, reference, standardization, threshold, variation, "
            "side, hold, clock, subset, source, or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVMVRV preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVMVRV boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
