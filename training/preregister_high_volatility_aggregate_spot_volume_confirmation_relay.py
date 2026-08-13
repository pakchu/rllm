"""Outcome-blind preregistration for HVASVC-24."""
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


POLICY_ID = "HVASVC-24"
DEFAULT_OUTPUT = Path("results/high_volatility_aggregate_spot_volume_confirmation_relay_preregistration_2026-08-13.json")
ENDPOINT = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ("volume_reported_spot_usd_1d", "AssetEODCompletionTime")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_aggregate_spot_volume_confirmation_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-13",
        singleton=True,
        outcomes_opened=False,
        source_incidence_opened=False,
        gross9_rows_opened=False,
        mechanism={
            "claim": (
                "A causal upper-tail day in reported aggregate BTC spot dollar volume confirms that a completed "
                "high-variation Bitcoin move was broadly transacted across spot venues rather than produced only "
                "by one derivatives venue. Follow the completed Bitcoin direction for one day."
            ),
            "side": "strict sign of BTCUSDT return over [decision-24h,decision); exact zero rejects",
            "external_support": {
                "paper": "Balcilar, Bouri, Gupta and Roubaud (2017), Can volume predict Bitcoin returns and volatility? A quantiles-based approach, Economic Modelling 64, 74-81",
                "doi": "10.1016/j.econmod.2017.03.019",
                "reported_fact": "The peer-reviewed paper reports that Bitcoin trading volume predicts returns in bullish and bearish return states while predictability is weak around normal return quantiles.",
                "official_metric_definition": "Coin Metrics volume_reported_spot_usd_1d aggregates reported BTC spot trading volume in US dollars across its covered spot markets at daily frequency.",
                "inference_disclosure": (
                    "Using an upper-tail aggregate-volume state to confirm the sign of the simultaneously completed "
                    "BTC day, a completion-time clock, causal variation rank, Binance execution and 24-hour hold is "
                    "a preregistered adaptation rather than a replication of the paper's quantile-causality test."
                ),
            },
            "why_distinct": (
                "Exact repository scans found no volume_reported_spot_usd_1d candidate. Existing volume candidates "
                "use Binance single-venue intraday base/quote volume, aggressor flow, trade counts, or fitted price "
                "features; HVASVC uses an independently completed multi-market aggregate spot-volume day as a "
                "cross-venue confirmation state and reuses no prior event set or control."
            ),
            "why_suited_to_volatile_regimes": "Both aggregate spot volume and completed BTC variation must occupy causal upper tails, directly targeting July-like volatile states.",
            "why_low_gross9_overlap_is_plausible": "Coin Metrics asset-completion timestamps and an external aggregate spot-volume state are absent from Gross9 primitives.",
        },
        features={
            "source_day": "Coin Metrics BTC UTC daily observation D at the frozen current download vintage",
            "aggregate_spot_volume": "exact finite positive volume_reported_spot_usd_1d for D",
            "availability": "D is eligible only if AssetEODCompletionTime is finite, strictly after D+1 00:00 UTC and no later than D+1 12:00 UTC; late rows reject",
            "volume_rank": "strict-prior midrank against at most 270 prior source-valid daily volumes, minimum 180, current excluded; rank>=0.75",
            "btc_return": "log(BTCUSDT open at decision / open at decision-24h) over an exact 1,440-row minute grid; strict nonzero",
            "btc_variation": "sqrt(sum squared one-minute log(close/open) returns) over the same exact [decision-24h,decision) grid",
            "btc_variation_rank": "strict-prior midrank against at most 270 prior source-valid decisions, minimum 180, current excluded; rank>=0.65",
            "eligibility": "valid volume rank>=0.75, variation rank>=0.65, and strict nonzero completed BTC return",
            "missing": "missing, duplicate, nonpositive, late-completion, rank-history, or BTC-grid drift rejects; no imputation",
        },
        clock={
            "decision": "ceiling to the next exact five-minute boundary at or after AssetEODCompletionTime for D",
            "entry": "exact BTCUSDT five-minute open one five-minute bar after decision",
            "hold": "24 elapsed hours",
            "reservation": "global chronological half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
        },
        policy={
            "volume_prior_days": 270,
            "volume_prior_minimum": 180,
            "volume_midrank_min": 0.75,
            "variation_prior_days": 270,
            "variation_prior_minimum": 180,
            "variation_midrank_min": 0.65,
            "latest_completion_hour_after_d_plus_1": 12,
            "entry_delay_minutes_after_decision": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "coin_metrics": {
                "endpoint": ENDPOINT, "asset": "btc", "metrics": list(METRICS), "frequency": "1d",
                "start_time": "2022-01-01", "end_time_inclusive": "2026-07-29", "page_size": 10000,
                "current_vintage_not_historical_revision_archive": True,
                "catalog_metadata_opened_before_preregistration": True,
                "historical_values_opened_before_preregistration": False,
                "read_after_preregistration_commit": True, "read_only": True,
            },
            "btc_1m": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": ["no_volume_tail", "no_btc_variation_gate", "one_day_stale_volume", "direction_flip", "same_clock_forced_long"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "paper_and_coin_metrics_catalog_metadata_opened": True,
            "historical_aggregate_spot_volume_values_opened": False,
            "source_values_used_to_select_rule_or_threshold": False,
            "candidate_source_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_exact_metric_or_event_candidate_found": False,
            "prior_volume_event_sets_or_controls_reused": False,
            "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False,
            "selection_basis": "published state-dependent volume predictability, unopened free causal aggregate source, high-variation targeting, and exact repository absence",
        },
        stopping_rule=(
            "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, "
            "then RV20 q90 audit; no metric, vintage, volume, rank, return, variation, side, hold, clock, subset, "
            "source, comparator, or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVASVC preregistration drift")
    if value["outcomes_opened"] or value["source_incidence_opened"] or value["gross9_rows_opened"]:
        raise RuntimeError("HVASVC research boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); result = build(); validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
