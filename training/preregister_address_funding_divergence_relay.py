"""Freeze the outcome-blind AFDR-864 source and policy contract.

This module hashes frozen source and comparator bytes only. It must not parse
address values, funding values, comparator rows, BTC market bars, returns, or
PnL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "address_funding_divergence_relay_preregistration_v1"
CANDIDATE = "AFDR-864"
AS_OF_DATE = "2026-07-20"

ADDRESS_SOURCE = Path("data/coinmetrics_btc_address_reservoir_2019_2023.csv.gz")
ADDRESS_SOURCE_SHA256 = (
    "15550072f954d29ae4c9ffe16e11f07c492ee5b6b956e54654b14b9a7af5170a"
)
ADDRESS_MANIFEST = Path(
    "results/coinmetrics_btc_address_reservoir_source_manifest_2026-07-20.json"
)
ADDRESS_MANIFEST_SHA256 = (
    "f16827d26a1e095e504623c24f94cfd77af36d4466439cf203c4bf0f72ddad97"
)
ADDRESS_BUILDER = Path("training/download_coinmetrics_btc_address_reservoir_daily.py")
ADDRESS_BUILDER_SHA256 = (
    "40b759a06038782ebc6e676b3320b4f8eb360e097de9384e5f0f0db45c890c94"
)

FUNDING_SOURCE = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
FUNDING_SOURCE_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)
FUNDING_BUILDER = Path(
    "training/freeze_binance_um_btcusdt_funding_marks_2020_2023.py"
)
FUNDING_BUILDER_SHA256 = (
    "c8a2c1f9cd9bd19563a2721483e52f0f09e8bf83826f4c3d9feccb5da5882db3"
)

MECHANISM_DOCUMENT = Path(
    "docs/address-funding-divergence-relay-mechanism-decision-2026-07-20.md"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/address-funding-divergence-relay-preregistration-2026-07-20.md"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_address_funding_divergence_relay.py"
)
DEFAULT_OUTPUT = Path(
    "results/address_funding_divergence_relay_preregistration_2026-07-20.json"
)

COMPARATORS: tuple[dict[str, Any], ...] = (
    {
        "candidate": "NTB-7",
        "path": "results/network_topology_broadening_clock_2026-07-17.csv",
        "sha256": "6b1bd7c7458cffa062e40872c3ad1730007c01426790b1ba8e52c6eb853de42f",
        "format": "csv",
        "capability": "directional_interval",
        "filters": {"policy_id": ["NTB-7"]},
        "group_column": None,
        "comparison_start": "2021-01-01T00:00:00Z",
        "comparison_end_exclusive": "2024-01-01T00:00:00Z",
        "entry_column": "entry_date",
        "side_column": "side",
        "exit_column": "exit_date",
    },
    {
        "candidate": "CVTR-1",
        "path": (
            "results/cboe_volatility_term_rotation_preregistered_clock_"
            "2026-07-17.csv.gz"
        ),
        "sha256": "c0250d1f40c87049f6d7639ba43f5285835441399a62968434b65c7d46ed2a93",
        "format": "csv",
        "capability": "directional_interval",
        "filters": {"clock_mode": ["primary"]},
        "group_column": None,
        "comparison_start": "2021-01-01T00:00:00Z",
        "comparison_end_exclusive": "2024-01-01T00:00:00Z",
        "entry_column": "entry_time",
        "side_column": "side",
        "exit_column": "exit_time",
    },
    {
        "candidate": "ORFR-1",
        "path": "results/overnight_rrp_flow_release_preregistered_clock_2026-07-17.csv.gz",
        "sha256": "9f09bc88c9661441a33cee724e59524f57c0b021abff0fe81263e1a341b7b7b7",
        "format": "csv",
        "capability": "directional_interval",
        "filters": {"clock_mode": ["primary"]},
        "group_column": None,
        "comparison_start": "2021-01-01T00:00:00Z",
        "comparison_end_exclusive": "2024-01-01T00:00:00Z",
        "entry_column": "entry_time",
        "side_column": "side",
        "exit_column": "scheduled_exit_time",
    },
    {
        "candidate": "FLCC-1",
        "path": (
            "results/federal_liquidity_component_concordance_preregistered_"
            "clock_2026-07-17.csv.gz"
        ),
        "sha256": "7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c",
        "format": "csv",
        "capability": "directional_interval",
        "filters": {"clock_name": ["primary"]},
        "group_column": "candidate_id",
        "comparison_start": "2021-01-01T00:00:00Z",
        "comparison_end_exclusive": "2024-01-01T00:00:00Z",
        "entry_column": "entry_time",
        "side_column": "side",
        "exit_column": "exit_time",
    },
    {
        "candidate": "prior_microstructure",
        "path": "results/prior_microstructure_comparator_clock_bundle_2026-07-20.json",
        "sha256": "c5584256140799b380973f9f376e5751ad754a81c9683473467b9d05af0bb9f0",
        "format": "json_comparator_event_bundle",
        "capability": "timestamp_only",
        "filters": {},
        "group_column": "comparators.*",
        "comparison_start": "max(2021-01-01T00:00:00Z, coverage_start_inclusive)",
        "comparison_end_exclusive": (
            "min(2024-01-01T00:00:00Z, coverage_end_exclusive)"
        ),
        "entry_column": "signal_date",
        "side_column": None,
        "exit_column": None,
    },
    {
        "candidate": "BFMWD-144",
        "path": "data/bitfinex_margin_warehouse_deployment_clocks_2021_2023.csv.gz",
        "sha256": "02b4fcc462a5a48be7673649f4cf4b2f9bb210baca4294eed1696d479820cccc",
        "format": "csv",
        "capability": "directional_interval",
        "filters": {"control": ["primary"]},
        "group_column": "variant_id",
        "comparison_start": "2021-01-01T00:00:00Z",
        "comparison_end_exclusive": "2024-01-01T00:00:00Z",
        "entry_column": "entry_time",
        "side_column": "side",
        "exit_column": "exit_time",
    },
    {
        "candidate": "DLPD-12",
        "path": "data/btcdom_leverage_polarity_decomposition_clocks_2022_2023.csv.gz",
        "sha256": "b33990f1629465caa837aa1f6f74430054b7185b68ece47b8c7540f9c11bf0fb",
        "format": "csv",
        "capability": "directional_interval",
        "filters": {"control": ["primary"]},
        "group_column": None,
        "comparison_start": "2022-01-01T00:00:00Z",
        "comparison_end_exclusive": "2024-01-01T00:00:00Z",
        "entry_column": "entry_time",
        "side_column": "side",
        "exit_column": "exit_time",
    },
)


@dataclass(frozen=True)
class PolicyConfig:
    address_change_days: int = 7
    funding_window_hours: int = 72
    required_funding_settlements: int = 9
    funding_publication_delay_minutes: int = 5
    maximum_funding_slot_offset_ms: int = 60_000
    maximum_latest_funding_age_hours: int = 8
    reference_lookback_days: int = 365
    minimum_prior_observations: int = 180
    lower_rank: float = 0.25
    upper_rank: float = 0.75
    maximum_address_lag_days: float = 3.0
    entry_delay_minutes: int = 5
    bar_minutes: int = 5
    hold_bars: int = 864
    leverage: float = 0.5
    onset_only: bool = True
    global_nonoverlap: bool = True
    warmup_start: str = "2019-01-01T00:00:00Z"
    train_start: str = "2021-01-01T00:00:00Z"
    train_end_exclusive: str = "2023-01-01T00:00:00Z"
    selection_start: str = "2023-01-01T00:00:00Z"
    selection_end_exclusive: str = "2024-01-01T00:00:00Z"


FROZEN_CONFIG = PolicyConfig()

SUPPORT_GATES = {
    "minimum_train_events": 50,
    "minimum_events_each_train_year": 20,
    "minimum_selection_events": 25,
    "minimum_events_each_selection_half": 10,
    "minimum_train_events_each_side": 15,
    "minimum_selection_events_each_side": 7,
    "maximum_month_share_each_split": 0.20,
    "maximum_weekday_share_each_split": 0.35,
    "maximum_rolling_30day_share_each_split": 0.25,
    "maximum_exact_entry_jaccard": 0.10,
    "novelty_containment_hours": 6,
    "maximum_bidirectional_novelty_containment": 0.35,
    "maximum_absolute_signed_exposure_correlation": 0.40,
    "minimum_common_candidate_events": 10,
    "minimum_common_comparator_events": 5,
    "stop_if_failed": True,
}

SUPPORT_CONTRACT = {
    "split_containment": (
        "entry >= split_start and exit <= split_end_exclusive; never truncate"
    ),
    "nonoverlap": (
        "global chronological greedy per control; skip entry < prior accepted "
        "exit; entry == prior exit is admissible"
    ),
    "month_share": (
        "maximum UTC calendar-month entry count divided by split event count"
    ),
    "weekday_share": (
        "maximum UTC entry-weekday count divided by split event count"
    ),
    "rolling_30day_share": (
        "for every accepted entry t, count entries in [t,t+30 elapsed days); "
        "take maximum divided by split event count"
    ),
    "all_concentration_checks_are_per_split": True,
}

ECONOMIC_GATES = {
    "base_cost_bp_per_side": 6.0,
    "stress_cost_bp_per_side": 10.0,
    "minimum_cagr_to_strict_mdd": 3.0,
    "maximum_strict_mdd": 0.15,
    "minimum_stress_cagr_to_strict_mdd": 2.5,
    "minimum_mean_gross_side_adjusted_bp": 30.0,
    "require_positive_absolute_return": True,
    "require_each_contained_half_positive": True,
    "require_each_side_contribution_positive": True,
    "require_one_extra_bar_delay_positive": True,
    "weekly_cluster_one_sided_pvalue_maximum": 0.10,
    "weekly_cluster_test": {
        "input": "base-cost exact-funding net_return per accepted trade",
        "cluster": "UTC ISO year-week of entry_time",
        "statistic": "arithmetic mean of trade net_return",
        "null_randomization": (
            "multiply all trade returns in each cluster by one shared "
            "independent Rademacher sign"
        ),
        "alternative": "observed mean greater than zero",
        "draws": 100_000,
        "seed": 20_260_720,
        "cluster_order": "ascending UTC ISO (year, week)",
        "draw_order": (
            "one NumPy default_rng(seed) integers(0,2,size=(draws,n_clusters)) "
            "call mapped 0->-1 and 1->+1"
        ),
        "pvalue": "(1 + count(randomized_statistic >= observed)) / (draws + 1)",
        "run_separately_in_every_opened_split": True,
    },
    "full_calendar_cagr": True,
    "strict_path_mdd": True,
}

SOURCE_ONLY_CONTROLS = (
    "balance_only",
    "activity_only",
    "funding_only",
    "one_address_report_delay",
    "direction_flip",
    "deterministic_random_side",
)

CONTROL_CONTRACTS = {
    "balance_only": {
        "clock": "valid daily address feature clock",
        "long": "balance_growth_rank >= 0.75",
        "short": "balance_growth_rank <= 0.25",
        "event": "FLAT-to-nonzero onset",
    },
    "activity_only": {
        "clock": "valid daily address feature clock",
        "long": "activity_growth_rank >= 0.75",
        "short": "activity_growth_rank <= 0.25",
        "event": "FLAT-to-nonzero onset",
    },
    "funding_only": {
        "clock": "valid daily address/funding feature clock",
        "long": "funding_pressure_rank <= 0.25",
        "short": "funding_pressure_rank >= 0.75",
        "event": "FLAT-to-nonzero onset",
    },
    "one_address_report_delay": {
        "clock": "exact primary side moved to next valid address report",
        "feature_recomputed": False,
        "execution_and_nonoverlap_recomputed": True,
    },
    "direction_flip": {
        "clock": "exact primary entry and exit",
        "side": "negative primary side",
    },
    "deterministic_random_side": {
        "clock": "exact primary entry and exit",
        "material": "AFDR-864|20260720|<primary_entry_time_utc>",
        "side": "LONG when first SHA-256 byte < 128, otherwise SHORT",
    },
}

NOVELTY_CONTRACT = {
    "scope": (
        "exact per-comparator comparison_start/comparison_end_exclusive; "
        "never infer coverage from first or last event"
    ),
    "member_identity": (
        "each group_column value independently; ungrouped CSV is one member; "
        "each comparators.* JSON event list is one timestamp-only member"
    ),
    "timestamp_parse": "exact timezone-aware UTC; no rounding or date coercion",
    "exact_jaccard": "size(unique(candidate) intersect unique(comparator)) / size(union)",
    "candidate_near_share": (
        "fraction of unique candidate entries with any comparator entry at "
        "absolute elapsed distance <= 6h"
    ),
    "comparator_near_share": (
        "fraction of unique comparator entries with any candidate entry at "
        "absolute elapsed distance <= 6h"
    ),
    "containment_gate": "max(candidate_near_share, comparator_near_share) <= 0.35",
    "minimum_common_support": {
        "candidate_events": 10,
        "comparator_events": 5,
        "failure_action": "fail closed, never mark not-applicable",
    },
    "signed_exposure": {
        "applies_to": "directional_interval members only",
        "grid": "complete five-minute UTC opens over the clipped common scope",
        "interval": "entry-inclusive and exit-exclusive",
        "values": "LONG +1, SHORT -1, flat 0",
        "overlap": "any within-member overlap fails closed",
        "alignment": "entry and exit must lie exactly on five-minute UTC opens",
        "correlation": "ordinary Pearson correlation on the complete grid",
        "zero_variance": "fail closed",
        "gate": "absolute correlation <= 0.40",
    },
    "malformed_empty_or_unknown_capability": "fail closed",
}

STRICT_MDD_CONTRACT = {
    "initial_equity": 1.0,
    "global_high_water_mark": True,
    "pre_entry_high_water_mark": True,
    "entry_cost": True,
    "every_held_five_minute_adverse_path": True,
    "exact_realized_funding": True,
    "virtual_adverse_exit_cost_at_every_held_bar": True,
    "actual_exit_cost": True,
    "funding_boundary": (
        "exact entry/exit funding credits excluded; exact entry/exit debits retained"
    ),
    "full_calendar_idle_cash": True,
}

ADDRESS_COLUMNS = (
    "observation_date",
    "available_at",
    "AdrBalCnt",
    "AdrActCnt",
)
FUNDING_SIGNAL_COLUMNS = (
    "funding_time_ms",
    "funding_time_utc",
    "symbol",
    "funding_rate",
)
FUNDING_PHYSICAL_COLUMNS = (
    *FUNDING_SIGNAL_COLUMNS,
    "settlement_mark_price",
    "mark_open_time_ms",
    "mark_open_time_utc",
    "funding_time_offset_ms",
    "mark_source",
)


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_contract(config: PolicyConfig = FROZEN_CONFIG) -> None:
    if config != FROZEN_CONFIG:
        raise ValueError("AFDR-864 policy configuration is frozen")
    if config.address_change_days != 7:
        raise ValueError("AFDR-864 address change must remain seven days")
    if config.funding_window_hours != 72:
        raise ValueError("AFDR-864 funding window must remain 72 hours")
    if config.required_funding_settlements != 9:
        raise ValueError("AFDR-864 requires exactly nine funding settlements")
    if config.maximum_funding_slot_offset_ms != 60_000:
        raise ValueError("AFDR-864 funding slot tolerance must remain 60 seconds")
    if config.maximum_latest_funding_age_hours != 8:
        raise ValueError("AFDR-864 newest funding age must remain eight hours")
    if config.entry_delay_minutes != config.bar_minutes:
        raise ValueError("AFDR-864 must retain one complete latency bar")
    if config.hold_bars * config.bar_minutes != 72 * 60:
        raise ValueError("AFDR-864 hold must remain exactly 72 hours")
    if not 0 < config.lower_rank < config.upper_rank < 1:
        raise ValueError("AFDR-864 rank tails are invalid")
    if config.lower_rank != 1 - config.upper_rank:
        raise ValueError("AFDR-864 rank tails must remain symmetric")
    if set(CONTROL_CONTRACTS) != set(SOURCE_ONLY_CONTROLS):
        raise ValueError("AFDR-864 control contracts are incomplete")
    if NOVELTY_CONTRACT["minimum_common_support"] != {
        "candidate_events": SUPPORT_GATES["minimum_common_candidate_events"],
        "comparator_events": SUPPORT_GATES["minimum_common_comparator_events"],
        "failure_action": "fail closed, never mark not-applicable",
    }:
        raise ValueError("AFDR-864 novelty common-support contract drifted")
    weekly = ECONOMIC_GATES["weekly_cluster_test"]
    if not isinstance(weekly, dict) or weekly.get("draws") != 100_000:
        raise ValueError("AFDR-864 weekly-cluster test contract drifted")
    if weekly.get("seed") != 20_260_720:
        raise ValueError("AFDR-864 weekly-cluster seed drifted")

    anchors = {
        ADDRESS_SOURCE: ADDRESS_SOURCE_SHA256,
        ADDRESS_MANIFEST: ADDRESS_MANIFEST_SHA256,
        ADDRESS_BUILDER: ADDRESS_BUILDER_SHA256,
        FUNDING_SOURCE: FUNDING_SOURCE_SHA256,
        FUNDING_MANIFEST: FUNDING_MANIFEST_SHA256,
        FUNDING_BUILDER: FUNDING_BUILDER_SHA256,
    }
    for path, expected_sha256 in anchors.items():
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"AFDR-864 frozen anchor mismatch: {path}")
    for comparator in COMPARATORS:
        if sha256_file(comparator["path"]) != comparator["sha256"]:
            raise ValueError(
                f"AFDR-864 comparator mismatch: {comparator['candidate']}"
            )


def preregistration_payload() -> dict[str, Any]:
    """Return the frozen contract without parsing a source/comparator row."""
    validate_contract()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": CANDIDATE,
        "as_of_date": AS_OF_DATE,
        "decision": "freeze_source_support_before_combining_source_values",
        "sources": {
            "address": {
                "path": str(ADDRESS_SOURCE),
                "sha256": ADDRESS_SOURCE_SHA256,
                "manifest": str(ADDRESS_MANIFEST),
                "manifest_sha256": ADDRESS_MANIFEST_SHA256,
                "builder": str(ADDRESS_BUILDER),
                "builder_sha256": ADDRESS_BUILDER_SHA256,
                "allowed_columns": list(ADDRESS_COLUMNS),
                "availability": "exact available_at; current row no later than D+3d",
            },
            "funding": {
                "path": str(FUNDING_SOURCE),
                "sha256": FUNDING_SOURCE_SHA256,
                "manifest": str(FUNDING_MANIFEST),
                "manifest_sha256": FUNDING_MANIFEST_SHA256,
                "builder": str(FUNDING_BUILDER),
                "builder_sha256": FUNDING_BUILDER_SHA256,
                "signal_allowed_columns": list(FUNDING_SIGNAL_COLUMNS),
                "exact_physical_columns": list(FUNDING_PHYSICAL_COLUMNS),
                "forbidden_during_support": [
                    "settlement_mark_price",
                    "mark_open_time_ms",
                    "mark_open_time_utc",
                    "funding_time_offset_ms",
                    "mark_source",
                ],
                "availability": "funding_time_utc + 5 minutes",
            },
            "numeric_rows_parsed_during_preregistration": 0,
        },
        "policy": {
            "config": asdict(FROZEN_CONFIG),
            "features": {
                "balance_growth_7d": "log(AdrBalCnt_t / AdrBalCnt_t-7d)",
                "activity_growth_7d": "log(AdrActCnt_t / AdrActCnt_t-7d)",
                "address_feature_availability": (
                    "max(current available_at, exact t-7d available_at); if "
                    "later than current available_at, reference-only after "
                    "that time and never signal-eligible"
                ),
                "funding_pressure_72h": (
                    "sum the nine most recent already-available funding_rate "
                    "values; canonical settlement slots must be consecutive "
                    "8h after floor(time_ms/8h), offset must be in [0,60000]ms, "
                    "UTC and millisecond timestamps must agree, and newest "
                    "causal-availability age must be <=8h"
                ),
                "rank": (
                    "tie-midrank against strictly prior finite observations "
                    "within 365 calendar days; minimum 180; current excluded"
                ),
                "network_rank": (
                    "mean(balance_growth_rank, activity_growth_rank)"
                ),
            },
            "direction": (
                "LONG if network_rank>=0.75 and funding_rank<=0.25; "
                "SHORT if network_rank<=0.25 and funding_rank>=0.75"
            ),
            "event": (
                "current LONG/SHORT only when the immediately preceding exact "
                "daily observation is valid FLAT; missing/invalid predecessor "
                "cannot be treated as FLAT"
            ),
            "entry": "ceil(decision_time,5m)+5m",
            "exit": "entry plus exactly 864 five-minute bars",
            "controls": {
                name: CONTROL_CONTRACTS[name] for name in SOURCE_ONLY_CONTROLS
            },
        },
        "support_gates": dict(SUPPORT_GATES),
        "support_contract": dict(SUPPORT_CONTRACT),
        "novelty_contract": dict(NOVELTY_CONTRACT),
        "economic_gates": dict(ECONOMIC_GATES),
        "strict_mdd_contract": dict(STRICT_MDD_CONTRACT),
        "comparators": [dict(item) for item in COMPARATORS],
        "outcome_boundary": {
            "outcomes_opened": False,
            "outcome_sources_opened": False,
            "address_numeric_rows_parsed": 0,
            "funding_numeric_rows_parsed": 0,
            "comparator_rows_parsed": 0,
            "btc_market_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_rows_read": 0,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
        "one_way_sequence": {
            "support_evaluator_frozen_before_source_combination": True,
            "stop_on_source_or_novelty_failure": True,
            "economic_evaluator_frozen_before_market_access": True,
            "train_before_selection_outcome_transport": True,
            "selection_before_post_2023": True,
            "failure_action": "retire AFDR-864 without repair",
            "llm_or_rl_rescue_forbidden": True,
        },
        "files": {
            "mechanism_document": str(MECHANISM_DOCUMENT),
            "mechanism_document_sha256": sha256_file(MECHANISM_DOCUMENT),
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": sha256_file(
                PREREGISTRATION_DOCUMENT
            ),
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": sha256_file(PREREGISTRATION_SOURCE),
        },
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def write_once_json(payload: dict[str, Any], output: str | Path) -> None:
    destination = _path(output)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    if destination.exists():
        if destination.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(
                f"refusing to overwrite frozen AFDR-864 artifact: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = preregistration_payload()
    write_once_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
