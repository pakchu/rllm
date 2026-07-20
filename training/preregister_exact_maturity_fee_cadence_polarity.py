"""Generate the outcome-blind block-level EMFC-864 preregistration artifact.

This module intentionally does not parse the real UTXO CSV.  It validates only
source-manifest JSON identity, the bound source/manifest file hashes, frozen
schema metadata, and explicit outcome-boundary counters before writing a
canonical preregistration manifest for the EMFC-864 evaluator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


POLICY_ID = "EMFC-864"
PROTOCOL_VERSION = "exact_maturity_fee_cadence_polarity_preregistration_v1"
SOURCE_PROTOCOL_VERSION = "bitcoin_utxo_fee_block_stats_source_v1"
SOURCE_MANIFEST = Path("results/bitcoin_utxo_fee_block_stats_source_manifest_2026-07-20.json")
EXPECTED_SOURCE_MANIFEST_SHA256 = "ceb096b7bfeaf309729c4cab9f5cd4d2e0d526e261b1e460e9d03cd4f24e8084"
EXPECTED_SOURCE_MANIFEST_HASH = "98a84b0bd0338300f62eaa047b87498cc5a8d9505a03f6bd1912d1deb9564e8c"
EXPECTED_SOURCE_OUTPUT = Path("data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz")
EXPECTED_SOURCE_OUTPUT_SHA256 = "8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f"
EXPECTED_SOURCE_OUTPUT_BYTES = 13_991_597
SOURCE_DECISION = Path("docs/utxo-fee-clearing-polarity-mechanism-decision-2026-07-20.md")
SOURCE_DECISION_SHA256 = "95bf889fd053987e1717b182dc5da4f19ef51d75a1cbda427913089368c4852e"
MECHANISM_DECISION = Path(
    "docs/exact-maturity-fee-cadence-polarity-mechanism-decision-2026-07-20.md"
)
MECHANISM_DECISION_SHA256 = (
    "a640d13f02b23b0c76d5acb73427be0ad6fb87a3d08f9cd392a47b22f2918a39"
)
SOURCE_BUILDER = Path("training/download_bitcoin_utxo_fee_stats.py")
EXPECTED_SOURCE_BUILDER_SHA256 = "099454feff009a5a4d44a96bd3790ff586d0365eba2e9b72e7b071d34e743633"
EXPECTED_REFERENCE = Path("data/bitcoin_block_summaries_2020_2023.csv.gz")
EXPECTED_REFERENCE_SHA256 = "1f8d1c0153717f2c18d8fa6f09428c780a850b2aecc8fb42bc497e16e68e1833"
PREREGISTRATION_SOURCE = Path("training/preregister_exact_maturity_fee_cadence_polarity.py")
DEFAULT_OUTPUT = Path("results/exact_maturity_fee_cadence_polarity_preregistration_2026-07-20.json")
FROZEN_START_HEIGHT = 610_691
FROZEN_END_HEIGHT = 823_785
FROZEN_ROWS = FROZEN_END_HEIGHT - FROZEN_START_HEIGHT + 1
FROZEN_END_TIMESTAMP_EXCLUSIVE = 1_704_067_200
SOURCE_COLUMNS = [
    "height",
    "id",
    "previousblockhash",
    "timestamp",
    "mediantime",
    "tx_count",
    "size",
    "weight",
    "total_fees",
    "total_inputs",
    "total_outputs",
    "utxo_set_change",
]
REFERENCE_COLUMNS = SOURCE_COLUMNS[:8]
NOVELTY_COMPARATORS = {
    "BATE-288": {
        "path": "results/block_arrival_throughput_elasticity_clock_2026-07-20.csv",
        "sha256": "cd4fbd01c104bd969ca1c12a53b8da82dd0e9376990e233c286ff009a5115c02",
        "entry_column": "entry_time",
        "exit_column": "exit_time",
        "side_column": "side",
    },
    "UFCP-1": {
        "path": "results/utxo_fee_clearing_polarity_primary_clock_2026-07-20.csv",
        "sha256": "8338c290d63b522531c8d55c8a79ba73cc13915c936733ec03ffcf6ab0e86c1b",
        "entry_column": "entry_time",
        "exit_column": "exit_time",
        "side_column": "side",
    },
    "MCR-7": {
        "path": "results/miner_cadence_recovery_clock_2026-07-17.csv",
        "sha256": "2535244889b046ff00c369ee854973a91c23429dff82a6dd3c1a293a01352b0b",
        "entry_column": "entry_date",
        "exit_column": "exit_date",
        "side_column": "side",
    },
    "NTB-7": {
        "path": "results/network_topology_broadening_clock_2026-07-17.csv",
        "sha256": "6b1bd7c7458cffa062e40872c3ad1730007c01426790b1ba8e52c6eb853de42f",
        "entry_column": "entry_date",
        "exit_column": "exit_date",
        "side_column": "side",
    },
    "BFC-3": {
        "path": "results/blockspace_fee_confirmation_clock_2026-07-17.csv",
        "sha256": "edda7bb8ae8a1de4e51a3b86e98d533748e73d203125a3ded1a487e9a0e93632",
        "entry_column": "entry_date",
        "exit_column": "exit_date",
        "side_column": "side",
    },
}
SOURCE_OUTCOME_BOUNDARY = {
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "outcome_rows_loaded": 0,
    "return_or_pnl_fields": 0,
    "post_2023_source_rows_loaded": 0,
    "raw_mempool_responses_persisted": False,
    "unrelated_mempool_metadata_persisted": False,
}
PREREGISTRATION_OUTCOME_BOUNDARY = {
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "return_rows_loaded": 0,
    "market_values_read": 0,
    "funding_values_read": 0,
    "return_or_pnl_fields": 0,
    "source_csv_values_read": 0,
    "source_manifest_only": True,
}
CONTROL_DEFINITIONS = {
    "direction_flip": "same primary entry/exit clock with every side multiplied by -1; diagnostic only",
    "constant_long_same_clock": "same primary entry/exit clock with side fixed to long",
    "constant_short_same_clock": "same primary entry/exit clock with side fixed to short",
    "fee_only": "fee-rank tail onset with the primary side orientation, availability, hold, and non-overlap",
    "cadence_only": "cadence-rank tail onset with the primary side orientation, availability, hold, and non-overlap",
    "same_height_fee": "replace total_fees[h-100] by total_fees[h]; retain exact primary cadence and scheduling",
    "pseudo_maturity_99": "replace the exact origin by h-99 and elapsed span by 99 blocks",
    "pseudo_maturity_101": "replace the exact origin by h-101 and elapsed span by 101 blocks",
    "daily_aggregate_shadow": "aggregate the exact block primitives by completed UTC maturation day and enter only at D+2 00:05 UTC",
    "stale_7d": "at height h use the latest already observed feature pair at or before timestamp[h]-604800 seconds",
    "origin_day_shift_leakage_sentinel": "assigning the signal to h-100 or its UTC day is prohibited and must be reported as a leakage sentinel only",
    "year_month_side_activity_stratified_random_clock": "sample non-overlapping source times preserving year, month, side, event count, and source-activity stratum; seed 20260720",
    "one_bar_delayed_entry": "same primary signals and sides; shift entry and scheduled exit exactly one 5m bar later",
}


@dataclass(frozen=True)
class SourceFreeze:
    source_manifest: Path
    source_manifest_sha256: str
    source_manifest_hash: str
    source_output: Path
    source_output_sha256: str
    source_output_bytes: int
    source_builder_sha256: str
    reference: Path
    reference_sha256: str


PRODUCTION_SOURCE_FREEZE = SourceFreeze(
    source_manifest=SOURCE_MANIFEST,
    source_manifest_sha256=EXPECTED_SOURCE_MANIFEST_SHA256,
    source_manifest_hash=EXPECTED_SOURCE_MANIFEST_HASH,
    source_output=EXPECTED_SOURCE_OUTPUT,
    source_output_sha256=EXPECTED_SOURCE_OUTPUT_SHA256,
    source_output_bytes=EXPECTED_SOURCE_OUTPUT_BYTES,
    source_builder_sha256=EXPECTED_SOURCE_BUILDER_SHA256,
    reference=EXPECTED_REFERENCE,
    reference_sha256=EXPECTED_REFERENCE_SHA256,
)


@dataclass(frozen=True)
class Config:
    source_manifest: str = str(SOURCE_MANIFEST)
    preregistration_output: str = str(DEFAULT_OUTPUT)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("EMFC source manifest must be a JSON object")
    return payload


def _manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_hash"}


def _validate_config(cfg: Config, freeze: SourceFreeze) -> None:
    source_manifest = Path(cfg.source_manifest)
    output = Path(cfg.preregistration_output)
    if source_manifest.resolve() != freeze.source_manifest.resolve():
        raise RuntimeError("EMFC source manifest path differs from the frozen source")
    if source_manifest == output or source_manifest.resolve() == output.resolve():
        raise ValueError("EMFC artifact path must not alias the source manifest")
    protected = {
        SOURCE_DECISION.resolve(),
        MECHANISM_DECISION.resolve(),
        SOURCE_BUILDER.resolve(),
        PREREGISTRATION_SOURCE.resolve(),
        *(Path(item["path"]).resolve() for item in NOVELTY_COMPARATORS.values()),
    }
    if output.resolve() in protected:
        raise ValueError("EMFC artifact path must not overwrite a source file")
    if output.suffix != ".json" or source_manifest.suffix != ".json":
        raise ValueError("EMFC preregistration and source manifest paths must be JSON files")


def _validate_mechanism_decision() -> dict[str, str]:
    if not MECHANISM_DECISION.is_file():
        raise RuntimeError("EMFC mechanism decision is missing")
    observed = sha256_file(MECHANISM_DECISION)
    if observed != MECHANISM_DECISION_SHA256:
        raise RuntimeError("EMFC mechanism decision drift")
    return {"path": str(MECHANISM_DECISION), "sha256": observed}


def _validate_novelty_comparators() -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for policy_id, definition in NOVELTY_COMPARATORS.items():
        path = Path(definition["path"])
        if not path.is_file():
            raise RuntimeError(f"EMFC novelty comparator is missing: {policy_id}")
        observed = sha256_file(path)
        if observed != definition["sha256"]:
            raise RuntimeError(f"EMFC novelty comparator drift: {policy_id}")
        bindings[policy_id] = dict(definition)
    return bindings


def _validate_source_manifest(
    manifest_path: str | Path,
    freeze: SourceFreeze,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    if manifest_file.resolve() != freeze.source_manifest.resolve():
        raise RuntimeError("EMFC source manifest path differs from the frozen source")
    manifest_file_sha256 = sha256_file(manifest_file)
    manifest = _read_json(manifest_file)
    if canonical_hash(_manifest_core(manifest)) != manifest.get("manifest_hash"):
        raise RuntimeError("EMFC source manifest hash mismatch")
    if manifest.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
        raise RuntimeError("EMFC source manifest protocol version mismatch")

    decision = manifest.get("source_decision")
    if decision != {"path": str(SOURCE_DECISION), "sha256": SOURCE_DECISION_SHA256}:
        raise RuntimeError("EMFC source decision binding drift")
    if sha256_file(SOURCE_DECISION) != SOURCE_DECISION_SHA256:
        raise RuntimeError("EMFC source decision file drift")

    builder = manifest.get("source_builder")
    if not isinstance(builder, dict):
        raise RuntimeError("EMFC source builder binding missing")
    if builder.get("path") != str(SOURCE_BUILDER):
        raise RuntimeError("EMFC source builder path drift")
    builder_sha = builder.get("sha256")
    if builder_sha != freeze.source_builder_sha256:
        raise RuntimeError("EMFC source builder SHA differs from the frozen source")
    if builder_sha != sha256_file(SOURCE_BUILDER):
        raise RuntimeError("EMFC source builder SHA drift")

    output = manifest.get("output")
    if not isinstance(output, dict):
        raise RuntimeError("EMFC source output metadata missing")
    source_path = output.get("path")
    if not isinstance(source_path, str) or not source_path:
        raise RuntimeError("EMFC source output path missing")
    if Path(source_path).resolve() == manifest_file.resolve():
        raise RuntimeError("EMFC source output path aliases manifest")
    if Path(source_path).resolve() != freeze.source_output.resolve():
        raise RuntimeError("EMFC source output path differs from the frozen source")
    if output.get("columns") != SOURCE_COLUMNS:
        raise RuntimeError("EMFC source schema drift")
    observed_source_sha = sha256_file(source_path)
    if output.get("sha256") != observed_source_sha:
        raise RuntimeError("EMFC source file SHA mismatch")
    if observed_source_sha != freeze.source_output_sha256:
        raise RuntimeError("EMFC source frozen SHA drift")
    if output.get("bytes") != Path(source_path).stat().st_size:
        raise RuntimeError("EMFC source file byte-size mismatch")
    if output.get("bytes") != freeze.source_output_bytes:
        raise RuntimeError("EMFC source frozen byte-size drift")

    manifest_sha = sha256_file(manifest_file)
    if manifest.get("outcome_boundary") != SOURCE_OUTCOME_BOUNDARY:
        raise RuntimeError("EMFC source outcome boundary drift")
    audit = manifest.get("source_audit")
    if not isinstance(audit, dict):
        raise RuntimeError("EMFC source audit missing")
    if audit.get("complete_inclusive_height_range") is not True:
        raise RuntimeError("EMFC source height range is incomplete")
    if audit.get("unique_block_hashes") is not True:
        raise RuntimeError("EMFC source block hashes are not unique")
    if audit.get("all_rows_pre_cutoff") is not True:
        raise RuntimeError("EMFC source crossed the frozen cutoff")
    if audit.get("utxo_identity_checked") is not True:
        raise RuntimeError("EMFC source UTXO identity was not checked")
    if audit.get("start_height") != FROZEN_START_HEIGHT:
        raise RuntimeError("EMFC source start height drift")
    if audit.get("end_height") != FROZEN_END_HEIGHT:
        raise RuntimeError("EMFC source end height drift")
    if audit.get("expected_rows") != FROZEN_ROWS or audit.get("observed_rows") != FROZEN_ROWS:
        raise RuntimeError("EMFC source exact row-count drift")
    if audit.get("end_timestamp_exclusive") != FROZEN_END_TIMESTAMP_EXCLUSIVE:
        raise RuntimeError("EMFC source timestamp boundary drift")
    if audit.get("height_links_checked") != FROZEN_ROWS - 1:
        raise RuntimeError("EMFC source hash-link audit count drift")
    if audit.get("latest_eligible_packet_end") != audit.get("end_height", 0) - 6:
        raise RuntimeError("EMFC source six-successor boundary drift")
    reference = manifest.get("reference_audit")
    if not isinstance(reference, dict):
        raise RuntimeError("EMFC frozen-reference audit missing")
    if reference.get("rows_cross_checked") != FROZEN_ROWS:
        raise RuntimeError("EMFC frozen-reference row-count drift")
    if reference.get("reference_path") != str(freeze.reference):
        raise RuntimeError("EMFC frozen-reference path drift")
    if reference.get("reference_sha256") != freeze.reference_sha256:
        raise RuntimeError("EMFC frozen-reference SHA binding drift")
    if sha256_file(freeze.reference) != freeze.reference_sha256:
        raise RuntimeError("EMFC frozen-reference file SHA drift")
    if reference.get("columns_cross_checked") != REFERENCE_COLUMNS:
        raise RuntimeError("EMFC frozen-reference column drift")
    if reference.get("all_basic_fields_match_reference") is not True:
        raise RuntimeError("EMFC frozen-reference field mismatch")
    if manifest.get("manifest_hash") != freeze.source_manifest_hash:
        raise RuntimeError("EMFC source frozen manifest-hash drift")
    if manifest_file_sha256 != freeze.source_manifest_sha256:
        raise RuntimeError("EMFC source frozen manifest-file SHA drift")

    return {
        "path": str(manifest_file),
        "sha256": manifest_sha,
        "manifest_hash": manifest["manifest_hash"],
        "protocol_version": manifest["protocol_version"],
        "source_output": {
            "path": source_path,
            "sha256": output["sha256"],
            "bytes": output["bytes"],
            "columns": output["columns"],
        },
        "source_decision": decision,
        "source_builder": builder,
        "reference_audit": reference,
        "source_audit": {
            "observed_rows": audit.get("observed_rows"),
            "start_height": audit.get("start_height"),
            "end_height": audit.get("end_height"),
            "latest_eligible_packet_end": audit.get("latest_eligible_packet_end"),
            "height_links_checked": audit.get("height_links_checked"),
            "complete_inclusive_height_range": audit.get("complete_inclusive_height_range"),
            "unique_block_hashes": audit.get("unique_block_hashes"),
            "all_rows_pre_cutoff": audit.get("all_rows_pre_cutoff"),
            "utxo_identity_checked": audit.get("utxo_identity_checked"),
            "end_timestamp_exclusive": audit.get("end_timestamp_exclusive"),
        },
    }


def policy() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "singleton": True,
        "source_features": {
            "origin_height": "h-100",
            "maturity_height": "h",
            "confirmation_height": "h+6",
            "matured_fee_component": "total_fees[h-100]",
            "fee_pressure": "log1p(total_fees[h-100])",
            "maturity_elapsed_seconds": "mediantime[h]-mediantime[h-100]",
            "cadence_compression": "-log(maturity_elapsed_seconds/60000)",
            "valid_height": "all required heights exist, total_fees[h-100]>=0, and maturity_elapsed_seconds>0",
            "expected_candidate_heights": 212_989,
        },
        "causal_availability": {
            "confirmation_blocks": 6,
            "historical_embargo_seconds": 7_200,
            "raw_available": "max(timestamp[h:h+6])+7200 seconds",
            "decision_boundary": "ceil raw_available to the next 5m UTC boundary",
            "publication_latency": "one complete 5m latency bar",
            "entry_time": "decision_boundary+5m",
            "forbidden_inputs": [
                "firstSeen",
                "pool identity",
                "unconfirmed mempool state",
                "origin-height or origin-day assignment",
                "current or future row in normalization",
                "post-entry block",
            ],
        },
        "normalization": {
            "method": "strict-prior rolling empirical midrank",
            "midrank_formula": "(count(prior < current) + 0.5 * count(prior == current)) / prior_count",
            "reference_valid_heights": 26_208,
            "nominal_reference_days_at_144_blocks": 182,
            "require_full_reference": True,
            "invalid_heights_excluded_without_reset": True,
            "fee_rank": "midrank of fee_pressure over the last 26208 valid maturity heights strictly below h",
            "cadence_rank": "midrank of cadence_compression over the last 26208 valid maturity heights strictly below h",
        },
        "eligibility": {
            "high_pressure_compressed": "fee_rank>=0.90 and cadence_rank>=0.90; side=-1 short",
            "low_pressure_expanded": "fee_rank<=0.10 and cadence_rank<=0.10; side=+1 long",
            "neutral": "all other valid heights",
            "onset": "current valid state is extreme and differs from the immediately preceding valid state; invalid heights retain but cannot change prior state",
            "ordering": "increasing maturity height, then chronological non-overlap",
            "suppression": "accept the earliest onset with entry_time>=prior accepted exit_time; never replace a suppressed onset",
        },
        "execution": {
            "bar_size": "5m",
            "hold_bars": 864,
            "notional_leverage": 0.5,
            "base_cost_bp_per_notional_per_side": 6,
            "stress_cost_bp_per_notional_per_side": 10,
            "funding": "exact funding, entry-inclusive/exit-exclusive, fixed entry quantity",
        },
        "source_integrity_gates": {
            "exact_candidate_heights": 212_989,
            "minimum_positive_elapsed_ratio": 0.9995,
            "maximum_invalid_elapsed_run": 12,
            "confirmation_containment": True,
            "contiguous_hash_linked_pre_2024_source": True,
            "nonnegative_total_fees": True,
            "market_funding_return_or_post_2023_rows_loaded": 0,
        },
        "event_support_gates": {
            "train_2021_2022_total_minimum": 60,
            "train_2021_2022_total_maximum": 200,
            "train_each_year_minimum": 24,
            "selection_2023_total_minimum": 24,
            "selection_2023_total_maximum": 105,
            "selection_each_half_minimum": 10,
            "selection_each_quarter_minimum": 3,
            "long_short_share_train_and_selection": "each side between 25% and 75% inclusive in each window",
            "each_side_each_train_year_minimum": 7,
            "each_side_each_selection_half_minimum": 3,
            "maximum_month_share": "<=0.20 separately in train and selection",
            "exact_72h_boundary_gap_share_maximum": 0.50,
            "median_entry_gap_hours_minimum": 84,
        },
        "source_novelty_gates": {
            "feature_spearman_absolute_maximum": 0.90,
            "feature_pairs": [
                "fee_pressure versus same-height log1p(total_fees[h])",
                "primary extreme state versus pseudo-maturity-99 extreme state",
                "primary extreme state versus pseudo-maturity-101 extreme state",
            ],
            "shadow_exposure_absolute_correlation_maximum": 0.80,
            "shadow_controls": [
                "fee_only",
                "cadence_only",
                "same_height_fee",
                "pseudo_maturity_99",
                "pseudo_maturity_101",
                "daily_aggregate_shadow",
                "stale_7d",
            ],
            "existing_network_alpha_exposure_absolute_correlation_maximum": 0.35,
            "existing_network_alpha_comparators": list(NOVELTY_COMPARATORS),
            "exposure_grid": "5m UTC over 2021-01-01 inclusive through 2024-01-01 exclusive; flat=0",
            "failure_action": "reject before loading market or funding outcomes; no threshold, lag, side, onset, or hold repair",
        },
        "control_construction": {
            "block_controls": "reuse the primary 26208-valid-height midrank, +/-0.90 tails, six-confirmation availability, onset, and 864-bar non-overlap unless the named transform changes a field",
            "daily_aggregate_shadow": {
                "source_day": "UTC day of timestamp[h] for valid exact-maturity height h",
                "fee_pressure": "log1p(sum(total_fees[h-100])) over the completed source day",
                "cadence_compression": "median of block-level cadence_compression over the completed source day",
                "normalization": "strict-prior empirical midrank over exactly 180 valid source days",
                "states": "same joint 0.90/0.10 tails and side orientation as primary",
                "availability": "D+2 00:00 UTC plus one complete 5m latency bar",
                "onset_and_nonoverlap": "same valid-state onset and 864-bar chronological non-overlap as primary",
            },
            "stale_7d": "for each h, use ranks from the latest valid maturity height with timestamp<=timestamp[h]-604800; retain h availability",
            "random_clock": {
                "candidate_lattice": "valid normalized maturity heights not used by the primary clock",
                "strata": "entry calendar year, entry calendar month, primary side, and cadence-rank quartile",
                "matching": "preserve exact primary count in every stratum and enforce 864-bar non-overlap globally",
                "seed": 20260720,
            },
            "exposure_correlation": "Pearson correlation of signed 5m position exposure; zero when either vector has zero variance",
        },
        "performance_gates": {
            "required_windows": ["train_2021_2022", "selection_2023"],
            "absolute_return_positive_each": True,
            "cagr_to_strict_mdd_minimum_each": 3.0,
            "strict_max_drawdown_maximum_each": 0.15,
            "weekly_cluster_sign_flip_p_maximum_each": 0.10,
            "mean_gross_bp_minimum_each": 30.0,
            "stress_absolute_return_positive_each": True,
            "positive_subperiods": ["2021", "2022", "2023H1", "2023H2"],
            "long_only_and_short_only_absolute_return_positive_each_window": True,
            "one_bar_delayed_entry_absolute_return_positive_each_window": True,
            "component_control_full_gate_pass_rejects_mechanism": [
                "constant_long_same_clock",
                "constant_short_same_clock",
                "fee_only",
                "cadence_only",
                "same_height_fee",
                "pseudo_maturity_99",
                "pseudo_maturity_101",
                "daily_aggregate_shadow",
                "stale_7d",
                "year_month_side_activity_stratified_random_clock",
            ],
            "direction_flip_is_diagnostic_only": True,
            "post_outcome_threshold_side_hold_or_latency_repair": "forbidden",
        },
        "strict_accounting": {
            "cagr_clock": "full declared wall-clock window including idle cash",
            "drawdown_high_water_mark": "global and pre-entry",
            "held_path_order": "all favorable OHLC/funding-credit extremes before all adverse OHLC/funding-debit extremes",
            "costs": "entry, scheduled exit, and hypothetical adverse liquidation costs included",
            "cluster_test": "one-sided weekly entry-cluster sign flip; 100000 draws; seed 20260720",
        },
        "controls": CONTROL_DEFINITIONS,
        "selection_stages": {
            "stage_1": "open train 2021-2022 first",
            "stage_2": "open 2023 only after exact train pass",
            "stage_3": "2024+ remains sealed",
        },
    }


def artifact_core(
    cfg: Config,
    source_binding: dict[str, Any],
    mechanism_decision: dict[str, str],
    novelty_comparators: dict[str, dict[str, str]],
) -> dict[str, Any]:
    protocol = policy()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "config": asdict(cfg),
        "source_manifest": source_binding,
        "mechanism_decision": mechanism_decision,
        "novelty_comparators": novelty_comparators,
        "policy": protocol,
        "policy_hash": canonical_hash(protocol),
        "outcomes_opened": False,
        "outcome_boundary": PREREGISTRATION_OUTCOME_BOUNDARY,
        "research_sequence": {
            "train_first": "2021-2022",
            "selection_second": "2023 only after exact train pass",
            "sealed": "2024+",
        },
        "preregistration_source": {
            "path": str(PREREGISTRATION_SOURCE),
            "sha256": sha256_file(PREREGISTRATION_SOURCE),
        },
    }


def _write_preregistration(cfg: Config, freeze: SourceFreeze) -> dict[str, Any]:
    _validate_config(cfg, freeze)
    source_binding = _validate_source_manifest(cfg.source_manifest, freeze)
    mechanism_decision = _validate_mechanism_decision()
    novelty_comparators = _validate_novelty_comparators()
    output = Path(cfg.preregistration_output)
    if output.resolve() == Path(source_binding["source_output"]["path"]).resolve():
        raise ValueError("EMFC preregistration output must not overwrite source data")
    core = artifact_core(
        cfg,
        source_binding,
        mechanism_decision,
        novelty_comparators,
    )
    artifact = {**core, "manifest_hash": canonical_hash(core)}
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
    return artifact


def write_preregistration(cfg: Config) -> dict[str, Any]:
    return _write_preregistration(cfg, PRODUCTION_SOURCE_FREEZE)


def _load_preregistration(
    path: str | Path,
    freeze: SourceFreeze,
) -> dict[str, Any]:
    artifact = _read_json(path)
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    if canonical_hash(core) != artifact.get("manifest_hash"):
        raise RuntimeError("EMFC preregistration manifest hash mismatch")
    if artifact.get("policy") != policy():
        raise RuntimeError("EMFC preregistration policy drift")
    if artifact.get("policy_hash") != canonical_hash(policy()):
        raise RuntimeError("EMFC preregistration policy hash drift")
    if artifact.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("EMFC preregistration protocol version drift")
    if artifact.get("policy_id") != POLICY_ID:
        raise RuntimeError("EMFC preregistration policy id drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("EMFC preregistration opened outcomes")
    if artifact.get("outcome_boundary") != PREREGISTRATION_OUTCOME_BOUNDARY:
        raise RuntimeError("EMFC preregistration source-only boundary drift")
    if artifact.get("mechanism_decision") != _validate_mechanism_decision():
        raise RuntimeError("EMFC preregistration mechanism binding drift")
    if artifact.get("novelty_comparators") != _validate_novelty_comparators():
        raise RuntimeError("EMFC preregistration novelty-comparator binding drift")
    source = artifact.get("preregistration_source")
    expected_source = {
        "path": str(PREREGISTRATION_SOURCE),
        "sha256": sha256_file(PREREGISTRATION_SOURCE),
    }
    if source != expected_source:
        raise RuntimeError("EMFC preregistration source binding drift")
    raw_config = artifact.get("config")
    if not isinstance(raw_config, dict):
        raise RuntimeError("EMFC preregistration config missing")
    try:
        cfg = Config(**raw_config)
    except TypeError as exc:
        raise RuntimeError("EMFC preregistration config drift") from exc
    if Path(cfg.preregistration_output).resolve() != Path(path).resolve():
        raise RuntimeError("EMFC preregistration output-path binding drift")
    _validate_config(cfg, freeze)
    if artifact.get("source_manifest") != _validate_source_manifest(
        cfg.source_manifest,
        freeze,
    ):
        raise RuntimeError("EMFC preregistration source-manifest binding drift")
    return artifact


def load_preregistration(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    return _load_preregistration(path, PRODUCTION_SOURCE_FREEZE)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=Config.source_manifest)
    parser.add_argument("--preregistration-output", default=Config.preregistration_output)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(write_preregistration(parse_args()), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
