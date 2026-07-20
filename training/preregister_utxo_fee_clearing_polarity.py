"""Generate the outcome-blind UFCP-1 preregistration artifact.

This module intentionally does not parse the real UTXO CSV.  It validates only
source-manifest JSON identity, the bound source/manifest file hashes, frozen
schema metadata, and explicit outcome-boundary counters before writing a
canonical preregistration manifest for the UFCP-1 evaluator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any


POLICY_ID = "UFCP-1"
PROTOCOL_VERSION = "utxo_fee_clearing_polarity_preregistration_v1"
SOURCE_PROTOCOL_VERSION = "bitcoin_utxo_fee_block_stats_source_v1"
SOURCE_MANIFEST = Path("results/bitcoin_utxo_fee_block_stats_source_manifest_2026-07-20.json")
EXPECTED_SOURCE_MANIFEST_SHA256 = "ceb096b7bfeaf309729c4cab9f5cd4d2e0d526e261b1e460e9d03cd4f24e8084"
EXPECTED_SOURCE_MANIFEST_HASH = "98a84b0bd0338300f62eaa047b87498cc5a8d9505a03f6bd1912d1deb9564e8c"
EXPECTED_SOURCE_OUTPUT = Path("data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz")
EXPECTED_SOURCE_OUTPUT_SHA256 = "8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f"
EXPECTED_SOURCE_OUTPUT_BYTES = 13_991_597
SOURCE_DECISION = Path("docs/utxo-fee-clearing-polarity-mechanism-decision-2026-07-20.md")
SOURCE_DECISION_SHA256 = "95bf889fd053987e1717b182dc5da4f19ef51d75a1cbda427913089368c4852e"
SOURCE_BUILDER = Path("training/download_bitcoin_utxo_fee_stats.py")
EXPECTED_SOURCE_BUILDER_SHA256 = "099454feff009a5a4d44a96bd3790ff586d0365eba2e9b72e7b071d34e743633"
EXPECTED_REFERENCE = Path("data/bitcoin_block_summaries_2020_2023.csv.gz")
EXPECTED_REFERENCE_SHA256 = "1f8d1c0153717f2c18d8fa6f09428c780a850b2aecc8fb42bc497e16e68e1833"
PREREGISTRATION_SOURCE = Path("training/preregister_utxo_fee_clearing_polarity.py")
DEFAULT_OUTPUT = Path("results/utxo_fee_clearing_polarity_preregistration_2026-07-20.json")
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
    "topology_only": "remove fee_rank condition; retain polarity tails, side mapping, scheduling, and non-overlap",
    "low_fee_mirror": "replace fee_rank >= 0.75 with fee_rank <= 0.25; retain polarity tails and side mapping",
    "stale_7d": "at source day D use the already published fee_rank and polarity_rank attached to D-7; retain D publication/entry schedule",
    "year_side_stratified_random_clock": "sample without replacement from otherwise eligible publication days, preserving primary calendar-year and side counts; seed 20260720",
    "one_bar_delayed_entry": "same primary signals and sides; shift entry and scheduled exit exactly one 5m bar later",
}


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
        raise RuntimeError("UFCP source manifest must be a JSON object")
    return payload


def _manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_hash"}


def _validate_config(cfg: Config) -> None:
    source_manifest = Path(cfg.source_manifest)
    output = Path(cfg.preregistration_output)
    if source_manifest == output or source_manifest.resolve() == output.resolve():
        raise ValueError("UFCP artifact path must not alias the source manifest")
    protected = {
        SOURCE_DECISION.resolve(),
        SOURCE_BUILDER.resolve(),
        PREREGISTRATION_SOURCE.resolve(),
    }
    if output.resolve() in protected:
        raise ValueError("UFCP artifact path must not overwrite a source file")
    if output.suffix != ".json" or source_manifest.suffix != ".json":
        raise ValueError("UFCP preregistration and source manifest paths must be JSON files")


def _validate_source_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    if manifest_file.resolve() != SOURCE_MANIFEST.resolve():
        raise RuntimeError("UFCP source manifest path differs from the frozen source")
    manifest_file_sha256 = sha256_file(manifest_file)
    manifest = _read_json(manifest_file)
    if canonical_hash(_manifest_core(manifest)) != manifest.get("manifest_hash"):
        raise RuntimeError("UFCP source manifest hash mismatch")
    if manifest.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
        raise RuntimeError("UFCP source manifest protocol version mismatch")

    decision = manifest.get("source_decision")
    if decision != {"path": str(SOURCE_DECISION), "sha256": SOURCE_DECISION_SHA256}:
        raise RuntimeError("UFCP source decision binding drift")
    if sha256_file(SOURCE_DECISION) != SOURCE_DECISION_SHA256:
        raise RuntimeError("UFCP mechanism decision file drift")

    builder = manifest.get("source_builder")
    if not isinstance(builder, dict):
        raise RuntimeError("UFCP source builder binding missing")
    if builder.get("path") != str(SOURCE_BUILDER):
        raise RuntimeError("UFCP source builder path drift")
    builder_sha = builder.get("sha256")
    if builder_sha != EXPECTED_SOURCE_BUILDER_SHA256:
        raise RuntimeError("UFCP source builder SHA differs from the frozen source")
    if builder_sha != sha256_file(SOURCE_BUILDER):
        raise RuntimeError("UFCP source builder SHA drift")

    output = manifest.get("output")
    if not isinstance(output, dict):
        raise RuntimeError("UFCP source output metadata missing")
    source_path = output.get("path")
    if not isinstance(source_path, str) or not source_path:
        raise RuntimeError("UFCP source output path missing")
    if Path(source_path).resolve() == manifest_file.resolve():
        raise RuntimeError("UFCP source output path aliases manifest")
    if Path(source_path).resolve() != EXPECTED_SOURCE_OUTPUT.resolve():
        raise RuntimeError("UFCP source output path differs from the frozen source")
    if output.get("columns") != SOURCE_COLUMNS:
        raise RuntimeError("UFCP source schema drift")
    observed_source_sha = sha256_file(source_path)
    if output.get("sha256") != observed_source_sha:
        raise RuntimeError("UFCP source file SHA mismatch")
    if observed_source_sha != EXPECTED_SOURCE_OUTPUT_SHA256:
        raise RuntimeError("UFCP source frozen SHA drift")
    if output.get("bytes") != Path(source_path).stat().st_size:
        raise RuntimeError("UFCP source file byte-size mismatch")
    if output.get("bytes") != EXPECTED_SOURCE_OUTPUT_BYTES:
        raise RuntimeError("UFCP source frozen byte-size drift")

    manifest_sha = sha256_file(manifest_file)
    if manifest.get("outcome_boundary") != SOURCE_OUTCOME_BOUNDARY:
        raise RuntimeError("UFCP source outcome boundary drift")
    audit = manifest.get("source_audit")
    if not isinstance(audit, dict):
        raise RuntimeError("UFCP source audit missing")
    if audit.get("complete_inclusive_height_range") is not True:
        raise RuntimeError("UFCP source height range is incomplete")
    if audit.get("unique_block_hashes") is not True:
        raise RuntimeError("UFCP source block hashes are not unique")
    if audit.get("all_rows_pre_cutoff") is not True:
        raise RuntimeError("UFCP source crossed the frozen cutoff")
    if audit.get("utxo_identity_checked") is not True:
        raise RuntimeError("UFCP source UTXO identity was not checked")
    if audit.get("start_height") != FROZEN_START_HEIGHT:
        raise RuntimeError("UFCP source start height drift")
    if audit.get("end_height") != FROZEN_END_HEIGHT:
        raise RuntimeError("UFCP source end height drift")
    if audit.get("expected_rows") != FROZEN_ROWS or audit.get("observed_rows") != FROZEN_ROWS:
        raise RuntimeError("UFCP source exact row-count drift")
    if audit.get("end_timestamp_exclusive") != FROZEN_END_TIMESTAMP_EXCLUSIVE:
        raise RuntimeError("UFCP source timestamp boundary drift")
    if audit.get("height_links_checked") != FROZEN_ROWS - 1:
        raise RuntimeError("UFCP source hash-link audit count drift")
    if audit.get("latest_eligible_packet_end") != audit.get("end_height", 0) - 6:
        raise RuntimeError("UFCP source six-successor boundary drift")
    reference = manifest.get("reference_audit")
    if not isinstance(reference, dict):
        raise RuntimeError("UFCP frozen-reference audit missing")
    if reference.get("rows_cross_checked") != FROZEN_ROWS:
        raise RuntimeError("UFCP frozen-reference row-count drift")
    if reference.get("reference_path") != str(EXPECTED_REFERENCE):
        raise RuntimeError("UFCP frozen-reference path drift")
    if reference.get("reference_sha256") != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("UFCP frozen-reference SHA binding drift")
    if sha256_file(EXPECTED_REFERENCE) != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("UFCP frozen-reference file SHA drift")
    if reference.get("columns_cross_checked") != REFERENCE_COLUMNS:
        raise RuntimeError("UFCP frozen-reference column drift")
    if reference.get("all_basic_fields_match_reference") is not True:
        raise RuntimeError("UFCP frozen-reference field mismatch")
    if manifest.get("manifest_hash") != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("UFCP source frozen manifest-hash drift")
    if manifest_file_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("UFCP source frozen manifest-file SHA drift")

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
            "source_day": "UTC day D aggregated from completed confirmed Bitcoin blocks only",
            "edges": "sum(total_inputs + total_outputs)",
            "fee_burden": "log(sum(total_fees) / edges)",
            "utxo_polarity": "sum(utxo_set_change) / edges",
            "invalid_day": "reject a day when block count <72, edges<=0, total_fees<=0, or six successor blocks are unavailable",
            "daily_source_minimum_blocks": 72,
            "usable_range_requires_no_missing_utc_source_day": True,
        },
        "causal_availability": {
            "day_d_unavailable_before": "D+2 00:00 UTC",
            "hash_linked_successors_after_final_included_block": 6,
            "publication_latency": "one complete 5m latency bar",
            "entry_time": "D+2 00:05 UTC",
            "forbidden_inputs": [
                "firstSeen",
                "pool identity",
                "unconfirmed mempool state",
                "post-entry block",
            ],
        },
        "normalization": {
            "method": "strict-prior rolling empirical midrank",
            "midrank_formula": "(count(prior < current) + 0.5 * count(prior == current)) / prior_count",
            "lookback_source_days": 180,
            "minimum_prior_source_days": 120,
            "fee_rank": "midrank of fee_burden over strict-prior source days",
            "polarity_rank": "midrank of utxo_polarity over strict-prior source days",
        },
        "eligibility": {
            "long": "fee_rank >= 0.75 and polarity_rank >= 0.75",
            "short": "fee_rank >= 0.75 and polarity_rank <= 0.25",
            "every_eligible_day_considered": True,
            "ordering": "chronological non-overlap",
        },
        "execution": {
            "bar_size": "5m",
            "hold_bars": 288,
            "notional_leverage": 0.5,
            "base_cost_bp_per_notional_per_side": 6,
            "stress_cost_bp_per_notional_per_side": 10,
            "funding": "exact funding, entry-inclusive/exit-exclusive, fixed entry quantity",
        },
        "support_floors": {
            "train_2021_2022_total_minimum": 60,
            "train_each_year_minimum": 24,
            "selection_2023_total_minimum": 24,
            "selection_each_half_minimum": 10,
            "long_short_share_train_and_selection": "each side between 25% and 75% inclusive in each window",
            "maximum_month_share": "<=0.15 separately in train and selection",
            "daily_source_minimum_blocks": 72,
            "no_missing_utc_source_day_in_usable_range": True,
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
                "topology_only",
                "low_fee_mirror",
                "stale_7d",
                "year_side_stratified_random_clock",
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


def artifact_core(cfg: Config, source_binding: dict[str, Any]) -> dict[str, Any]:
    protocol = policy()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "config": asdict(cfg),
        "source_manifest": source_binding,
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


def write_preregistration(cfg: Config) -> dict[str, Any]:
    _validate_config(cfg)
    source_binding = _validate_source_manifest(cfg.source_manifest)
    output = Path(cfg.preregistration_output)
    if output.resolve() == Path(source_binding["source_output"]["path"]).resolve():
        raise ValueError("UFCP preregistration output must not overwrite source data")
    core = artifact_core(cfg, source_binding)
    artifact = {**core, "manifest_hash": canonical_hash(core)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return artifact


def load_preregistration(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    artifact = _read_json(path)
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    if canonical_hash(core) != artifact.get("manifest_hash"):
        raise RuntimeError("UFCP preregistration manifest hash mismatch")
    if artifact.get("policy") != policy():
        raise RuntimeError("UFCP preregistration policy drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("UFCP preregistration opened outcomes")
    if artifact.get("outcome_boundary") != PREREGISTRATION_OUTCOME_BOUNDARY:
        raise RuntimeError("UFCP preregistration source-only boundary drift")
    source = artifact.get("preregistration_source")
    expected_source = {
        "path": str(PREREGISTRATION_SOURCE),
        "sha256": sha256_file(PREREGISTRATION_SOURCE),
    }
    if source != expected_source:
        raise RuntimeError("UFCP preregistration source binding drift")
    raw_config = artifact.get("config")
    if not isinstance(raw_config, dict):
        raise RuntimeError("UFCP preregistration config missing")
    try:
        cfg = Config(**raw_config)
    except TypeError as exc:
        raise RuntimeError("UFCP preregistration config drift") from exc
    _validate_config(cfg)
    if artifact.get("source_manifest") != _validate_source_manifest(cfg.source_manifest):
        raise RuntimeError("UFCP preregistration source-manifest binding drift")
    return artifact


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=Config.source_manifest)
    parser.add_argument("--preregistration-output", default=Config.preregistration_output)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(write_preregistration(parse_args()), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
