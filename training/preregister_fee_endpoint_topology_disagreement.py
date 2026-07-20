"""Write the outcome-blind FETD-288 singleton preregistration.

Only frozen manifest metadata and exact artifact byte hashes are inspected.
The confirmed-ledger CSV is never decompressed or parsed, and no BTC market or
funding outcome is opened.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


POLICY_ID = "FETD-288"
PROTOCOL_VERSION = "fee_endpoint_topology_disagreement_preregistration_v1"
SOURCE_PROTOCOL_VERSION = "bitcoin_utxo_fee_block_stats_source_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

SOURCE_MANIFEST = Path(
    "results/bitcoin_utxo_fee_block_stats_source_manifest_2026-07-20.json"
)
EXPECTED_SOURCE_MANIFEST_FILE_SHA256 = (
    "ceb096b7bfeaf309729c4cab9f5cd4d2e0d526e261b1e460e9d03cd4f24e8084"
)
EXPECTED_SOURCE_MANIFEST_HASH = (
    "98a84b0bd0338300f62eaa047b87498cc5a8d9505a03f6bd1912d1deb9564e8c"
)
EXPECTED_SOURCE_OUTPUT = Path(
    "data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz"
)
EXPECTED_SOURCE_OUTPUT_SHA256 = (
    "8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f"
)
EXPECTED_SOURCE_OUTPUT_BYTES = 13_991_597
SOURCE_ORIGIN_DECISION = Path(
    "docs/utxo-fee-clearing-polarity-mechanism-decision-2026-07-20.md"
)
SOURCE_ORIGIN_DECISION_SHA256 = (
    "95bf889fd053987e1717b182dc5da4f19ef51d75a1cbda427913089368c4852e"
)
MECHANISM_DECISION = Path(
    "docs/fee-endpoint-topology-disagreement-mechanism-decision-2026-07-20.md"
)
MECHANISM_DECISION_SHA256 = (
    "3864ae9ec8bd93ad766e5c4c811c175dd34bac1609209d91d5ef13114ac75340"
)
SOURCE_BUILDER = Path("training/download_bitcoin_utxo_fee_stats.py")
SOURCE_BUILDER_SHA256 = (
    "099454feff009a5a4d44a96bd3790ff586d0365eba2e9b72e7b071d34e743633"
)
REFERENCE_SOURCE = Path("data/bitcoin_block_summaries_2020_2023.csv.gz")
REFERENCE_SOURCE_SHA256 = (
    "1f8d1c0153717f2c18d8fa6f09428c780a850b2aecc8fb42bc497e16e68e1833"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_fee_endpoint_topology_disagreement.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/fee-endpoint-topology-disagreement-fetd288-preregistration-2026-07-20.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "83af41a9f8d6a512f02e27cb7bcd326645af44ea09953ce0bdddffddfb5cb11c"
)
DEFAULT_OUTPUT = Path(
    "results/fee_endpoint_topology_disagreement_preregistration_2026-07-20.json"
)

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
EXPECTED_DATA_USE = (
    "private internal research cache of consensus-derived fields via public "
    "Mempool REST transport; the API output has no separately documented data "
    "licence, production must use an owned Bitcoin Core node, and no raw "
    "response or unrelated metadata is redistributed"
)
PREREGISTRATION_OUTCOME_BOUNDARY = {
    "source_manifest_json_read": True,
    "source_artifact_bytes_hashed": True,
    "source_csv_values_read": 0,
    "fetd_feature_rows_derived": 0,
    "signal_incidence_rows_derived": 0,
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "return_or_pnl_fields": 0,
}
CONTROL_DEFINITIONS = {
    "direction_flip": (
        "same primary entry/exit clock and eligibility with every side "
        "multiplied by -1; diagnostic only"
    ),
    "fee_only": (
        "require nonzero fee_transport and fee_magnitude_rank>=0.75; "
        "side=-sign(fee_transport); build an independent chronological "
        "non-overlap clock"
    ),
    "endpoint_only": (
        "require nonzero endpoint_transport and "
        "endpoint_magnitude_rank>=0.75; side=sign(endpoint_transport); build "
        "an independent chronological non-overlap clock"
    ),
    "same_direction": (
        "require fee_transport*endpoint_transport>0 and strain_rank>=0.75; "
        "side=sign(endpoint_transport); build an independent chronological "
        "non-overlap clock"
    ),
    "constant_long_same_clock": (
        "same primary entry/exit clock with side fixed long"
    ),
    "constant_short_same_clock": (
        "same primary entry/exit clock with side fixed short"
    ),
    "stale_14_packets": (
        "at source packet t apply primary eligibility and side from the fully "
        "formed feature/ranks at t-14; retain t availability and build an "
        "independent chronological non-overlap clock"
    ),
    "month_side_stratified_random_clock": (
        "within each split-month SHA256-order rank-ready split-contained "
        "candidates by 20260720|window|month|entry_time; take the primary "
        "month total and preserve the primary long/short counts"
    ),
    "one_bar_delayed_entry": (
        "same primary signals and sides; shift entry and scheduled exit "
        "exactly one complete 5m bar later; deterministically drop a shifted "
        "trade that loses its original split containment, never replace it, "
        "and report train/selection dropped counts before outcomes"
    ),
}


@dataclass(frozen=True)
class Config:
    source_manifest: str = str(SOURCE_MANIFEST)
    preregistration_output: str = str(DEFAULT_OUTPUT)


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
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


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"FETD JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(
        _repository_path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("FETD JSON must be an object")
    return payload


def _manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_hash"}


def _validate_config(cfg: Config, *, require_new_output: bool) -> None:
    manifest = _repository_path(cfg.source_manifest)
    output = _repository_path(cfg.preregistration_output)
    if manifest != _repository_path(SOURCE_MANIFEST):
        raise RuntimeError("FETD source manifest path differs from frozen source")
    if manifest.suffix != ".json" or output.suffix != ".json":
        raise ValueError("FETD source manifest and preregistration must be JSON")
    protected = {
        manifest,
        _repository_path(EXPECTED_SOURCE_OUTPUT),
        _repository_path(REFERENCE_SOURCE),
        _repository_path(SOURCE_ORIGIN_DECISION),
        _repository_path(MECHANISM_DECISION),
        _repository_path(SOURCE_BUILDER),
        _repository_path(PREREGISTRATION_SOURCE),
        _repository_path(PREREGISTRATION_DOCUMENT),
    }
    if output in protected:
        raise ValueError("FETD preregistration output aliases a protected source")
    if require_new_output and output.exists():
        raise FileExistsError("FETD preregistration is immutable")


def _validate_source_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = _repository_path(path)
    if sha256_file(manifest_path) != EXPECTED_SOURCE_MANIFEST_FILE_SHA256:
        raise RuntimeError("FETD source manifest file SHA drift")
    manifest = _read_json(manifest_path)
    if canonical_hash(_manifest_core(manifest)) != manifest.get("manifest_hash"):
        raise RuntimeError("FETD source manifest canonical hash mismatch")
    if manifest.get("manifest_hash") != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("FETD source frozen manifest hash drift")
    if manifest.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
        raise RuntimeError("FETD source protocol version drift")

    expected_origin = {
        "path": str(SOURCE_ORIGIN_DECISION),
        "sha256": SOURCE_ORIGIN_DECISION_SHA256,
    }
    if manifest.get("source_decision") != expected_origin:
        raise RuntimeError("FETD source-origin decision binding drift")
    if sha256_file(SOURCE_ORIGIN_DECISION) != SOURCE_ORIGIN_DECISION_SHA256:
        raise RuntimeError("FETD source-origin decision file drift")

    expected_builder = {
        "path": str(SOURCE_BUILDER),
        "sha256": SOURCE_BUILDER_SHA256,
    }
    if manifest.get("source_builder") != expected_builder:
        raise RuntimeError("FETD source builder binding drift")
    if sha256_file(SOURCE_BUILDER) != SOURCE_BUILDER_SHA256:
        raise RuntimeError("FETD source builder file drift")

    output = manifest.get("output")
    if not isinstance(output, dict):
        raise RuntimeError("FETD source output metadata missing")
    if output.get("path") != str(EXPECTED_SOURCE_OUTPUT):
        raise RuntimeError("FETD source output path drift")
    if output.get("columns") != SOURCE_COLUMNS:
        raise RuntimeError("FETD source schema drift")
    if output.get("bytes") != EXPECTED_SOURCE_OUTPUT_BYTES:
        raise RuntimeError("FETD source byte-size drift")
    if output.get("sha256") != EXPECTED_SOURCE_OUTPUT_SHA256:
        raise RuntimeError("FETD source output SHA binding drift")
    source_path = _repository_path(EXPECTED_SOURCE_OUTPUT)
    if source_path.stat().st_size != EXPECTED_SOURCE_OUTPUT_BYTES:
        raise RuntimeError("FETD source file byte-size mismatch")
    if sha256_file(source_path) != EXPECTED_SOURCE_OUTPUT_SHA256:
        raise RuntimeError("FETD source file SHA mismatch")

    audit = manifest.get("source_audit")
    if not isinstance(audit, dict):
        raise RuntimeError("FETD source audit missing")
    expected_audit = {
        "expected_rows": FROZEN_ROWS,
        "observed_rows": FROZEN_ROWS,
        "start_height": FROZEN_START_HEIGHT,
        "end_height": FROZEN_END_HEIGHT,
        "latest_eligible_packet_end": FROZEN_END_HEIGHT - 6,
        "height_links_checked": FROZEN_ROWS - 1,
        "end_timestamp_exclusive": FROZEN_END_TIMESTAMP_EXCLUSIVE,
        "complete_inclusive_height_range": True,
        "unique_block_hashes": True,
        "all_rows_pre_cutoff": True,
        "utxo_identity_checked": True,
    }
    for key, expected in expected_audit.items():
        if audit.get(key) != expected:
            raise RuntimeError(f"FETD source audit {key} drift")

    reference = manifest.get("reference_audit")
    if not isinstance(reference, dict):
        raise RuntimeError("FETD frozen-reference audit missing")
    if reference.get("reference_path") != str(REFERENCE_SOURCE):
        raise RuntimeError("FETD frozen-reference path drift")
    if reference.get("reference_sha256") != REFERENCE_SOURCE_SHA256:
        raise RuntimeError("FETD frozen-reference SHA binding drift")
    if sha256_file(REFERENCE_SOURCE) != REFERENCE_SOURCE_SHA256:
        raise RuntimeError("FETD frozen-reference file SHA drift")
    if reference.get("rows_cross_checked") != FROZEN_ROWS:
        raise RuntimeError("FETD frozen-reference row count drift")
    if reference.get("columns_cross_checked") != REFERENCE_COLUMNS:
        raise RuntimeError("FETD frozen-reference column drift")
    if reference.get("all_basic_fields_match_reference") is not True:
        raise RuntimeError("FETD frozen-reference field mismatch")

    if manifest.get("outcome_boundary") != SOURCE_OUTCOME_BOUNDARY:
        raise RuntimeError("FETD source outcome boundary drift")
    if manifest.get("data_use") != EXPECTED_DATA_USE:
        raise RuntimeError("FETD source data-use boundary drift")

    return {
        "path": str(SOURCE_MANIFEST),
        "sha256": EXPECTED_SOURCE_MANIFEST_FILE_SHA256,
        "manifest_hash": EXPECTED_SOURCE_MANIFEST_HASH,
        "protocol_version": SOURCE_PROTOCOL_VERSION,
        "source_output": {
            "path": str(EXPECTED_SOURCE_OUTPUT),
            "sha256": EXPECTED_SOURCE_OUTPUT_SHA256,
            "bytes": EXPECTED_SOURCE_OUTPUT_BYTES,
            "columns": list(SOURCE_COLUMNS),
        },
        "source_origin_decision": expected_origin,
        "source_builder": expected_builder,
        "source_audit": {key: audit[key] for key in expected_audit},
        "reference_audit": {
            "reference_path": reference["reference_path"],
            "reference_sha256": reference["reference_sha256"],
            "rows_cross_checked": reference["rows_cross_checked"],
            "columns_cross_checked": reference["columns_cross_checked"],
            "all_basic_fields_match_reference": reference[
                "all_basic_fields_match_reference"
            ],
        },
        "outcome_boundary": dict(SOURCE_OUTCOME_BOUNDARY),
        "data_use": EXPECTED_DATA_USE,
    }


def policy() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "singleton": True,
        "source_features": {
            "packet_blocks": 72,
            "packet_alignment": "packet_id=floor(height/72)",
            "first_complete_packet_start_height": 610_704,
            "last_complete_packet_end_height": 823_751,
            "complete_packet_count": 2_959,
            "edge_packet_policy": "drop incomplete first and last packets",
            "total_weight": "sum(weight)",
            "total_fees": "sum(total_fees)",
            "total_endpoints": "sum(total_inputs+total_outputs)",
            "fee_pressure": "log(total_fees/total_weight)",
            "endpoint_density": "log(total_endpoints/total_weight)",
            "transport_horizon_packets": 2,
            "transport_horizon_expected_hours": 24,
            "fee_transport": "fee_pressure[t]-fee_pressure[t-2]",
            "endpoint_transport": "endpoint_density[t]-endpoint_density[t-2]",
            "strain_magnitude": "abs(fee_transport*endpoint_transport)",
            "base_valid_feature_row": (
                "require t-2,t-1,t as consecutive valid 72-block packets; each "
                "packet has exactly 72 contiguous heights and positive finite "
                "total_weight, "
                "total_fees, and total_endpoints; all derived values finite"
            ),
            "forbidden_primary_fields": [
                "utxo_set_change",
                "tx_count",
                "size",
                "mediantime",
                "market",
                "funding",
                "premium",
                "open_interest",
                "liquidation",
                "order_book",
                "return",
                "pnl",
            ],
            "numeric_contract": (
                "IEEE-754 binary64 and natural log; no epsilon, clipping, "
                "rounding, interpolation, forward fill, reassignment, or "
                "imputation; exact binary64 equality defines zero and ties"
            ),
        },
        "normalization": {
            "method": "strict-prior rolling empirical midrank",
            "midrank_formula": (
                "(count(prior < current)+0.5*count(prior == current))/prior_count"
            ),
            "lookback_valid_feature_packets": 180,
            "minimum_prior_valid_feature_packets": 120,
            "window_selection": (
                "use exactly the most recent 180 strict-prior base-valid rows "
                "when available; fewer than 120 makes current rank-unready"
            ),
            "current_row_excluded": True,
            "prior_available_at_strictly_before_current": True,
            "tie_equality": "exact IEEE-754 binary64 equality",
            "strain_rank": "midrank of strain_magnitude",
            "fee_magnitude_rank_control": "midrank of abs(fee_transport)",
            "endpoint_magnitude_rank_control": (
                "midrank of abs(endpoint_transport)"
            ),
        },
        "eligibility": {
            "common": (
                "fee_transport*endpoint_transport<0 and strain_rank>=0.75"
            ),
            "long": "common and fee_transport<0 and endpoint_transport>0",
            "short": "common and fee_transport>0 and endpoint_transport<0",
            "side": "-sign(fee_transport), equal to sign(endpoint_transport)",
            "zero_tolerance": "none; exact zero in either transport is ineligible",
            "every_eligible_source_clock_considered": True,
            "ordering": (
                "sort by entry_time then packet_id; accept earliest when "
                "entry_time>=prior accepted exit; equal boundary allowed; "
                "suppress intervening candidates without score priority or "
                "replacement"
            ),
            "post_support_repair": "forbidden",
        },
        "causal_availability": {
            "hash_linked_successors_after_packet_end": 6,
            "source_available_at": (
                "max header timestamp from packet start through h+6 + 48 hours"
            ),
            "availability_lag_seconds": 172_800,
            "ceil_5m": "((unix_seconds+299)//300)*300",
            "entry_time": "ceil_5m(source_available_at)+300 seconds",
            "entry_latency_seconds": 300,
            "already_aligned_rule": (
                "an aligned available_at still receives the additional 300s bar"
            ),
            "historical_header_time_is_not_receipt_time": True,
            "height_packet_prevents_calendar_backfill": True,
            "source_gap_action": "reject source; never fill or backdate",
        },
        "execution": {
            "bar_size": "5m",
            "entry": "next open exactly at entry_time",
            "hold_bars": 288,
            "hold_hours": 24,
            "scheduled_exit_time": "entry_time+86400 seconds",
            "interval": "[entry_time,scheduled_exit_time)",
            "split_containment": (
                "entry_time>=split_start and scheduled_exit_time<=split_end"
            ),
            "non_overlap": True,
            "notional_leverage": 0.5,
            "base_cost_bp_per_notional_per_side": 6,
            "stress_cost_bp_per_notional_per_side": 10,
            "funding": (
                "exact funding, entry-inclusive/exit-exclusive, fixed entry quantity"
            ),
        },
        "calendar": {
            "warmup_source": "[2020-01-01T00:00:00Z,2021-01-01T00:00:00Z)",
            "train": "[2021-01-01T00:00:00Z,2023-01-01T00:00:00Z)",
            "selection": "[2023-01-01T00:00:00Z,2024-01-01T00:00:00Z)",
            "sealed": "2024+",
            "window_assignment": (
                "UTC half-open windows with entry and scheduled exit contained"
            ),
            "fit_permission": "no fitted coefficient; all thresholds frozen",
        },
        "support_gates": {
            "count_basis": (
                "accepted primary entries after eligibility, split containment, "
                "and chronological non-overlap"
            ),
            "train_total_minimum": 80,
            "train_each_year_minimum": 32,
            "train_long_minimum": 25,
            "train_short_minimum": 25,
            "train_each_side_each_year_minimum": 10,
            "train_each_half_year_minimum": 14,
            "train_maximum_month_share": 0.15,
            "selection_total_minimum": 35,
            "selection_long_minimum": 12,
            "selection_short_minimum": 12,
            "selection_each_half_minimum": 14,
            "selection_each_side_each_half_minimum": 5,
            "selection_each_quarter_minimum": 6,
            "selection_maximum_month_share": 0.20,
            "complete_packet_count": 2_959,
            "blocks_per_complete_packet": 72,
            "consecutive_complete_packet_ids": True,
            "delayed_entry_split_edge_reporting": (
                "report train and selection dropped counts before outcomes; "
                "dropped trades receive no replacement"
            ),
            "support_failure_action": (
                "reject FETD-288 before market/funding/outcomes; no repair"
            ),
        },
        "performance_gates": {
            "required_sequence": ["train", "selection"],
            "absolute_return_positive_each": True,
            "cagr_to_strict_mdd_minimum_each": 3.0,
            "strict_max_drawdown_maximum_each": 0.15,
            "weekly_cluster_sign_flip_p_maximum_each": 0.10,
            "mean_gross_bp_minimum_each": 30.0,
            "stress_absolute_return_positive_each": True,
            "positive_subperiods": ["2021", "2022", "2023H1", "2023H2"],
            "long_and_short_absolute_return_positive_each_window": True,
            "one_bar_delayed_entry_absolute_return_positive_each_window": True,
            "component_control_full_gate_pass_rejects_mechanism": [
                "fee_only",
                "endpoint_only",
                "same_direction",
                "constant_long_same_clock",
                "constant_short_same_clock",
                "stale_14_packets",
                "month_side_stratified_random_clock",
            ],
            "direction_flip_is_diagnostic_only": True,
            "control_replacement": "forbidden under FETD-288",
            "post_outcome_repair": "forbidden",
        },
        "strict_accounting": {
            "required_report_fields": [
                "absolute_return",
                "cagr",
                "strict_mdd",
                "cagr_to_strict_mdd",
                "trades",
                "long_trades",
                "short_trades",
                "calendar_clusters",
            ],
            "cagr_clock": "full declared wall-clock window including idle cash",
            "drawdown_high_water_mark": "global and pre-entry",
            "held_path_order": (
                "all favorable OHLC/funding-credit extremes before all adverse "
                "OHLC/funding-debit extremes"
            ),
            "costs": (
                "entry, scheduled exit, and hypothetical adverse liquidation "
                "costs included"
            ),
            "cluster_test": (
                "one-sided weekly entry-cluster sign flip; 100000 draws; "
                "seed 20260720"
            ),
        },
        "controls": dict(CONTROL_DEFINITIONS),
        "promotion_boundary": (
            "pre-2024 pass required before a separately frozen 2024-2026 "
            "source extension; no live promotion before field parity and 90 "
            "forward shadow days"
        ),
        "stopping_rule": (
            "stop permanently at first support/train/selection failure; no "
            "sign, threshold, rank-window, packet, support-floor, hold, latency, "
            "calendar, or clock repair; no failed-policy inversion"
        ),
    }


def _artifact_core(cfg: Config, source_binding: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(MECHANISM_DECISION) != MECHANISM_DECISION_SHA256:
        raise RuntimeError("FETD mechanism decision file drift")
    if sha256_file(PREREGISTRATION_DOCUMENT) != PREREGISTRATION_DOCUMENT_SHA256:
        raise RuntimeError("FETD preregistration document file drift")
    frozen_policy = policy()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "config": asdict(cfg),
        "source_manifest": source_binding,
        "mechanism_decision": {
            "path": str(MECHANISM_DECISION),
            "sha256": MECHANISM_DECISION_SHA256,
        },
        "preregistration_document": {
            "path": str(PREREGISTRATION_DOCUMENT),
            "sha256": PREREGISTRATION_DOCUMENT_SHA256,
        },
        "policy": frozen_policy,
        "policy_hash": canonical_hash(frozen_policy),
        "outcomes_opened": False,
        "outcome_boundary": dict(PREREGISTRATION_OUTCOME_BOUNDARY),
        "research_sequence": {
            "support_first": "source-only train/selection incidence",
            "evaluator_second": "commit and hash-freeze strict evaluator",
            "outcomes": "train first; selection only after exact train pass",
            "sealed": "2024+ source and outcomes",
        },
        "preregistration_source": {
            "path": str(PREREGISTRATION_SOURCE),
            "sha256": sha256_file(PREREGISTRATION_SOURCE),
        },
    }


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    handle.close()
    return Path(handle.name)


def write_preregistration(cfg: Config) -> dict[str, Any]:
    _validate_config(cfg, require_new_output=True)
    source_binding = _validate_source_manifest(cfg.source_manifest)
    core = _artifact_core(cfg, source_binding)
    artifact = {**core, "manifest_hash": canonical_hash(core)}
    output = _repository_path(cfg.preregistration_output)
    temporary = _temporary_path(output)
    try:
        temporary.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        os.link(temporary, output)
        return artifact
    finally:
        temporary.unlink(missing_ok=True)


def load_preregistration(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    artifact = _read_json(path)
    core = _manifest_core(artifact)
    if canonical_hash(core) != artifact.get("manifest_hash"):
        raise RuntimeError("FETD preregistration canonical hash mismatch")
    if artifact.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("FETD preregistration protocol drift")
    frozen_policy = policy()
    if artifact.get("policy") != frozen_policy:
        raise RuntimeError("FETD preregistration policy drift")
    if artifact.get("policy_hash") != canonical_hash(frozen_policy):
        raise RuntimeError("FETD preregistration policy hash drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("FETD preregistration opened outcomes")
    if artifact.get("outcome_boundary") != PREREGISTRATION_OUTCOME_BOUNDARY:
        raise RuntimeError("FETD preregistration outcome boundary drift")

    expected_source = {
        "path": str(PREREGISTRATION_SOURCE),
        "sha256": sha256_file(PREREGISTRATION_SOURCE),
    }
    if artifact.get("preregistration_source") != expected_source:
        raise RuntimeError("FETD preregistration source binding drift")
    expected_decision = {
        "path": str(MECHANISM_DECISION),
        "sha256": MECHANISM_DECISION_SHA256,
    }
    if artifact.get("mechanism_decision") != expected_decision:
        raise RuntimeError("FETD mechanism-decision binding drift")
    if sha256_file(MECHANISM_DECISION) != MECHANISM_DECISION_SHA256:
        raise RuntimeError("FETD mechanism decision file drift")
    expected_document = {
        "path": str(PREREGISTRATION_DOCUMENT),
        "sha256": PREREGISTRATION_DOCUMENT_SHA256,
    }
    if artifact.get("preregistration_document") != expected_document:
        raise RuntimeError("FETD preregistration document binding drift")
    if sha256_file(PREREGISTRATION_DOCUMENT) != PREREGISTRATION_DOCUMENT_SHA256:
        raise RuntimeError("FETD preregistration document file drift")

    raw_config = artifact.get("config")
    if not isinstance(raw_config, dict):
        raise RuntimeError("FETD preregistration config missing")
    try:
        cfg = Config(**raw_config)
    except TypeError as exc:
        raise RuntimeError("FETD preregistration config drift") from exc
    _validate_config(cfg, require_new_output=False)
    if _repository_path(path) != _repository_path(cfg.preregistration_output):
        raise RuntimeError("FETD preregistration output-path binding drift")
    if artifact.get("source_manifest") != _validate_source_manifest(
        cfg.source_manifest
    ):
        raise RuntimeError("FETD preregistration source-manifest binding drift")
    return artifact


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=Config.source_manifest)
    parser.add_argument(
        "--preregistration-output", default=Config.preregistration_output
    )
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(
        json.dumps(
            write_preregistration(parse_args()),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
