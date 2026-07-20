"""Write the outcome-blind WCTR-288 singleton preregistration.

Only the frozen source manifest and artifact byte hashes are inspected here.
The raw/normalized size-weight values and all BTC outcomes remain unopened.
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


POLICY_ID = "WCTR-288"
PROTOCOL_VERSION = "witness_composition_transport_preregistration_v1"
SOURCE_PROTOCOL_VERSION = "mempool_witness_composition_history_source_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = Path(
    "results/mempool_witness_composition_source_manifest_2026-07-20.json"
)
EXPECTED_SOURCE_MANIFEST_FILE_SHA256 = (
    "2506429ebcbf9b2ada6c745bcc58bd9ec3b0fdbe245f726a408ff99bd3111342"
)
EXPECTED_SOURCE_MANIFEST_HASH = (
    "55914b3ec31fe8fb66d8a8dc31acb3784a10b256625073a5aeff1d317660ea8d"
)
SOURCE_DECISION = Path(
    "docs/witness-composition-transport-mechanism-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "101e84303efb3146dae587048c179b6688c93b50bb4c99edec8ba8daab72bc98"
)
SOURCE_FREEZE = Path(
    "docs/witness-composition-transport-source-freeze-2026-07-20.md"
)
SOURCE_FREEZE_SHA256 = (
    "6697872b690dc38c8bfe4700d75fb3e7fce8fbab507794ff4200aa0b1ff410aa"
)
SOURCE_BUILDER = Path("training/download_mempool_witness_composition_history.py")
SOURCE_BUILDER_SHA256 = (
    "d5cd3f2cab5e501d5484539f1ea3c5aac5a96916dd65aa1060bb561fa639d721"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_witness_composition_transport.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/witness-composition-transport-preregistration-2026-07-20.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "23b94a5c7dcf8af16e31ff6c1e62483b74a0ffeb7457eb6884f2412bf9cb4a96"
)
DEFAULT_OUTPUT = Path(
    "results/witness_composition_transport_preregistration_2026-07-20.json"
)

EXPECTED_SOURCE_CONFIG = {
    "raw_output": "data/mempool_witness_composition_4y_2026-07-20.raw.json.gz",
    "output_csv": "data/mempool_witness_composition_4y_2026-07-20.csv.gz",
    "manifest_output": str(SOURCE_MANIFEST),
    "timeout_sec": 30.0,
    "request_pause_sec": 0.25,
    "maximum_retries": 8,
    "maximum_response_bytes": 3_000_000,
    "minimum_response_rows": 2_800,
}
EXPECTED_SOURCE_DECISION = {
    "path": str(SOURCE_DECISION),
    "sha256": SOURCE_DECISION_SHA256,
}
EXPECTED_SOURCE_BUILDER = {
    "path": str(SOURCE_BUILDER),
    "sha256": SOURCE_BUILDER_SHA256,
}
EXPECTED_SOURCE_AUDIT = {
    "endpoint": "https://mempool.space/api/v1/mining/blocks/sizes-weights/4y",
    "official_rest_docs": "https://mempool.space/docs/api/rest",
    "upstream_repository": "https://github.com/mempool/mempool",
    "upstream_commit": "e9d6cf8c042f946be53e372bb36530cd7b7851a4",
    "http_status": 200,
    "final_url": "https://mempool.space/api/v1/mining/blocks/sizes-weights/4y",
    "retrieved_at_utc": "2026-07-20T10:27:31.250986Z",
    "response_headers": {
        "date": "Mon, 20 Jul 2026 10:27:29 GMT",
        "etag": 'W/"59e71-wimSunVghY3LTE+daWrtvIOPIaY"',
        "last-modified": None,
        "content-type": "application/json; charset=utf-8",
    },
    "raw_response_bytes": 368_241,
    "raw_response_sha256": (
        "6fdb0db77ae56c5b348918bc966384c1a88032655b9e061c510bfcc3df642e94"
    ),
    "response_rows": 2_923,
    "retained_rows": 2_921,
    "edge_rows_dropped": 2,
    "first_dropped_bucket_start_utc": "2022-07-20T00:00:00Z",
    "last_dropped_bucket_start_utc": "2026-07-20T00:00:00Z",
    "first_retained_bucket_start_utc": "2022-07-20T12:00:00Z",
    "last_retained_bucket_start_utc": "2026-07-19T12:00:00Z",
    "first_retained_avg_height": 745_785,
    "last_retained_avg_height": 958_761,
    "strictly_increasing_unique_timestamps": True,
    "strictly_increasing_unique_heights": True,
    "unique_bucket_ids": True,
    "paired_size_weight_rows": 2_923,
    "size_weight_pairing_valid": True,
    "block_weight_limit_valid": True,
    "witness_share_bounds_valid": True,
    "rounding_tolerance_rows": 0,
    "rounding_tolerance_bytes": 4,
    "missing_12h_buckets": 0,
    "maximum_bucket_gap_seconds": 43_200,
    "missing_bucket_policy": "reject any gap; never forward-fill or backdate",
}
EXPECTED_RAW_ARTIFACT = {
    "path": EXPECTED_SOURCE_CONFIG["raw_output"],
    "sha256": "ddd3615294d501ed3b24c5d43e2fc16319bd87f1add3c14dde2362c4b789c4c1",
    "bytes": 73_145,
    "decompressed_sha256": EXPECTED_SOURCE_AUDIT["raw_response_sha256"],
    "encoding": "gzip of exact HTTP response bytes; mtime=0",
}
SOURCE_COLUMNS = [
    "bucket_start_utc",
    "bucket_end_utc",
    "available_at_utc",
    "avg_height",
    "avg_timestamp",
    "avg_size",
    "avg_weight",
]
EXPECTED_NORMALIZED_ARTIFACT = {
    "path": EXPECTED_SOURCE_CONFIG["output_csv"],
    "sha256": "ee761e813085dfdee675ca9d420516f814c4c2824f3f5cef604acc3871d46c61",
    "bytes": 69_318,
    "rows": 2_921,
    "columns": SOURCE_COLUMNS,
}
EXPECTED_SOURCE_SEMANTICS = {
    "avg_size": (
        "integer-cast average serialized block size among non-stale blocks in "
        "the 12-hour source bucket"
    ),
    "avg_weight": (
        "integer-cast average BIP 141 block weight among the same non-stale "
        "blocks and source bucket"
    ),
    "pairing": (
        "sizes and weights require exact avgHeight and timestamp identity "
        "before normalization"
    ),
    "derived_features": "none in the frozen source artifact",
}
EXPECTED_CAUSAL_AVAILABILITY = {
    "source_bucket_seconds": 43_200,
    "source_availability_lag_seconds": 172_800,
    "available_at_rule": "fixed bucket end + 48 hours",
    "earliest_entry_rule": (
        "first complete 5-minute execution bar after available_at"
    ),
    "earliest_entry_additional_seconds": 300,
    "edge_policy": "drop first and last response buckets",
    "missing_bucket_policy": "reject any gap; never forward-fill or backdate",
}
SOURCE_OUTCOME_BOUNDARY = {
    "btc_market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "premium_or_oi_rows_loaded": 0,
    "return_or_pnl_fields": 0,
    "wctr_features_derived": 0,
    "signal_incidence_rows_derived": 0,
}
EXPECTED_DATA_USE = (
    "private research snapshot; public-route operational and legal review "
    "remains required before live promotion"
)
PREREGISTRATION_OUTCOME_BOUNDARY = {
    "source_manifest_json_read": True,
    "raw_source_artifact_bytes_hashed": True,
    "normalized_source_artifact_bytes_hashed": True,
    "raw_source_values_read": 0,
    "normalized_source_values_read": 0,
    "source_feature_rows_derived": 0,
    "signal_incidence_rows_derived": 0,
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "return_or_pnl_fields": 0,
}
CONTROL_DEFINITIONS = {
    "direction_flip": (
        "same primary clock and eligibility with every side multiplied by -1; "
        "diagnostic only"
    ),
    "transport_only": (
        "require transport_7d!=0, retain magnitude_rank>=0.75 and "
        "side=sign(transport_7d); remove impulse confirmation and fullness "
        "floor; build its own chronological non-overlap clock"
    ),
    "impulse_only": (
        "require impulse_24h!=0, retain impulse_magnitude_rank>=0.75 and "
        "fullness_rank>=0.50 with side=sign(impulse_24h); remove seven-day "
        "transport; build its own chronological non-overlap clock"
    ),
    "low_fullness_complement": (
        "retain transport magnitude_rank>=0.75 and transport/impulse sign "
        "agreement but require fullness_rank<0.50; build its own clock"
    ),
    "serialized_size_only": (
        "require nonzero log_size_7d/log_size_24h with equal signs; use "
        "strict-prior rank of abs(log_size_7d)>=0.75, fullness_rank>=0.50, "
        "and side=sign(log_size_7d); build its own clock"
    ),
    "block_weight_only": (
        "require nonzero log_weight_7d/log_weight_24h with equal signs; use "
        "strict-prior rank of abs(log_weight_7d)>=0.75, "
        "fullness_rank>=0.50, and side=sign(log_weight_7d); build its own clock"
    ),
    "constant_long_same_clock": "same primary entry/exit clock with side fixed long",
    "constant_short_same_clock": "same primary entry/exit clock with side fixed short",
    "stale_7d": (
        "at source bucket t apply primary eligibility and side to the fully "
        "formed primary feature/ranks from t-14; retain t availability and "
        "execution clock"
    ),
    "month_side_stratified_random_clock": (
        "build a feature-agnostic rank-ready split-contained chronological "
        "non-overlap pool; within each split-month SHA256-order candidates by "
        "20260720|window|month|entry_time, take the primary month total, assign "
        "the first primary-long count long and the remainder short"
    ),
    "one_bar_delayed_entry": (
        "same primary signals and sides; shift entry and scheduled exit exactly "
        "one complete 5m bar later; drop any shifted interval that is no longer "
        "split-contained"
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
            raise RuntimeError(f"WCTR JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(
        _repository_path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("WCTR manifest must be a JSON object")
    return payload


def _manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_hash"}


def _validate_config(cfg: Config, *, require_new_output: bool) -> None:
    manifest = _repository_path(cfg.source_manifest)
    output = _repository_path(cfg.preregistration_output)
    if manifest.suffix != ".json" or output.suffix != ".json":
        raise ValueError("WCTR source manifest and preregistration must be JSON")
    protected = {
        _repository_path(SOURCE_DECISION),
        _repository_path(SOURCE_FREEZE),
        _repository_path(SOURCE_BUILDER),
        _repository_path(PREREGISTRATION_SOURCE),
        _repository_path(PREREGISTRATION_DOCUMENT),
        _repository_path(EXPECTED_RAW_ARTIFACT["path"]),
        _repository_path(EXPECTED_NORMALIZED_ARTIFACT["path"]),
        manifest,
    }
    if output in protected:
        raise ValueError("WCTR preregistration output aliases a protected source")
    if require_new_output and output.exists():
        raise FileExistsError("WCTR preregistration is immutable")


def _require_exact(manifest: dict[str, Any], key: str, expected: Any) -> None:
    if manifest.get(key) != expected:
        raise RuntimeError(f"WCTR frozen source {key} drift")


def _validate_source_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = _repository_path(path)
    if manifest_path != _repository_path(SOURCE_MANIFEST):
        raise RuntimeError("WCTR source manifest path differs from frozen source")
    if sha256_file(manifest_path) != EXPECTED_SOURCE_MANIFEST_FILE_SHA256:
        raise RuntimeError("WCTR source manifest file SHA drift")
    manifest = _read_json(manifest_path)
    core = _manifest_core(manifest)
    if canonical_hash(core) != manifest.get("manifest_hash"):
        raise RuntimeError("WCTR source manifest canonical hash mismatch")
    if manifest.get("manifest_hash") != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("WCTR source frozen manifest hash drift")
    if manifest.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
        raise RuntimeError("WCTR source protocol version drift")
    _require_exact(manifest, "config", EXPECTED_SOURCE_CONFIG)
    _require_exact(manifest, "source_decision", EXPECTED_SOURCE_DECISION)
    _require_exact(manifest, "source_builder", EXPECTED_SOURCE_BUILDER)
    _require_exact(manifest, "source_audit", EXPECTED_SOURCE_AUDIT)
    _require_exact(manifest, "raw_artifact", EXPECTED_RAW_ARTIFACT)
    _require_exact(
        manifest, "normalized_artifact", EXPECTED_NORMALIZED_ARTIFACT
    )
    _require_exact(manifest, "source_semantics", EXPECTED_SOURCE_SEMANTICS)
    _require_exact(
        manifest, "causal_availability", EXPECTED_CAUSAL_AVAILABILITY
    )
    _require_exact(manifest, "outcome_boundary", SOURCE_OUTCOME_BOUNDARY)
    _require_exact(manifest, "data_use", EXPECTED_DATA_USE)

    if sha256_file(SOURCE_DECISION) != SOURCE_DECISION_SHA256:
        raise RuntimeError("WCTR mechanism decision file drift")
    if sha256_file(SOURCE_FREEZE) != SOURCE_FREEZE_SHA256:
        raise RuntimeError("WCTR source freeze file drift")
    if sha256_file(SOURCE_BUILDER) != SOURCE_BUILDER_SHA256:
        raise RuntimeError("WCTR source builder file drift")
    for label, expected in (
        ("raw", EXPECTED_RAW_ARTIFACT),
        ("normalized", EXPECTED_NORMALIZED_ARTIFACT),
    ):
        artifact_path = _repository_path(expected["path"])
        if artifact_path == manifest_path:
            raise RuntimeError(f"WCTR {label} artifact aliases source manifest")
        if artifact_path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"WCTR {label} artifact byte-size drift")
        if sha256_file(artifact_path) != expected["sha256"]:
            raise RuntimeError(f"WCTR {label} artifact SHA drift")

    return {
        "path": str(SOURCE_MANIFEST),
        "sha256": EXPECTED_SOURCE_MANIFEST_FILE_SHA256,
        "manifest_hash": EXPECTED_SOURCE_MANIFEST_HASH,
        "protocol_version": SOURCE_PROTOCOL_VERSION,
        "source_decision": EXPECTED_SOURCE_DECISION,
        "source_freeze": {
            "path": str(SOURCE_FREEZE),
            "sha256": SOURCE_FREEZE_SHA256,
        },
        "source_builder": EXPECTED_SOURCE_BUILDER,
        "source_audit": EXPECTED_SOURCE_AUDIT,
        "raw_artifact": EXPECTED_RAW_ARTIFACT,
        "normalized_artifact": EXPECTED_NORMALIZED_ARTIFACT,
        "source_semantics": EXPECTED_SOURCE_SEMANTICS,
        "causal_availability": EXPECTED_CAUSAL_AVAILABILITY,
        "outcome_boundary": SOURCE_OUTCOME_BOUNDARY,
    }


def policy() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "singleton": True,
        "source_features": {
            "witness_share": (
                "(4.0*avg_size-avg_weight)/(3.0*avg_size)"
            ),
            "fullness": "avg_weight/4000000.0",
            "transport_horizon_buckets": 14,
            "transport_horizon_days": 7,
            "transport_7d": "witness_share[t]-witness_share[t-14]",
            "impulse_horizon_buckets": 2,
            "impulse_horizon_hours": 24,
            "impulse_24h": "witness_share[t]-witness_share[t-2]",
            "component_controls": {
                "log_size_7d": "log(avg_size[t])-log(avg_size[t-14])",
                "log_size_24h": "log(avg_size[t])-log(avg_size[t-2])",
                "log_weight_7d": "log(avg_weight[t])-log(avg_weight[t-14])",
                "log_weight_24h": "log(avg_weight[t])-log(avg_weight[t-2])",
                "primary_excludes_component_fields": True,
            },
            "numeric_contract": (
                "IEEE-754 binary64; math.log for component controls; no epsilon, "
                "clipping, rounding, interpolation, fill, or imputation; exact "
                "binary64 equality defines zero and rank ties"
            ),
            "base_valid_feature_row": (
                "require t through t-14 present as 15 consecutive exact 12h "
                "buckets; avg_size and avg_weight finite and >0; every derived "
                "witness_share and current fullness finite and within [0,1]"
            ),
            "source_gap_action": "reject entire support build before features",
        },
        "normalization": {
            "method": "strict-prior rolling empirical midrank",
            "midrank_formula": (
                "(count(prior < current) + 0.5*count(prior == current)) / "
                "prior_count"
            ),
            "lookback_valid_feature_buckets": 180,
            "minimum_prior_valid_feature_buckets": 120,
            "window_selection": (
                "use exactly the most recent 180 strict-prior base-valid rows "
                "when available, otherwise all available; fewer than 120 makes "
                "the current row rank-unready"
            ),
            "current_row_excluded": True,
            "prior_rows_must_have_available_at_strictly_before_current": True,
            "tie_equality": "exact IEEE-754 binary64 equality",
            "magnitude_rank": "midrank of abs(transport_7d)",
            "fullness_rank": "midrank of fullness",
            "impulse_magnitude_rank_control": "midrank of abs(impulse_24h)",
            "log_size_magnitude_rank_control": "midrank of abs(log_size_7d)",
            "log_weight_magnitude_rank_control": "midrank of abs(log_weight_7d)",
        },
        "eligibility": {
            "common": (
                "magnitude_rank>=0.75 and fullness_rank>=0.50 and "
                "sign(transport_7d)==sign(impulse_24h)"
            ),
            "long": "common and transport_7d>0 and impulse_24h>0",
            "short": "common and transport_7d<0 and impulse_24h<0",
            "side": "sign(transport_7d)",
            "zero_tolerance": "none; exact binary64 zero is invalid",
            "every_eligible_source_clock_considered": True,
            "ordering": (
                "sort by entry_time then bucket_start; accept earliest eligible "
                "candidate when entry_time>=prior accepted exit_time; entry equal "
                "to prior exit is allowed; suppress intervening candidates with "
                "no score priority or replacement"
            ),
            "post_support_threshold_or_side_repair": "forbidden",
        },
        "causal_availability": {
            "source_available_at": "fixed 12h bucket end + 48 hours",
            "ceil_5m": "((unix_seconds+299)//300)*300",
            "entry_time": "ceil_5m(source_available_at) + 300 seconds",
            "already_aligned_rule": (
                "an available_at exactly on a 5m boundary still receives the "
                "additional 300-second complete-bar latency"
            ),
            "entry_latency_seconds": 300,
            "feature_rows": "t and strict-prior source rows only",
            "missing_bucket_policy": "reject source; never fill or backdate",
            "retrieval_cutoff": (
                "rows whose frozen available_at or entry_time lies outside a "
                "split are omitted; never backfill or relax a window"
            ),
        },
        "execution": {
            "bar_size": "5m",
            "entry": "open exactly at entry_time after the latency bar closes",
            "hold_bars": 288,
            "hold_hours": 24,
            "scheduled_exit_time": "entry_time + 86400 seconds",
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
            "warmup_source": (
                "2022-07-20T12:00:00Z through 2022-10-31T23:59:59Z"
            ),
            "train": "[2022-11-01T00:00:00Z,2024-01-01T00:00:00Z)",
            "test": "[2024-01-01T00:00:00Z,2025-01-01T00:00:00Z)",
            "eval": "[2025-01-01T00:00:00Z,2026-01-01T00:00:00Z)",
            "forward": "[2026-01-01T00:00:00Z,2026-07-20T00:00:00Z)",
            "stitched_full_horizon": (
                "[2022-11-01T00:00:00Z,2026-07-20T00:00:00Z)"
            ),
            "window_assignment": (
                "UTC half-open windows; entry and scheduled exit both contained"
            ),
            "fit_permission": "no fitted coefficient; every threshold frozen here",
            "later_window_use": (
                "report-only; never select, rerank, invert, refit, or repair"
            ),
        },
        "support_gates": {
            "count_basis": (
                "source-only accepted primary entries after eligibility, split "
                "containment, and chronological non-overlap"
            ),
            "month_share_basis": (
                "maximum calendar-month count divided by total window count; "
                "total entries, not side-specific entries"
            ),
            "train_total_minimum": 45,
            "train_long_minimum": 14,
            "train_short_minimum": 14,
            "train_2022_nov_dec_minimum": 5,
            "train_2023_h1_minimum": 16,
            "train_2023_h2_minimum": 16,
            "train_maximum_month_share": 0.20,
            "test_total_minimum": 35,
            "test_long_minimum": 10,
            "test_short_minimum": 10,
            "test_each_half_minimum": 14,
            "test_each_quarter_minimum": 5,
            "test_maximum_month_share": 0.20,
            "eval_total_minimum": 35,
            "eval_long_minimum": 10,
            "eval_short_minimum": 10,
            "eval_each_half_minimum": 14,
            "eval_each_quarter_minimum": 5,
            "eval_maximum_month_share": 0.20,
            "forward_total_minimum": 18,
            "forward_long_minimum": 5,
            "forward_short_minimum": 5,
            "forward_2026_h1_minimum": 16,
            "forward_maximum_month_share": 0.28,
            "missing_12h_source_buckets": 0,
            "controls": (
                "report source-only incidence for every control; a control may "
                "challenge specificity only if it independently satisfies the "
                "same applicable support floors before its outcomes are opened"
            ),
            "support_failure_action": (
                "reject WCTR-288 without opening market/funding/outcomes; no repair"
            ),
        },
        "performance_gates": {
            "required_sequence": [
                "train",
                "test",
                "eval",
                "forward",
                "stitched_full_horizon",
            ],
            "absolute_return_positive_each_constituent": True,
            "cagr_to_strict_mdd_minimum_each_constituent": 3.0,
            "strict_max_drawdown_maximum_each_constituent": 0.15,
            "weekly_cluster_sign_flip_p_maximum_each_constituent": 0.10,
            "mean_gross_bp_minimum_each_constituent": 20.0,
            "stress_absolute_return_positive_each_constituent": True,
            "positive_subperiods": [
                "train_2023H1",
                "train_2023H2",
                "test_2024H1",
                "test_2024H2",
                "eval_2025H1",
                "eval_2025H2",
                "forward_2026H1",
            ],
            "long_and_short_absolute_return_positive": [
                "train",
                "test",
                "eval",
            ],
            "one_bar_delayed_entry_absolute_return_positive_each_constituent": True,
            "stitched_absolute_return_positive": True,
            "stitched_cagr_to_strict_mdd_minimum": 3.0,
            "stitched_strict_max_drawdown_maximum": 0.15,
            "component_control_full_gate_pass_rejects_specific_mechanism": [
                "transport_only",
                "impulse_only",
                "low_fullness_complement",
                "serialized_size_only",
                "block_weight_only",
                "constant_long_same_clock",
                "constant_short_same_clock",
                "stale_7d",
                "month_side_stratified_random_clock",
            ],
            "direction_flip_is_diagnostic_only": True,
            "control_replacement": "forbidden under WCTR-288",
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
            "snapshot research only; no live or shadow promotion before 90 "
            "forward shadow days pass frozen schema, freshness, revision, and "
            "value-stability checks"
        ),
        "stopping_rule": (
            "stop permanently at first support/train/test/eval/forward/stitched "
            "gate failure; no sign, threshold, rank-window, support-floor, hold, "
            "latency, calendar, or clock repair; no failed-policy inversion"
        ),
    }


def _artifact_core(cfg: Config, source_binding: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(PREREGISTRATION_DOCUMENT) != PREREGISTRATION_DOCUMENT_SHA256:
        raise RuntimeError("WCTR preregistration document file drift")
    frozen_policy = policy()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "config": asdict(cfg),
        "source_manifest": source_binding,
        "policy": frozen_policy,
        "policy_hash": canonical_hash(frozen_policy),
        "outcomes_opened": False,
        "outcome_boundary": PREREGISTRATION_OUTCOME_BOUNDARY,
        "research_sequence": {
            "support_first": "source-only train/test/eval/forward incidence",
            "evaluator_second": "commit and hash-freeze strict evaluator",
            "outcomes": (
                "train, test, eval, forward, then stitched full horizon; stop "
                "at first failure"
            ),
        },
        "preregistration_source": {
            "path": str(PREREGISTRATION_SOURCE),
            "sha256": sha256_file(PREREGISTRATION_SOURCE),
        },
        "preregistration_document": {
            "path": str(PREREGISTRATION_DOCUMENT),
            "sha256": PREREGISTRATION_DOCUMENT_SHA256,
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
        raise RuntimeError("WCTR preregistration canonical hash mismatch")
    if artifact.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("WCTR preregistration protocol drift")
    if artifact.get("policy") != policy():
        raise RuntimeError("WCTR preregistration policy drift")
    if artifact.get("policy_hash") != canonical_hash(policy()):
        raise RuntimeError("WCTR preregistration policy hash drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("WCTR preregistration opened outcomes")
    if artifact.get("outcome_boundary") != PREREGISTRATION_OUTCOME_BOUNDARY:
        raise RuntimeError("WCTR preregistration outcome boundary drift")
    source = artifact.get("preregistration_source")
    expected_source = {
        "path": str(PREREGISTRATION_SOURCE),
        "sha256": sha256_file(PREREGISTRATION_SOURCE),
    }
    if source != expected_source:
        raise RuntimeError("WCTR preregistration source binding drift")
    document = artifact.get("preregistration_document")
    expected_document = {
        "path": str(PREREGISTRATION_DOCUMENT),
        "sha256": PREREGISTRATION_DOCUMENT_SHA256,
    }
    if document != expected_document:
        raise RuntimeError("WCTR preregistration document binding drift")
    if sha256_file(PREREGISTRATION_DOCUMENT) != PREREGISTRATION_DOCUMENT_SHA256:
        raise RuntimeError("WCTR preregistration document file drift")
    raw_config = artifact.get("config")
    if not isinstance(raw_config, dict):
        raise RuntimeError("WCTR preregistration config missing")
    try:
        cfg = Config(**raw_config)
    except TypeError as exc:
        raise RuntimeError("WCTR preregistration config drift") from exc
    _validate_config(cfg, require_new_output=False)
    if _repository_path(path) != _repository_path(cfg.preregistration_output):
        raise RuntimeError("WCTR preregistration output-path binding drift")
    if artifact.get("source_manifest") != _validate_source_manifest(
        cfg.source_manifest
    ):
        raise RuntimeError("WCTR preregistration source-manifest binding drift")
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
