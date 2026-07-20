"""Write the outcome-blind BFRT-288 singleton preregistration.

Only the frozen source manifest and artifact byte hashes are inspected here.
The raw/normalized fee-rate values and all BTC outcomes remain unopened.
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


POLICY_ID = "BFRT-288"
PROTOCOL_VERSION = "block_feerate_breadth_transport_preregistration_v1"
SOURCE_PROTOCOL_VERSION = "mempool_block_feerate_history_source_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = Path(
    "results/mempool_block_feerates_source_manifest_2026-07-20.json"
)
EXPECTED_SOURCE_MANIFEST_FILE_SHA256 = (
    "1ad4ee8bc9e81d3f7e7169426de21f0398bc6ab1d739e6f27e6d9ff02f331555"
)
EXPECTED_SOURCE_MANIFEST_HASH = (
    "fe616bcf294e8b3b2abc6dec124e922f77df4bca47a86249fc270f2af6b46f21"
)
SOURCE_DECISION = Path(
    "docs/block-feerate-breadth-transport-mechanism-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "f6dd7d52b03d1370483a1157e24efad45b6886230bafed0931b0ac88cbde82cb"
)
SOURCE_FREEZE = Path(
    "docs/block-feerate-breadth-transport-source-freeze-2026-07-20.md"
)
SOURCE_FREEZE_SHA256 = (
    "0336a5d567b894964ad3aaca541548a7962e16b6a79a34dcda4bb5478e2f3092"
)
SOURCE_BUILDER = Path("training/download_mempool_block_feerate_history.py")
SOURCE_BUILDER_SHA256 = (
    "ebd30dd109a92c4dc5a2a6a444a5d5760fa4360c7fd848b02923f0670e4a2910"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_block_feerate_breadth_transport.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/block-feerate-breadth-transport-preregistration-2026-07-20.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "04b77d6358aead4ecae494f75c71e1c0f40066fc466df01e4a7b9ede4c7fab2c"
)
DEFAULT_OUTPUT = Path(
    "results/block_feerate_breadth_transport_preregistration_2026-07-20.json"
)

EXPECTED_SOURCE_CONFIG = {
    "raw_output": "data/mempool_block_feerates_3y_2026-07-20.raw.json.gz",
    "output_csv": "data/mempool_block_feerates_3y_2026-07-20.csv.gz",
    "manifest_output": str(SOURCE_MANIFEST),
    "timeout_sec": 30.0,
    "request_pause_sec": 0.25,
    "maximum_retries": 8,
    "maximum_response_bytes": 5_000_000,
    "minimum_response_rows": 2_000,
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
    "endpoint": "https://mempool.space/api/v1/mining/blocks/fee-rates/3y",
    "official_rest_docs": "https://mempool.space/docs/api/rest",
    "upstream_repository": "https://github.com/mempool/mempool",
    "upstream_commit": "e9d6cf8c042f946be53e372bb36530cd7b7851a4",
    "http_status": 200,
    "final_url": "https://mempool.space/api/v1/mining/blocks/fee-rates/3y",
    "retrieved_at_utc": "2026-07-20T09:02:30.628066Z",
    "response_headers": {
        "date": "Mon, 20 Jul 2026 09:02:28 GMT",
        "etag": 'W/"4e564-RJvYwNUJqLGdQWvvCgzgfWMQELc"',
        "last-modified": None,
        "content-type": "application/json; charset=utf-8",
    },
    "raw_response_bytes": 320_868,
    "raw_response_sha256": (
        "480a99c3ebfd49f98511f94fe05d9f8d76a2e28ebef7cde937768b0d4321e008"
    ),
    "response_rows": 2_193,
    "retained_rows": 2_191,
    "edge_rows_dropped": 2,
    "first_dropped_bucket_start_utc": "2023-07-20T00:00:00Z",
    "last_dropped_bucket_start_utc": "2026-07-20T00:00:00Z",
    "first_retained_bucket_start_utc": "2023-07-20T12:00:00Z",
    "last_retained_bucket_start_utc": "2026-07-19T12:00:00Z",
    "first_retained_avg_height": 799_556,
    "last_retained_avg_height": 958_761,
    "strictly_increasing_unique_timestamps": True,
    "strictly_increasing_unique_heights": True,
    "unique_bucket_ids": True,
    "percentile_ordering_valid": True,
    "missing_12h_buckets": 0,
    "maximum_bucket_gap_seconds": 43_200,
    "missing_bucket_policy": "retain gap; never forward-fill or backdate",
}
EXPECTED_RAW_ARTIFACT = {
    "path": EXPECTED_SOURCE_CONFIG["raw_output"],
    "sha256": "4309dfbbdb08b89cd9cc92a341bd6186146b1e67adc2c3f926c8154ddabc4898",
    "bytes": 35_671,
    "decompressed_sha256": EXPECTED_SOURCE_AUDIT["raw_response_sha256"],
    "encoding": "gzip of exact HTTP response bytes; mtime=0",
}
SOURCE_COLUMNS = [
    "bucket_start_utc",
    "bucket_end_utc",
    "available_at_utc",
    "avg_height",
    "avg_timestamp",
    "fee_p0",
    "fee_p10",
    "fee_p25",
    "fee_p50",
    "fee_p75",
    "fee_p90",
    "fee_p100",
]
EXPECTED_NORMALIZED_ARTIFACT = {
    "path": EXPECTED_SOURCE_CONFIG["output_csv"],
    "sha256": "007d13ba756fd29faae1ae87caa11554438b54bb5028f24b2f0c21ddf3a0e55d",
    "bytes": 47_095,
    "rows": 2_191,
    "columns": SOURCE_COLUMNS,
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
    "missing_bucket_policy": "retain gap; never forward-fill or backdate",
}
SOURCE_OUTCOME_BOUNDARY = {
    "btc_market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "premium_or_oi_rows_loaded": 0,
    "return_or_pnl_fields": 0,
    "bfrt_features_derived": 0,
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
    "magnitude_only": (
        "retain magnitude_rank >= 0.75 and side=sign(location); remove coherence "
        "and tail-divergence filters; sign(location) must still equal "
        "sign(signed_coherence); build its own chronological non-overlap clock"
    ),
    "tail_only": (
        "retain magnitude_rank>=0.75 and coherence>=0.60 but require "
        "tail_divergence_rank>0.75; use its own chronological non-overlap "
        "clock and side=sign(signed_coherence)"
    ),
    "constant_long_same_clock": "same primary entry/exit clock with side fixed long",
    "constant_short_same_clock": "same primary entry/exit clock with side fixed short",
    "stale_7d": (
        "at source bucket t use the fully formed primary feature/ranks from t-14; "
        "retain t availability and execution clock"
    ),
    "month_side_stratified_random_clock": (
        "sample without replacement from rank-ready source clocks preserving "
        "primary calendar-month and side counts; seed 20260720"
    ),
    "one_bar_delayed_entry": (
        "same primary signals and sides; shift entry and scheduled exit exactly "
        "one complete 5m bar later"
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
            raise RuntimeError(f"BFRT JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(
        _repository_path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("BFRT manifest must be a JSON object")
    return payload


def _manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_hash"}


def _validate_config(cfg: Config, *, require_new_output: bool) -> None:
    manifest = _repository_path(cfg.source_manifest)
    output = _repository_path(cfg.preregistration_output)
    if manifest.suffix != ".json" or output.suffix != ".json":
        raise ValueError("BFRT source manifest and preregistration must be JSON")
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
        raise ValueError("BFRT preregistration output aliases a protected source")
    if require_new_output and output.exists():
        raise FileExistsError("BFRT preregistration is immutable")


def _require_exact(manifest: dict[str, Any], key: str, expected: Any) -> None:
    if manifest.get(key) != expected:
        raise RuntimeError(f"BFRT frozen source {key} drift")


def _validate_source_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = _repository_path(path)
    if manifest_path != _repository_path(SOURCE_MANIFEST):
        raise RuntimeError("BFRT source manifest path differs from frozen source")
    if sha256_file(manifest_path) != EXPECTED_SOURCE_MANIFEST_FILE_SHA256:
        raise RuntimeError("BFRT source manifest file SHA drift")
    manifest = _read_json(manifest_path)
    core = _manifest_core(manifest)
    if canonical_hash(core) != manifest.get("manifest_hash"):
        raise RuntimeError("BFRT source manifest canonical hash mismatch")
    if manifest.get("manifest_hash") != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("BFRT source frozen manifest hash drift")
    if manifest.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
        raise RuntimeError("BFRT source protocol version drift")
    _require_exact(manifest, "config", EXPECTED_SOURCE_CONFIG)
    _require_exact(manifest, "source_decision", EXPECTED_SOURCE_DECISION)
    _require_exact(manifest, "source_builder", EXPECTED_SOURCE_BUILDER)
    _require_exact(manifest, "source_audit", EXPECTED_SOURCE_AUDIT)
    _require_exact(manifest, "raw_artifact", EXPECTED_RAW_ARTIFACT)
    _require_exact(
        manifest, "normalized_artifact", EXPECTED_NORMALIZED_ARTIFACT
    )
    _require_exact(
        manifest, "causal_availability", EXPECTED_CAUSAL_AVAILABILITY
    )
    _require_exact(manifest, "outcome_boundary", SOURCE_OUTCOME_BOUNDARY)
    _require_exact(manifest, "data_use", EXPECTED_DATA_USE)

    if sha256_file(SOURCE_DECISION) != SOURCE_DECISION_SHA256:
        raise RuntimeError("BFRT mechanism decision file drift")
    if sha256_file(SOURCE_FREEZE) != SOURCE_FREEZE_SHA256:
        raise RuntimeError("BFRT source freeze file drift")
    if sha256_file(SOURCE_BUILDER) != SOURCE_BUILDER_SHA256:
        raise RuntimeError("BFRT source builder file drift")
    for label, expected in (
        ("raw", EXPECTED_RAW_ARTIFACT),
        ("normalized", EXPECTED_NORMALIZED_ARTIFACT),
    ):
        artifact_path = _repository_path(expected["path"])
        if artifact_path == manifest_path:
            raise RuntimeError(f"BFRT {label} artifact aliases source manifest")
        if artifact_path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"BFRT {label} artifact byte-size drift")
        if sha256_file(artifact_path) != expected["sha256"]:
            raise RuntimeError(f"BFRT {label} artifact SHA drift")

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
        "causal_availability": EXPECTED_CAUSAL_AVAILABILITY,
        "outcome_boundary": SOURCE_OUTCOME_BOUNDARY,
    }


def policy() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "singleton": True,
        "source_features": {
            "percentiles": [10, 25, 50, 75, 90],
            "excluded_from_features": [0, 100],
            "log_surface": "x[p,t] = log1p(fee_p[t])",
            "transport_horizon_buckets": 2,
            "transport_horizon_hours": 24,
            "delta": "d[p,t] = x[p,t] - x[p,t-2]",
            "location": "median(d[10,t],d[25,t],d[50,t],d[75,t],d[90,t])",
            "l1_motion": "sum_p(abs(d[p,t]))",
            "signed_coherence": "sum_p(d[p,t]) / l1_motion",
            "coherence": "abs(signed_coherence)",
            "tail_divergence": (
                "abs((d[90,t]-d[75,t])-(d[25,t]-d[10,t])) / "
                "l1_motion"
            ),
            "numeric_contract": (
                "IEEE-754 binary64 math.log1p; no epsilon, clipping, rounding, "
                "or imputation; exact binary64 equality defines rank ties"
            ),
            "invalid_feature": (
                "reject unless t,t-1,t-2 are consecutive 12h buckets, all five "
                "percentiles are finite/non-negative, l1_motion>0, location!=0, "
                "signed_coherence!=0, and sign(signed_coherence)==sign(location)"
            ),
        },
        "normalization": {
            "method": "strict-prior rolling empirical midrank",
            "midrank_formula": (
                "(count(prior < current) + 0.5*count(prior == current)) / "
                "prior_count"
            ),
            "lookback_valid_feature_buckets": 180,
            "minimum_prior_valid_feature_buckets": 120,
            "magnitude_rank": "midrank of abs(location)",
            "tail_divergence_rank": "midrank of tail_divergence",
            "current_row_excluded": True,
            "prior_rows_must_be_available_before_current": True,
        },
        "eligibility": {
            "common": (
                "magnitude_rank>=0.75 and coherence>=0.60 and "
                "tail_divergence_rank<=0.75"
            ),
            "long": "common and signed_coherence>0 and location>0",
            "short": "common and signed_coherence<0 and location<0",
            "side": "sign(signed_coherence), which must equal sign(location)",
            "every_eligible_source_clock_considered": True,
            "ordering": (
                "sort by entry_time then bucket_start; accept only when entry_time "
                ">= prior accepted exit_time; suppress intervening signals with "
                "no score priority or replacement"
            ),
            "post_support_threshold_or_side_repair": "forbidden",
        },
        "causal_availability": {
            "source_available_at": "fixed 12h bucket end + 48 hours",
            "entry_time": "ceil_5m(source_available_at) + one complete 5m bar",
            "entry_latency_seconds": 300,
            "feature_rows": "t and strict-prior source rows only",
            "missing_bucket_policy": "invalidate affected features; never fill",
        },
        "execution": {
            "bar_size": "5m",
            "entry": "open at entry_time after the latency bar closes",
            "hold_bars": 288,
            "hold_hours": 24,
            "interval": "[entry_time, scheduled_exit_time)",
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
                "2023-07-20T12:00:00Z through 2023-10-31T23:59:59Z"
            ),
            "train": "[2023-11-01T00:00:00Z,2025-01-01T00:00:00Z)",
            "test": "[2025-01-01T00:00:00Z,2026-01-01T00:00:00Z)",
            "eval": "[2026-01-01T00:00:00Z,2026-07-20T00:00:00Z)",
            "window_assignment": "entry and scheduled exit must both lie in window",
            "fit_permission": "no fitted coefficient; thresholds frozen here",
            "test_eval_use": "report-only; never select, rerank, invert, or repair",
        },
        "support_gates": {
            "count_basis": (
                "accepted primary non-overlap entries assigned by entry_time"
            ),
            "train_total_minimum": 80,
            "train_long_minimum": 25,
            "train_short_minimum": 25,
            "train_2023_nov_dec_minimum": 8,
            "train_2024_h1_minimum": 14,
            "train_2024_h2_minimum": 14,
            "train_maximum_month_share": 0.15,
            "test_total_minimum": 35,
            "test_long_minimum": 12,
            "test_short_minimum": 12,
            "test_each_half_minimum": 14,
            "test_each_quarter_minimum": 6,
            "test_maximum_month_share": 0.20,
            "eval_total_minimum": 20,
            "eval_long_minimum": 6,
            "eval_short_minimum": 6,
            "eval_2026_h1_minimum": 18,
            "eval_maximum_month_share": 0.25,
            "missing_12h_source_buckets": 0,
            "support_failure_action": (
                "reject BFRT-288 without opening market/funding/outcomes; no repair"
            ),
        },
        "performance_gates": {
            "required_sequence": ["train", "test", "eval"],
            "absolute_return_positive_each": True,
            "cagr_to_strict_mdd_minimum_each": 3.0,
            "strict_max_drawdown_maximum_each": 0.15,
            "weekly_cluster_sign_flip_p_maximum_each": 0.10,
            "mean_gross_bp_minimum_each": 20.0,
            "stress_absolute_return_positive_each": True,
            "positive_subperiods": [
                "train_2024H1",
                "train_2024H2",
                "test_2025H1",
                "test_2025H2",
                "eval_2026H1",
            ],
            "long_and_short_absolute_return_positive": ["train", "test"],
            "one_bar_delayed_entry_absolute_return_positive_each": True,
            "component_control_full_gate_pass_rejects_specific_mechanism": [
                "magnitude_only",
                "tail_only",
                "constant_long_same_clock",
                "constant_short_same_clock",
                "stale_7d",
                "month_side_stratified_random_clock",
            ],
            "direction_flip_is_diagnostic_only": True,
            "post_outcome_repair": "forbidden",
        },
        "strict_accounting": {
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
            "forward shadow days pass frozen schema, freshness, and value-stability "
            "checks"
        ),
        "stopping_rule": (
            "stop permanently at first support/train/test/eval gate failure; "
            "no sign, threshold, rank-window, hold, latency, or calendar repair"
        ),
    }


def _artifact_core(cfg: Config, source_binding: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(PREREGISTRATION_DOCUMENT) != PREREGISTRATION_DOCUMENT_SHA256:
        raise RuntimeError("BFRT preregistration document file drift")
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
            "support_first": "source-only train/test/eval incidence",
            "evaluator_second": "commit and hash-freeze strict evaluator",
            "outcomes": "train, then test, then eval; stop at first failure",
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
        raise RuntimeError("BFRT preregistration canonical hash mismatch")
    if artifact.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("BFRT preregistration protocol drift")
    if artifact.get("policy") != policy():
        raise RuntimeError("BFRT preregistration policy drift")
    if artifact.get("policy_hash") != canonical_hash(policy()):
        raise RuntimeError("BFRT preregistration policy hash drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("BFRT preregistration opened outcomes")
    if artifact.get("outcome_boundary") != PREREGISTRATION_OUTCOME_BOUNDARY:
        raise RuntimeError("BFRT preregistration outcome boundary drift")
    source = artifact.get("preregistration_source")
    expected_source = {
        "path": str(PREREGISTRATION_SOURCE),
        "sha256": sha256_file(PREREGISTRATION_SOURCE),
    }
    if source != expected_source:
        raise RuntimeError("BFRT preregistration source binding drift")
    document = artifact.get("preregistration_document")
    expected_document = {
        "path": str(PREREGISTRATION_DOCUMENT),
        "sha256": PREREGISTRATION_DOCUMENT_SHA256,
    }
    if document != expected_document:
        raise RuntimeError("BFRT preregistration document binding drift")
    if sha256_file(PREREGISTRATION_DOCUMENT) != PREREGISTRATION_DOCUMENT_SHA256:
        raise RuntimeError("BFRT preregistration document file drift")
    raw_config = artifact.get("config")
    if not isinstance(raw_config, dict):
        raise RuntimeError("BFRT preregistration config missing")
    try:
        cfg = Config(**raw_config)
    except TypeError as exc:
        raise RuntimeError("BFRT preregistration config drift") from exc
    _validate_config(cfg, require_new_output=False)
    if _repository_path(path) != _repository_path(cfg.preregistration_output):
        raise RuntimeError("BFRT preregistration output-path binding drift")
    if artifact.get("source_manifest") != _validate_source_manifest(
        cfg.source_manifest
    ):
        raise RuntimeError("BFRT preregistration source-manifest binding drift")
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
