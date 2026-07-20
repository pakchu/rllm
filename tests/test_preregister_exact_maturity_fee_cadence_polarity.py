from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from training import preregister_exact_maturity_fee_cadence_polarity as prereg


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _write_source(path: Path, payload: bytes = b"not a gzip csv; prereg must not parse values\n") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _source_manifest(tmp_path: Path, **overrides: Any) -> tuple[Path, dict[str, Any]]:
    source_path = tmp_path / "source.csv.gz"
    source_sha = _write_source(source_path)
    reference_path = tmp_path / "reference.csv.gz"
    reference_sha = _write_source(reference_path, b"frozen reference bytes\n")
    core: dict[str, Any] = {
        "protocol_version": prereg.SOURCE_PROTOCOL_VERSION,
        "source_decision": {"path": str(prereg.SOURCE_DECISION), "sha256": prereg.SOURCE_DECISION_SHA256},
        "source_builder": {"path": str(prereg.SOURCE_BUILDER), "sha256": prereg.sha256_file(prereg.SOURCE_BUILDER)},
        "config": {"output_csv": str(source_path), "manifest_output": str(tmp_path / "source_manifest.json")},
        "source_audit": {
            "expected_rows": prereg.FROZEN_ROWS,
            "observed_rows": prereg.FROZEN_ROWS,
            "start_height": prereg.FROZEN_START_HEIGHT,
            "end_height": prereg.FROZEN_END_HEIGHT,
            "latest_eligible_packet_end": prereg.FROZEN_END_HEIGHT - 6,
            "height_links_checked": prereg.FROZEN_ROWS - 1,
            "end_timestamp_exclusive": prereg.FROZEN_END_TIMESTAMP_EXCLUSIVE,
            "complete_inclusive_height_range": True,
            "unique_block_hashes": True,
            "all_rows_pre_cutoff": True,
            "utxo_identity_checked": True,
        },
        "reference_audit": {
            "reference_path": str(reference_path),
            "reference_sha256": reference_sha,
            "rows_cross_checked": prereg.FROZEN_ROWS,
            "columns_cross_checked": list(prereg.REFERENCE_COLUMNS),
            "all_basic_fields_match_reference": True,
        },
        "output": {
            "path": str(source_path),
            "sha256": source_sha,
            "bytes": source_path.stat().st_size,
            "columns": list(prereg.SOURCE_COLUMNS),
        },
        "outcome_boundary": dict(prereg.SOURCE_OUTCOME_BOUNDARY),
    }
    for key, value in overrides.items():
        if key == "source_sha":
            core["output"]["sha256"] = value
        elif key == "columns":
            core["output"]["columns"] = value
        elif key == "outcome_boundary":
            core["outcome_boundary"] = value
        elif key == "source_decision":
            core["source_decision"] = value
        elif key == "source_builder":
            core["source_builder"] = value
        elif key == "source_audit":
            core["source_audit"].update(value)
        else:
            core[key] = value
    manifest = {**core, "manifest_hash": _canonical_hash(core)}
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prereg.SOURCE_MANIFEST = path
    prereg.EXPECTED_SOURCE_MANIFEST_SHA256 = prereg.sha256_file(path)
    prereg.EXPECTED_SOURCE_MANIFEST_HASH = manifest["manifest_hash"]
    prereg.EXPECTED_SOURCE_OUTPUT = source_path
    prereg.EXPECTED_SOURCE_OUTPUT_SHA256 = source_sha
    prereg.EXPECTED_SOURCE_OUTPUT_BYTES = source_path.stat().st_size
    prereg.EXPECTED_SOURCE_BUILDER_SHA256 = prereg.sha256_file(prereg.SOURCE_BUILDER)
    prereg.EXPECTED_REFERENCE = reference_path
    prereg.EXPECTED_REFERENCE_SHA256 = reference_sha
    return path, manifest


def _cfg(tmp_path: Path, manifest: Path | None = None, output: Path | None = None) -> prereg.Config:
    return prereg.Config(
        source_manifest=str(manifest or tmp_path / "source_manifest.json"),
        preregistration_output=str(output or tmp_path / "prereg.json"),
    )


def test_writes_exact_source_only_block_emfc_policy_controls_gates_and_hash(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _source_manifest(tmp_path)
    cfg = _cfg(tmp_path, manifest_path)

    with patch("gzip.open", side_effect=AssertionError("CSV must not be decompressed")):
        artifact = prereg.write_preregistration(cfg)

    assert artifact == json.loads(Path(cfg.preregistration_output).read_text(encoding="utf-8"))
    core = {k: v for k, v in artifact.items() if k != "manifest_hash"}
    assert artifact["manifest_hash"] == _canonical_hash(core)
    assert artifact["protocol_version"] == prereg.PROTOCOL_VERSION
    assert artifact["policy_id"] == "EMFC-864"
    assert artifact["outcomes_opened"] is False
    assert artifact["outcome_boundary"] == {
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "return_rows_loaded": 0,
        "market_values_read": 0,
        "funding_values_read": 0,
        "return_or_pnl_fields": 0,
        "source_csv_values_read": 0,
        "source_manifest_only": True,
    }
    assert artifact["source_manifest"]["manifest_hash"] == manifest["manifest_hash"]
    assert artifact["source_manifest"]["sha256"] == prereg.sha256_file(manifest_path)
    assert artifact["source_manifest"]["source_decision"] == manifest["source_decision"]
    assert artifact["source_manifest"]["source_builder"] == manifest["source_builder"]
    assert artifact["mechanism_decision"] == {
        "path": str(prereg.MECHANISM_DECISION),
        "sha256": prereg.MECHANISM_DECISION_SHA256,
    }
    assert artifact["novelty_comparators"] == prereg.NOVELTY_COMPARATORS

    policy = artifact["policy"]
    assert policy["singleton"] is True
    assert policy["source_features"] == {
        "origin_height": "h-100",
        "maturity_height": "h",
        "confirmation_height": "h+6",
        "matured_fee_component": "total_fees[h-100]",
        "fee_pressure": "log1p(total_fees[h-100])",
        "maturity_elapsed_seconds": "mediantime[h]-mediantime[h-100]",
        "cadence_compression": "-log(maturity_elapsed_seconds/60000)",
        "valid_height": "all required heights exist, total_fees[h-100]>=0, and maturity_elapsed_seconds>0",
        "expected_candidate_heights": 212_989,
    }
    assert policy["causal_availability"]["confirmation_blocks"] == 6
    assert policy["causal_availability"]["historical_embargo_seconds"] == 7_200
    assert policy["causal_availability"]["raw_available"] == (
        "max(timestamp[h:h+6])+7200 seconds"
    )
    assert policy["causal_availability"]["publication_latency"] == "one complete 5m latency bar"
    assert policy["causal_availability"]["entry_time"] == "decision_boundary+5m"
    assert policy["normalization"] == {
        "method": "strict-prior rolling empirical midrank",
        "midrank_formula": "(count(prior < current) + 0.5 * count(prior == current)) / prior_count",
        "reference_valid_heights": 26_208,
        "nominal_reference_days_at_144_blocks": 182,
        "require_full_reference": True,
        "invalid_heights_excluded_without_reset": True,
        "fee_rank": "midrank of fee_pressure over the last 26208 valid maturity heights strictly below h",
        "cadence_rank": "midrank of cadence_compression over the last 26208 valid maturity heights strictly below h",
    }
    assert policy["eligibility"]["high_pressure_compressed"].endswith("side=-1 short")
    assert policy["eligibility"]["low_pressure_expanded"].endswith("side=+1 long")
    assert "immediately preceding valid state" in policy["eligibility"]["onset"]
    assert policy["execution"] == {
        "bar_size": "5m",
        "hold_bars": 864,
        "notional_leverage": 0.5,
        "base_cost_bp_per_notional_per_side": 6,
        "stress_cost_bp_per_notional_per_side": 10,
        "funding": "exact funding, entry-inclusive/exit-exclusive, fixed entry quantity",
    }
    assert policy["controls"] == prereg.CONTROL_DEFINITIONS
    assert policy["source_integrity_gates"]["exact_candidate_heights"] == 212_989
    assert policy["event_support_gates"]["train_2021_2022_total_minimum"] == 60
    assert policy["event_support_gates"]["train_2021_2022_total_maximum"] == 200
    assert policy["event_support_gates"]["selection_2023_total_minimum"] == 24
    assert policy["event_support_gates"]["selection_2023_total_maximum"] == 105
    assert policy["event_support_gates"]["exact_72h_boundary_gap_share_maximum"] == 0.50
    assert policy["source_novelty_gates"]["feature_spearman_absolute_maximum"] == 0.90
    assert policy["source_novelty_gates"]["shadow_exposure_absolute_correlation_maximum"] == 0.80
    assert (
        policy["source_novelty_gates"][
            "existing_network_alpha_exposure_absolute_correlation_maximum"
        ]
        == 0.35
    )
    assert policy["performance_gates"]["cagr_to_strict_mdd_minimum_each"] == 3.0
    assert policy["performance_gates"]["strict_max_drawdown_maximum_each"] == 0.15
    assert set(policy["performance_gates"]["component_control_full_gate_pass_rejects_mechanism"]) >= {
        "fee_only",
        "cadence_only",
        "same_height_fee",
        "pseudo_maturity_99",
        "pseudo_maturity_101",
        "daily_aggregate_shadow",
        "stale_7d",
    }
    assert artifact["research_sequence"] == {
        "train_first": "2021-2022",
        "selection_second": "2023 only after exact train pass",
        "sealed": "2024+",
    }
    assert prereg.load_preregistration(cfg.preregistration_output) == artifact


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"source_decision": {"path": str(prereg.SOURCE_DECISION), "sha256": "0" * 64}}, "decision"),
        ({"source_builder": {"path": str(prereg.SOURCE_BUILDER), "sha256": "0" * 64}}, "builder SHA"),
        ({"source_builder": {"path": "training/other.py", "sha256": "0" * 64}}, "builder path"),
        ({"columns": prereg.SOURCE_COLUMNS[:-1]}, "schema"),
        ({"outcome_boundary": {**prereg.SOURCE_OUTCOME_BOUNDARY, "market_rows_loaded": 1}}, "outcome boundary"),
        ({"source_audit": {"latest_eligible_packet_end": 823721}}, "six-successor"),
        ({"source_audit": {"complete_inclusive_height_range": False}}, "height range"),
        ({"source_audit": {"utxo_identity_checked": False}}, "UTXO identity"),
    ],
)
def test_rejects_source_manifest_schema_builder_decision_boundary_drift(
    tmp_path: Path, override: dict[str, Any], message: str
) -> None:
    manifest_path, _ = _source_manifest(tmp_path, **override)
    with pytest.raises(RuntimeError, match=message):
        prereg.write_preregistration(_cfg(tmp_path, manifest_path))


def test_rejects_manifest_hash_and_source_file_hash_drift(tmp_path: Path) -> None:
    manifest_path, manifest = _source_manifest(tmp_path)
    manifest["output"]["bytes"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest hash"):
        prereg.write_preregistration(_cfg(tmp_path, manifest_path))

    manifest_path, _ = _source_manifest(tmp_path / "sha", source_sha="0" * 64)
    with pytest.raises(RuntimeError, match="source file SHA"):
        prereg.write_preregistration(_cfg(tmp_path / "sha", manifest_path))


def test_rejects_artifact_path_alias_and_path_config_errors(tmp_path: Path) -> None:
    manifest_path, _ = _source_manifest(tmp_path)
    with pytest.raises(ValueError, match="alias"):
        prereg.write_preregistration(_cfg(tmp_path, manifest_path, manifest_path))
    with pytest.raises(ValueError, match="JSON"):
        prereg.write_preregistration(_cfg(tmp_path, manifest_path, tmp_path / "artifact.txt"))
    with pytest.raises(ValueError, match="source file"):
        prereg.write_preregistration(_cfg(tmp_path, manifest_path, prereg.PREREGISTRATION_SOURCE))
    comparator = Path(next(iter(prereg.NOVELTY_COMPARATORS.values()))["path"])
    with pytest.raises(ValueError, match="source file"):
        prereg.write_preregistration(_cfg(tmp_path, manifest_path, comparator))
    source_path = Path(_source_manifest(tmp_path / "source-alias")[1]["output"]["path"])
    source_manifest = tmp_path / "source-alias" / "source_manifest.json"
    with pytest.raises(ValueError, match="JSON|source data"):
        prereg.write_preregistration(_cfg(tmp_path / "source-alias", source_manifest, source_path))


def test_rejects_mechanism_and_novelty_comparator_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _source_manifest(tmp_path)
    cfg = _cfg(tmp_path, manifest_path)

    monkeypatch.setattr(prereg, "MECHANISM_DECISION_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="mechanism decision drift"):
        prereg.write_preregistration(cfg)
    monkeypatch.undo()

    manifest_path, _ = _source_manifest(tmp_path / "comparator")
    comparator_definitions = {
        key: dict(value) for key, value in prereg.NOVELTY_COMPARATORS.items()
    }
    first = next(iter(comparator_definitions))
    comparator_definitions[first]["sha256"] = "0" * 64
    monkeypatch.setattr(prereg, "NOVELTY_COMPARATORS", comparator_definitions)
    with pytest.raises(RuntimeError, match="novelty comparator drift"):
        prereg.write_preregistration(_cfg(tmp_path / "comparator", manifest_path))


def test_load_rejects_preregistration_policy_and_outcome_boundary_drift(tmp_path: Path) -> None:
    manifest_path, _ = _source_manifest(tmp_path)
    cfg = _cfg(tmp_path, manifest_path)
    artifact = prereg.write_preregistration(cfg)

    drift = dict(artifact)
    drift["policy"] = {**drift["policy"], "policy_id": "EMFC-2"}
    core = {k: v for k, v in drift.items() if k != "manifest_hash"}
    drift["manifest_hash"] = _canonical_hash(core)
    Path(cfg.preregistration_output).write_text(json.dumps(drift), encoding="utf-8")
    with pytest.raises(RuntimeError, match="policy drift"):
        prereg.load_preregistration(cfg.preregistration_output)

    drift = dict(artifact)
    drift["outcome_boundary"] = {**drift["outcome_boundary"], "source_csv_values_read": 1}
    core = {k: v for k, v in drift.items() if k != "manifest_hash"}
    drift["manifest_hash"] = _canonical_hash(core)
    Path(cfg.preregistration_output).write_text(json.dumps(drift), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source-only boundary"):
        prereg.load_preregistration(cfg.preregistration_output)

    tampered = dict(artifact)
    tampered["outcomes_opened"] = True
    core = {k: v for k, v in tampered.items() if k != "manifest_hash"}
    tampered["manifest_hash"] = _canonical_hash(core)
    Path(cfg.preregistration_output).write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="opened outcomes"):
        prereg.load_preregistration(cfg.preregistration_output)

    tampered = dict(artifact)
    tampered["source_manifest"] = {**tampered["source_manifest"], "manifest_hash": "0" * 64}
    core = {k: v for k, v in tampered.items() if k != "manifest_hash"}
    tampered["manifest_hash"] = _canonical_hash(core)
    Path(cfg.preregistration_output).write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source-manifest binding"):
        prereg.load_preregistration(cfg.preregistration_output)
