from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from training import preregister_utxo_fee_clearing_polarity as prereg


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
            "rows_cross_checked": prereg.FROZEN_ROWS,
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
    return path, manifest


def _cfg(tmp_path: Path, manifest: Path | None = None, output: Path | None = None) -> prereg.Config:
    return prereg.Config(
        source_manifest=str(manifest or tmp_path / "source_manifest.json"),
        preregistration_output=str(output or tmp_path / "prereg.json"),
    )


def test_writes_exact_source_only_ufcp_policy_controls_gates_and_hash(tmp_path: Path) -> None:
    manifest_path, manifest = _source_manifest(tmp_path)
    cfg = _cfg(tmp_path, manifest_path)

    with patch("gzip.open", side_effect=AssertionError("CSV must not be decompressed")):
        artifact = prereg.write_preregistration(cfg)

    assert artifact == json.loads(Path(cfg.preregistration_output).read_text(encoding="utf-8"))
    core = {k: v for k, v in artifact.items() if k != "manifest_hash"}
    assert artifact["manifest_hash"] == _canonical_hash(core)
    assert artifact["protocol_version"] == prereg.PROTOCOL_VERSION
    assert artifact["policy_id"] == "UFCP-1"
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

    policy = artifact["policy"]
    assert policy["singleton"] is True
    assert policy["source_features"] == {
        "source_day": "UTC day D aggregated from completed confirmed Bitcoin blocks only",
        "edges": "sum(total_inputs + total_outputs)",
        "fee_burden": "log(sum(total_fees) / edges)",
            "utxo_polarity": "sum(utxo_set_change) / edges",
            "invalid_day": "reject a day when block count <72, edges<=0, total_fees<=0, or six successor blocks are unavailable",
        "daily_source_minimum_blocks": 72,
        "usable_range_requires_no_missing_utc_source_day": True,
    }
    assert policy["causal_availability"]["day_d_unavailable_before"] == "D+2 00:00 UTC"
    assert policy["causal_availability"]["hash_linked_successors_after_final_included_block"] == 6
    assert policy["causal_availability"]["publication_latency"] == "one complete 5m latency bar"
    assert policy["causal_availability"]["entry_time"] == "D+2 00:05 UTC"
    assert policy["normalization"] == {
        "method": "strict-prior rolling empirical midrank",
        "midrank_formula": "(count(prior < current) + 0.5 * count(prior == current)) / prior_count",
        "lookback_source_days": 180,
        "minimum_prior_source_days": 120,
        "fee_rank": "midrank of fee_burden over strict-prior source days",
        "polarity_rank": "midrank of utxo_polarity over strict-prior source days",
    }
    assert policy["eligibility"] == {
        "long": "fee_rank >= 0.75 and polarity_rank >= 0.75",
        "short": "fee_rank >= 0.75 and polarity_rank <= 0.25",
        "every_eligible_day_considered": True,
        "ordering": "chronological non-overlap",
    }
    assert policy["execution"] == {
        "bar_size": "5m",
        "hold_bars": 288,
        "notional_leverage": 0.5,
        "base_cost_bp_per_notional_per_side": 6,
        "stress_cost_bp_per_notional_per_side": 10,
        "funding": "exact funding, entry-inclusive/exit-exclusive, fixed entry quantity",
    }
    assert policy["controls"] == prereg.CONTROL_DEFINITIONS
    assert policy["support_floors"]["train_2021_2022_total_minimum"] == 60
    assert policy["support_floors"]["train_each_year_minimum"] == 24
    assert policy["support_floors"]["selection_2023_total_minimum"] == 24
    assert policy["support_floors"]["selection_each_half_minimum"] == 10
    assert policy["support_floors"]["maximum_month_share"] == "<=0.15 separately in train and selection"
    assert policy["performance_gates"] == {
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
    source_path = Path(_source_manifest(tmp_path / "source-alias")[1]["output"]["path"])
    source_manifest = tmp_path / "source-alias" / "source_manifest.json"
    with pytest.raises(ValueError, match="JSON|source data"):
        prereg.write_preregistration(_cfg(tmp_path / "source-alias", source_manifest, source_path))


def test_load_rejects_preregistration_policy_and_outcome_boundary_drift(tmp_path: Path) -> None:
    manifest_path, _ = _source_manifest(tmp_path)
    cfg = _cfg(tmp_path, manifest_path)
    artifact = prereg.write_preregistration(cfg)

    drift = dict(artifact)
    drift["policy"] = {**drift["policy"], "policy_id": "UFCP-2"}
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
