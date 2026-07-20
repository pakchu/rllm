from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from training import preregister_fee_endpoint_topology_disagreement as prereg


def _write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return prereg.sha256_file(path)


def _install_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    source = tmp_path / "source.csv.gz"
    source_sha = _write_bytes(
        source, b"not a CSV; preregistration must never parse these bytes\n"
    )
    reference = tmp_path / "reference.csv.gz"
    reference_sha = _write_bytes(reference, b"frozen reference bytes\n")
    manifest_path = tmp_path / "source-manifest.json"
    core: dict[str, Any] = {
        "protocol_version": prereg.SOURCE_PROTOCOL_VERSION,
        "source_decision": {
            "path": str(prereg.SOURCE_ORIGIN_DECISION),
            "sha256": prereg.SOURCE_ORIGIN_DECISION_SHA256,
        },
        "source_builder": {
            "path": str(prereg.SOURCE_BUILDER),
            "sha256": prereg.SOURCE_BUILDER_SHA256,
        },
        "config": {
            "output_csv": str(source),
            "manifest_output": str(manifest_path),
        },
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
            "reference_path": str(reference),
            "reference_sha256": reference_sha,
            "rows_cross_checked": prereg.FROZEN_ROWS,
            "columns_cross_checked": list(prereg.REFERENCE_COLUMNS),
            "all_basic_fields_match_reference": True,
        },
        "output": {
            "path": str(source),
            "sha256": source_sha,
            "bytes": source.stat().st_size,
            "columns": list(prereg.SOURCE_COLUMNS),
        },
        "outcome_boundary": dict(prereg.SOURCE_OUTCOME_BOUNDARY),
        "data_use": prereg.EXPECTED_DATA_USE,
    }
    if mutate is not None:
        mutate(core)
    manifest = {**core, "manifest_hash": prereg.canonical_hash(core)}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(prereg, "SOURCE_MANIFEST", manifest_path)
    monkeypatch.setattr(
        prereg,
        "EXPECTED_SOURCE_MANIFEST_FILE_SHA256",
        prereg.sha256_file(manifest_path),
    )
    monkeypatch.setattr(
        prereg, "EXPECTED_SOURCE_MANIFEST_HASH", manifest["manifest_hash"]
    )
    monkeypatch.setattr(prereg, "EXPECTED_SOURCE_OUTPUT", source)
    monkeypatch.setattr(prereg, "EXPECTED_SOURCE_OUTPUT_SHA256", source_sha)
    monkeypatch.setattr(
        prereg, "EXPECTED_SOURCE_OUTPUT_BYTES", source.stat().st_size
    )
    monkeypatch.setattr(prereg, "REFERENCE_SOURCE", reference)
    monkeypatch.setattr(prereg, "REFERENCE_SOURCE_SHA256", reference_sha)
    return manifest_path, manifest


def _cfg(tmp_path: Path, manifest: Path, **changes: Any) -> prereg.Config:
    cfg = prereg.Config(
        source_manifest=str(manifest),
        preregistration_output=str(tmp_path / "fetd-prereg.json"),
    )
    return replace(cfg, **changes)


def test_writes_exact_outcome_blind_singleton_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest = _install_source(tmp_path, monkeypatch)
    cfg = _cfg(tmp_path, manifest_path)
    artifact = prereg.write_preregistration(cfg)

    assert artifact == json.loads(
        Path(cfg.preregistration_output).read_text(encoding="utf-8")
    )
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    assert artifact["manifest_hash"] == prereg.canonical_hash(core)
    assert artifact["policy_hash"] == prereg.canonical_hash(artifact["policy"])
    assert artifact["policy_id"] == "FETD-288"
    assert artifact["outcomes_opened"] is False
    assert artifact["outcome_boundary"] == {
        "source_manifest_json_read": True,
        "source_artifact_bytes_hashed": True,
        "source_csv_values_read": 0,
        "fetd_feature_rows_derived": 0,
        "signal_incidence_rows_derived": 0,
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "return_or_pnl_fields": 0,
    }
    assert artifact["source_manifest"]["manifest_hash"] == manifest["manifest_hash"]
    assert artifact["mechanism_decision"] == {
        "path": str(prereg.MECHANISM_DECISION),
        "sha256": prereg.MECHANISM_DECISION_SHA256,
    }
    assert artifact["preregistration_document"] == {
        "path": str(prereg.PREREGISTRATION_DOCUMENT),
        "sha256": prereg.PREREGISTRATION_DOCUMENT_SHA256,
    }

    policy = artifact["policy"]
    assert policy["singleton"] is True
    assert policy["source_features"]["packet_blocks"] == 72
    assert policy["source_features"]["packet_alignment"] == (
        "packet_id=floor(height/72)"
    )
    assert policy["source_features"]["complete_packet_count"] == 2_959
    assert policy["source_features"]["fee_pressure"] == (
        "log(total_fees/total_weight)"
    )
    assert policy["source_features"]["endpoint_density"] == (
        "log(total_endpoints/total_weight)"
    )
    assert policy["source_features"]["transport_horizon_packets"] == 2
    assert "utxo_set_change" in policy["source_features"]["forbidden_primary_fields"]
    assert policy["normalization"]["lookback_valid_feature_packets"] == 180
    assert policy["normalization"]["minimum_prior_valid_feature_packets"] == 120
    assert policy["eligibility"]["common"] == (
        "fee_transport*endpoint_transport<0 and strain_rank>=0.75"
    )
    assert policy["eligibility"]["side"] == (
        "-sign(fee_transport), equal to sign(endpoint_transport)"
    )
    assert policy["causal_availability"]["availability_lag_seconds"] == 172_800
    assert policy["causal_availability"]["entry_latency_seconds"] == 300
    assert policy["execution"]["hold_bars"] == 288
    assert policy["execution"]["non_overlap"] is True
    assert policy["calendar"]["train"] == (
        "[2021-01-01T00:00:00Z,2023-01-01T00:00:00Z)"
    )
    assert policy["calendar"]["selection"] == (
        "[2023-01-01T00:00:00Z,2024-01-01T00:00:00Z)"
    )
    assert policy["calendar"]["sealed"] == "2024+"
    assert policy["support_gates"]["train_total_minimum"] == 80
    assert policy["support_gates"]["train_long_minimum"] == 25
    assert policy["support_gates"]["train_short_minimum"] == 25
    assert policy["support_gates"]["selection_total_minimum"] == 35
    assert policy["support_gates"]["selection_each_quarter_minimum"] == 6
    assert policy["support_gates"]["delayed_entry_split_edge_reporting"] == (
        "report train and selection dropped counts before outcomes; dropped "
        "trades receive no replacement"
    )
    assert policy["performance_gates"]["cagr_to_strict_mdd_minimum_each"] == 3.0
    assert policy["performance_gates"]["strict_max_drawdown_maximum_each"] == 0.15
    assert policy["controls"] == prereg.CONTROL_DEFINITIONS
    assert "report train/selection dropped counts" in policy["controls"][
        "one_bar_delayed_entry"
    ]
    assert prereg.load_preregistration(cfg.preregistration_output) == artifact


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["source_decision"].update(sha256="0" * 64),
            "source-origin decision",
        ),
        (
            lambda value: value["source_builder"].update(sha256="0" * 64),
            "source builder",
        ),
        (
            lambda value: value["output"].update(columns=prereg.SOURCE_COLUMNS[:-1]),
            "schema",
        ),
        (
            lambda value: value["outcome_boundary"].update(market_rows_loaded=1),
            "outcome boundary",
        ),
        (
            lambda value: value["source_audit"].update(
                latest_eligible_packet_end=823_721
            ),
            "latest_eligible_packet_end",
        ),
        (
            lambda value: value["source_audit"].update(
                complete_inclusive_height_range=False
            ),
            "complete_inclusive_height_range",
        ),
        (
            lambda value: value["source_audit"].update(utxo_identity_checked=False),
            "utxo_identity_checked",
        ),
        (
            lambda value: value.update(data_use="changed"),
            "data-use",
        ),
    ],
)
def test_rejects_source_semantic_and_boundary_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    manifest_path, _ = _install_source(tmp_path, monkeypatch, mutation)
    with pytest.raises(RuntimeError, match=message):
        prereg.write_preregistration(_cfg(tmp_path, manifest_path))


def test_rejects_manifest_and_source_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest = _install_source(tmp_path, monkeypatch)
    manifest["output"]["bytes"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest.*(SHA|hash)"):
        prereg.write_preregistration(_cfg(tmp_path, manifest_path))

    second = tmp_path / "source-drift"
    manifest_path, manifest = _install_source(second, monkeypatch)
    Path(manifest["output"]["path"]).write_bytes(b"changed after manifest\n")
    with pytest.raises(RuntimeError, match="byte-size|file SHA"):
        prereg.write_preregistration(_cfg(second, manifest_path))


def test_never_decompresses_or_parses_source_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest = _install_source(tmp_path, monkeypatch)
    source_path = Path(manifest["output"]["path"])
    original_open = Path.open

    class _HashOnlyReader:
        def __init__(self, handle: Any) -> None:
            self._handle = handle
            self.read_calls: list[int] = []

        def __enter__(self) -> "_HashOnlyReader":
            self._handle.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self._handle.__exit__(*args)

        def read(self, size: int = -1) -> bytes:
            self.read_calls.append(size)
            assert size == 1024 * 1024
            return self._handle.read(size)

    readers: list[_HashOnlyReader] = []

    def guarded_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = original_open(path, mode, *args, **kwargs)
        if path.resolve() == source_path.resolve():
            assert mode == "rb"
            wrapped = _HashOnlyReader(handle)
            readers.append(wrapped)
            return wrapped
        return handle

    monkeypatch.setattr(Path, "open", guarded_open)
    prereg.write_preregistration(_cfg(tmp_path, manifest_path))
    assert len(readers) == 1
    assert readers[0].read_calls[-1] == 1024 * 1024


def test_rejects_alias_non_json_existing_and_duplicate_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _install_source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="protected"):
        prereg.write_preregistration(
            _cfg(
                tmp_path,
                manifest_path,
                preregistration_output=str(manifest_path),
            )
        )
    with pytest.raises(ValueError, match="JSON"):
        prereg.write_preregistration(
            _cfg(
                tmp_path,
                manifest_path,
                preregistration_output=str(tmp_path / "artifact.txt"),
            )
        )

    cfg = _cfg(tmp_path, manifest_path)
    Path(cfg.preregistration_output).write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        prereg.write_preregistration(cfg)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate key"):
        prereg._read_json(duplicate)


def test_load_rejects_policy_outcome_binding_and_output_path_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _install_source(tmp_path, monkeypatch)
    cfg = _cfg(tmp_path, manifest_path)
    artifact = prereg.write_preregistration(cfg)
    path = Path(cfg.preregistration_output)

    def write_drift(mutator: Callable[[dict[str, Any]], None]) -> None:
        drift = deepcopy(artifact)
        mutator(drift)
        core = {
            key: value for key, value in drift.items() if key != "manifest_hash"
        }
        drift["manifest_hash"] = prereg.canonical_hash(core)
        path.write_text(json.dumps(drift), encoding="utf-8")

    write_drift(lambda value: value["policy"].update(singleton=False))
    with pytest.raises(RuntimeError, match="policy drift"):
        prereg.load_preregistration(path)

    write_drift(lambda value: value.update(outcomes_opened=True))
    with pytest.raises(RuntimeError, match="opened outcomes"):
        prereg.load_preregistration(path)

    write_drift(
        lambda value: value["outcome_boundary"].update(source_csv_values_read=1)
    )
    with pytest.raises(RuntimeError, match="outcome boundary drift"):
        prereg.load_preregistration(path)

    write_drift(
        lambda value: value["source_manifest"].update(manifest_hash="0" * 64)
    )
    with pytest.raises(RuntimeError, match="source-manifest binding drift"):
        prereg.load_preregistration(path)

    write_drift(
        lambda value: value["mechanism_decision"].update(sha256="0" * 64)
    )
    with pytest.raises(RuntimeError, match="mechanism-decision binding drift"):
        prereg.load_preregistration(path)

    write_drift(
        lambda value: value["config"].update(
            preregistration_output=str(tmp_path / "other.json")
        )
    )
    with pytest.raises(RuntimeError, match="output-path binding drift"):
        prereg.load_preregistration(path)
