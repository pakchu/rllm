from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from training import preregister_blockspace_load_settlement_relay as prereg


def _write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return prereg.sha256_file(path)


def _install_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, dict[str, Any], Path]:
    contract = prereg.source_contract
    source = tmp_path / "source.csv.gz"
    source_sha = _write_bytes(
        source, b"not a CSV; BLSR preregistration must never parse these bytes\n"
    )
    reference = tmp_path / "reference.csv.gz"
    reference_sha = _write_bytes(reference, b"frozen reference bytes\n")
    manifest_path = tmp_path / "source-manifest.json"
    core: dict[str, Any] = {
        "protocol_version": contract.SOURCE_PROTOCOL_VERSION,
        "source_decision": {
            "path": str(contract.SOURCE_ORIGIN_DECISION),
            "sha256": contract.SOURCE_ORIGIN_DECISION_SHA256,
        },
        "source_builder": {
            "path": str(contract.SOURCE_BUILDER),
            "sha256": contract.SOURCE_BUILDER_SHA256,
        },
        "config": {
            "output_csv": str(source),
            "manifest_output": str(manifest_path),
        },
        "source_audit": {
            "expected_rows": contract.FROZEN_ROWS,
            "observed_rows": contract.FROZEN_ROWS,
            "start_height": contract.FROZEN_START_HEIGHT,
            "end_height": contract.FROZEN_END_HEIGHT,
            "latest_eligible_packet_end": contract.FROZEN_END_HEIGHT - 6,
            "height_links_checked": contract.FROZEN_ROWS - 1,
            "end_timestamp_exclusive": contract.FROZEN_END_TIMESTAMP_EXCLUSIVE,
            "complete_inclusive_height_range": True,
            "unique_block_hashes": True,
            "all_rows_pre_cutoff": True,
            "utxo_identity_checked": True,
        },
        "reference_audit": {
            "reference_path": str(reference),
            "reference_sha256": reference_sha,
            "rows_cross_checked": contract.FROZEN_ROWS,
            "columns_cross_checked": list(contract.REFERENCE_COLUMNS),
            "all_basic_fields_match_reference": True,
        },
        "output": {
            "path": str(source),
            "sha256": source_sha,
            "bytes": source.stat().st_size,
            "columns": list(contract.SOURCE_COLUMNS),
        },
        "outcome_boundary": dict(contract.SOURCE_OUTCOME_BOUNDARY),
        "data_use": contract.EXPECTED_DATA_USE,
    }
    if mutate is not None:
        mutate(core)
    manifest = {**core, "manifest_hash": contract.canonical_hash(core)}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(prereg, "SOURCE_MANIFEST", manifest_path)
    monkeypatch.setattr(contract, "SOURCE_MANIFEST", manifest_path)
    monkeypatch.setattr(
        contract,
        "EXPECTED_SOURCE_MANIFEST_FILE_SHA256",
        contract.sha256_file(manifest_path),
    )
    monkeypatch.setattr(
        contract, "EXPECTED_SOURCE_MANIFEST_HASH", manifest["manifest_hash"]
    )
    monkeypatch.setattr(contract, "EXPECTED_SOURCE_OUTPUT", source)
    monkeypatch.setattr(contract, "EXPECTED_SOURCE_OUTPUT_SHA256", source_sha)
    monkeypatch.setattr(contract, "EXPECTED_SOURCE_OUTPUT_BYTES", source.stat().st_size)
    monkeypatch.setattr(contract, "REFERENCE_SOURCE", reference)
    monkeypatch.setattr(contract, "REFERENCE_SOURCE_SHA256", reference_sha)
    return manifest_path, manifest, source


def _install_comparators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, dict[str, str]], list[Path]]:
    bindings: dict[str, dict[str, str]] = {}
    paths: list[Path] = []
    for name, original in prereg.COMPARATOR_BINDINGS.items():
        path = tmp_path / "comparators" / f"{name}.bin"
        sha = _write_bytes(path, f"frozen comparator {name}\n".encode())
        bindings[name] = {
            "path": str(path),
            "sha256": sha,
            "role": original["role"],
        }
        paths.append(path)
    monkeypatch.setattr(prereg, "COMPARATOR_BINDINGS", bindings)
    return bindings, paths


def _cfg(tmp_path: Path, manifest: Path, **changes: Any) -> prereg.Config:
    cfg = prereg.Config(
        source_manifest=str(manifest),
        preregistration_output=str(tmp_path / "blsr-prereg.json"),
    )
    return replace(cfg, **changes)


def test_writes_exact_outcome_blind_singleton_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest, _ = _install_source(tmp_path, monkeypatch)
    comparators, _ = _install_comparators(tmp_path, monkeypatch)
    cfg = _cfg(tmp_path, manifest_path)
    artifact = prereg.write_preregistration(cfg)

    assert artifact == json.loads(
        Path(cfg.preregistration_output).read_text(encoding="utf-8")
    )
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    assert artifact["manifest_hash"] == prereg.canonical_hash(core)
    assert artifact["policy_hash"] == prereg.canonical_hash(artifact["policy"])
    assert artifact["policy_id"] == "BLSR-288"
    assert artifact["outcomes_opened"] is False
    assert artifact["outcome_boundary"] == prereg.OUTCOME_BOUNDARY
    assert artifact["outcome_boundary"]["source_csv_values_read"] == 0
    assert artifact["outcome_boundary"]["comparator_rows_read"] == 0
    assert artifact["source_manifest"]["manifest_hash"] == manifest["manifest_hash"]
    assert artifact["comparator_bindings"] == comparators

    policy = artifact["policy"]
    assert policy["singleton"] is True
    assert policy["source_features"]["packet_blocks"] == 72
    assert policy["source_features"]["complete_packet_count"] == 2_959
    assert policy["source_features"]["fee_change"] == (
        "fee_pressure[t]-fee_pressure[t-1]"
    )
    assert policy["normalization"]["significance_boundary"] == 0.75
    assert policy["normalization"]["parameter_grid"] == []
    assert policy["relay"]["deadline_packets_after_onset"] == 3
    assert policy["relay"]["no_retry"] is True
    assert policy["relay"]["side"] == ("load_sign; positive LONG, negative SHORT")
    assert policy["causal_availability"]["availability_lag_seconds"] == 172_800
    assert policy["causal_availability"]["entry_latency_seconds"] == 300
    assert policy["execution"]["hold_bars"] == 288
    assert policy["calendar"]["sealed"] == "2024+"
    assert policy["support_gates"]["train_total_minimum"] == 80
    assert policy["support_gates"]["selection_total_minimum"] == 35
    assert policy["novelty_gates"]["exact_entry_timestamp_jaccard_maximum"] == 0.20
    assert (
        policy["novelty_gates"][
            "candidate_one_to_one_within_six_hours_fraction_maximum"
        ]
        == 0.35
    )
    assert policy["performance_gates"]["cagr_to_strict_mdd_minimum_each"] == 3.0
    assert policy["performance_gates"]["strict_max_drawdown_maximum_each"] == 0.15
    assert policy["controls"] == prereg.CONTROL_DEFINITIONS
    assert prereg.load_preregistration(cfg.preregistration_output) == artifact


def test_never_decompresses_or_parses_source_or_comparator_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _, source = _install_source(tmp_path, monkeypatch)
    _, comparator_paths = _install_comparators(tmp_path, monkeypatch)
    guarded = {source.resolve(), *(path.resolve() for path in comparator_paths)}
    original_open = Path.open
    reads: dict[Path, list[int]] = {path: [] for path in guarded}

    class _HashOnlyReader:
        def __init__(self, path: Path, handle: Any) -> None:
            self._path = path
            self._handle = handle

        def __enter__(self) -> "_HashOnlyReader":
            self._handle.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self._handle.__exit__(*args)

        def read(self, size: int = -1) -> bytes:
            assert size == 1024 * 1024
            reads[self._path].append(size)
            return self._handle.read(size)

    def guarded_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        resolved = path.resolve()
        handle = original_open(path, mode, *args, **kwargs)
        if resolved in guarded:
            assert mode == "rb"
            return _HashOnlyReader(resolved, handle)
        return handle

    monkeypatch.setattr(Path, "open", guarded_open)
    prereg.write_preregistration(_cfg(tmp_path, manifest_path))
    assert all(calls and calls[-1] == 1024 * 1024 for calls in reads.values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["output"].update(
                columns=prereg.source_contract.SOURCE_COLUMNS[:-1]
            ),
            "schema",
        ),
        (
            lambda value: value["outcome_boundary"].update(market_rows_loaded=1),
            "outcome boundary",
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
    ],
)
def test_rejects_source_semantic_and_boundary_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    manifest_path, _, _ = _install_source(tmp_path, monkeypatch, mutation)
    _install_comparators(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match=message):
        prereg.write_preregistration(_cfg(tmp_path, manifest_path))


def test_rejects_source_and_comparator_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest, source = _install_source(tmp_path, monkeypatch)
    _, comparator_paths = _install_comparators(tmp_path, monkeypatch)
    source.write_bytes(b"changed after source manifest\n")
    with pytest.raises(RuntimeError, match="byte-size|file SHA"):
        prereg.write_preregistration(_cfg(tmp_path, manifest_path))

    second = tmp_path / "second"
    manifest_path, _, _ = _install_source(second, monkeypatch)
    _, second_comparator_paths = _install_comparators(second, monkeypatch)
    second_comparator_paths[0].write_bytes(b"changed comparator\n")
    with pytest.raises(RuntimeError, match="comparator.*SHA drift"):
        prereg.write_preregistration(_cfg(second, manifest_path))

    assert manifest["output"]["sha256"] != prereg.sha256_file(source)


def test_rejects_alias_non_json_existing_and_duplicate_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _, _ = _install_source(tmp_path, monkeypatch)
    _install_comparators(tmp_path, monkeypatch)
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


def test_load_rejects_policy_outcome_comparator_and_output_path_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _, _ = _install_source(tmp_path, monkeypatch)
    _install_comparators(tmp_path, monkeypatch)
    cfg = _cfg(tmp_path, manifest_path)
    artifact = prereg.write_preregistration(cfg)
    path = Path(cfg.preregistration_output)

    def write_drift(mutator: Callable[[dict[str, Any]], None]) -> None:
        drift = deepcopy(artifact)
        mutator(drift)
        core = {key: value for key, value in drift.items() if key != "manifest_hash"}
        drift["manifest_hash"] = prereg.canonical_hash(core)
        path.write_text(json.dumps(drift), encoding="utf-8")

    write_drift(lambda value: value["policy"].update(singleton=False))
    with pytest.raises(RuntimeError, match="policy drift"):
        prereg.load_preregistration(path)

    write_drift(lambda value: value.update(outcomes_opened=True))
    with pytest.raises(RuntimeError, match="opened outcomes"):
        prereg.load_preregistration(path)

    write_drift(
        lambda value: value["comparator_bindings"].pop("bate_288_primary_clock")
    )
    with pytest.raises(RuntimeError, match="comparator binding drift"):
        prereg.load_preregistration(path)

    other = tmp_path / "other.json"
    other.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(RuntimeError, match="output-path binding drift"):
        prereg.load_preregistration(other)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update(
                hidden_outcome_rows=[{"return": 0.123, "pnl": 456.0}]
            ),
            "top-level schema drift",
        ),
        (
            lambda value: value.pop("research_sequence"),
            "top-level schema drift",
        ),
        (
            lambda value: value.update(policy_id="NOT-BLSR-288"),
            "policy ID drift",
        ),
        (
            lambda value: value["research_sequence"].update(
                outcomes="open every window together"
            ),
            "research sequence drift",
        ),
        (
            lambda value: value["config"].pop("source_manifest"),
            "config drift",
        ),
    ],
)
def test_load_rejects_rehashed_schema_and_sequence_smuggling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    manifest_path, _, _ = _install_source(tmp_path, monkeypatch)
    _install_comparators(tmp_path, monkeypatch)
    cfg = _cfg(tmp_path, manifest_path)
    artifact = prereg.write_preregistration(cfg)
    path = Path(cfg.preregistration_output)

    mutation(artifact)
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    artifact["manifest_hash"] = prereg.canonical_hash(core)
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        prereg.load_preregistration(path)
