from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from training import build_gross9_structural_clock_bundle as builder
from training import preregister_gross9_structural_clock_bundle as prereg


ARTIFACT = prereg.REPOSITORY_ROOT / prereg.PREREGISTRATION_PATH
EXPECTED_AUTHORITY_AMENDMENT_IDENTITIES = [
    "G9CB-1A",
    "G9CB-1B",
    "G9CB-1C",
]
EXPECTED_RUNTIME_IMPORT_ROOTS = [
    "execution/gross9_rank7_clock_runtime.py",
    "training/gross9_structural_clock_primitives.py",
]
EXPECTED_CONSUMPTION_LEDGER_PATHS = [
    (
        "results/"
        "gross9_structural_clock_bundle_g9cb4_worker_capability_consumed_pass1_"
        "2026-07-31.json"
    ),
    (
        "results/"
        "gross9_structural_clock_bundle_g9cb4_worker_capability_consumed_pass2_"
        "2026-07-31.json"
    ),
]


def test_failed_g9cb1_artifacts_remain_nonoperative_evidence() -> None:
    historical_v1 = (
        prereg.REPOSITORY_ROOT
        / prereg.HISTORICAL_PREREGISTRATION_PATH
    )
    historical_v2 = (
        prereg.REPOSITORY_ROOT
        / prereg.FAILED_V2_PREREGISTRATION_PATH
    )
    assert historical_v1 != ARTIFACT
    assert historical_v2 != ARTIFACT
    assert prereg.validate_failed_predecessor_preregistrations() == (
        prereg.expected_failed_predecessor_preregistration_bindings()
    )
    payload_v1 = json.loads(historical_v1.read_bytes())
    payload_v2 = json.loads(historical_v2.read_bytes())
    assert payload_v1["protocol_version"] == prereg.HISTORICAL_PROTOCOL_VERSION
    assert payload_v2["protocol_version"] == prereg.FAILED_V2_PROTOCOL_VERSION
    assert payload_v1["protocol_version"] != prereg.PROTOCOL_VERSION
    assert payload_v2["protocol_version"] != prereg.PROTOCOL_VERSION
    assert [
        row["identity"]
        for row in payload_v1["bindings"]["authority_amendments"]
    ] == ["G9CB-1A", "G9CB-1B"]
    assert [
        row["identity"]
        for row in payload_v2["bindings"]["authority_amendments"]
    ] == ["G9CB-1A", "G9CB-1B", "G9CB-1C"]


def test_failed_g9cb2_and_g9cb3_terminal_evidence_and_residue_remain_exact() -> None:
    attempts = prereg.validate_failed_predecessor_attempts()
    assert attempts == prereg.expected_failed_predecessor_attempts()
    assert [row["identity"] for row in attempts] == ["G9CB-2", "G9CB-3"]
    for row in attempts:
        for path_text in row["permanently_absent_outputs"]:
            assert not (prereg.REPOSITORY_ROOT / path_text).exists()
        slot1 = prereg.REPOSITORY_ROOT / row["residue"]["slot1_stage"]["path"]
        slot2 = prereg.REPOSITORY_ROOT / row["residue"]["slot2_stage"]["path"]
        assert slot1.is_dir()
        assert list(slot1.iterdir()) == []
        assert slot1.stat().st_mode & 0o777 == 0o700
        assert not slot2.exists()


def test_committed_preregistration_artifact_is_exactly_reproducible() -> None:
    if not ARTIFACT.exists():
        committed = subprocess.run(
            [
                "git",
                "cat-file",
                "-e",
                f"HEAD:{prereg.PREREGISTRATION_PATH.as_posix()}",
            ],
            cwd=prereg.REPOSITORY_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert committed.returncode != 0, "committed canonical artifact is missing"
        pytest.skip("canonical artifact is absent before its generation commit")

    raw = ARTIFACT.read_bytes()
    payload = json.loads(raw)
    assert raw == prereg.canonical_json_bytes(payload, trailing_lf=True)
    prereg.validate_manifest(payload)
    assert payload == prereg.build_manifest()
    assert payload["protocol_version"] == (
        "gross9_structural_clock_bundle_g9cb4_preregistration_v1"
    )
    assert payload["protocol_implementation_commit"] == (
        prereg.validate_protocol_commit_topology()
    )
    assert payload["bindings"]["failed_predecessor_preregistrations"] == (
        prereg.expected_failed_predecessor_preregistration_bindings()
    )
    assert payload["bindings"]["authority_amendments"] == (
        prereg._authority_amendment_bindings()
    )
    assert [
        row["identity"] for row in payload["bindings"]["authority_amendments"]
    ] == EXPECTED_AUTHORITY_AMENDMENT_IDENTITIES
    assert payload["bindings"]["runtime_import_roots"] == (
        EXPECTED_RUNTIME_IMPORT_ROOTS
    )
    assert "adapter_import_roots" not in payload["bindings"]
    assert "adapter_import_closure" not in payload["bindings"]
    assert payload["output_paths"]["worker_capability_consumption_ledgers"] == (
        EXPECTED_CONSUMPTION_LEDGER_PATHS
    )
    worker_environment = payload["bindings"]["environment"][
        "worker_process_environment"
    ]
    assert len(worker_environment) == 18
    assert worker_environment == prereg.worker_process_environment()
    assert payload["access_counter_names"]["rows_used"][-10:] == [
        "rank7_training_trades_replayed",
        "rank7_net_labels_computed",
        "rank7_adverse_labels_computed",
        "rank7_price_factor_values_used",
        "rank7_funding_factor_values_used",
        "rank7_funding_debit_factor_values_used",
        "rank7_adverse_price_factor_values_used",
        "rank7_fee_factor_values_used",
        "rank7_bundle_activation_rows_scored",
        "rank7_bundle_parity_rows_compared",
    ]

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(prereg.PREREGISTRATION_PATH)],
        cwd=prereg.REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert tracked.returncode == 0, "present canonical artifact must be committed"
    blob = subprocess.run(
        ["git", "show", f"HEAD:{prereg.PREREGISTRATION_PATH.as_posix()}"],
        cwd=prereg.REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    assert blob == raw


def test_committed_publication_is_verified_at_d_when_present() -> None:
    manifest = prereg.REPOSITORY_ROOT / builder.MANIFEST_PATH
    if not manifest.exists():
        pytest.skip("publication artifact is absent before D")
    result = builder.validate_committed_publication()
    assert result["head"] == result["publication_commit"]
    assert result["protocol_implementation_commit"] == (
        prereg.validate_protocol_commit_topology()
    )
