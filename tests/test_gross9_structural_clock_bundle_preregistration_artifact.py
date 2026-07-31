from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from training import build_gross9_structural_clock_bundle as builder
from training import preregister_gross9_structural_clock_bundle as prereg


FROZEN_G9CB4_ARTIFACT_PATH = Path(
    "results/"
    "gross9_structural_clock_bundle_g9cb4_preregistration_2026-07-31.json"
)
FROZEN_G9CB4_ARTIFACT = (
    prereg.REPOSITORY_ROOT / FROZEN_G9CB4_ARTIFACT_PATH
)
ACTIVE_G9CB5_ARTIFACT = (
    prereg.REPOSITORY_ROOT / prereg.PREREGISTRATION_PATH
)
FROZEN_G9CB4_SHA256 = (
    "f65aaf5fd2219f90421912e6fc9065ddffb54f5adf881196986f25185fe7342e"
)
FROZEN_G9CB4_GIT_BLOB = "76f9011d5752282c058feb531442b203a0bbdb0d"
FROZEN_G9CB4_MANIFEST_HASH = (
    "fa3dab6f7e6ab86428c03fc5c3d7b005e0a165cd76662bba9a7c3cd5941beeed"
)
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
    assert historical_v1 != ACTIVE_G9CB5_ARTIFACT
    assert historical_v2 != ACTIVE_G9CB5_ARTIFACT
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
    assert [row["permanently_absent_outputs"] for row in attempts] == [
        [
            "results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass1_2026-07-31.json",
            "results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass2_2026-07-31.json",
            "results/gross9_structural_clock_bundle_g9cb2_2026-07-31.csv.gz",
            "results/gross9_structural_clock_bundle_g9cb2_manifest_2026-07-31.json",
        ],
        [
            "results/gross9_structural_clock_bundle_g9cb3_worker_capability_consumed_pass2_2026-07-31.json",
            "results/gross9_structural_clock_bundle_g9cb3_2026-07-31.csv.gz",
            "results/gross9_structural_clock_bundle_g9cb3_manifest_2026-07-31.json",
        ],
    ]
    assert attempts[0]["residue"] == {
        "slot1_stage": {
            "committed": False,
            "filesystem_mode_octal": "0700",
            "path": "results/.gross9-structural-clock-worker-ca9ca670ffb0d1b377ed6aef",
            "state": "empty_directory",
        },
        "slot2_stage": {
            "committed": False,
            "path": "results/.gross9-structural-clock-worker-2c9f266762f8864bf5e24691",
            "state": "absent",
        },
    }
    assert attempts[1]["residue"] == {
        "bytecode_cache": {
            "path": "results/.g9cb3-bytecode-cache-disabled",
            "state": "absent",
        },
        "slot1_stage": {
            "committed": False,
            "filesystem_mode_octal": "0700",
            "path": "results/.gross9-structural-clock-g9cb3-worker-a3dffd3cbec3afd582638a23",
            "staged_core_state": "absent",
            "staged_csv_state": "absent",
            "staged_receipt_state": "absent",
            "state": "empty_directory",
        },
        "slot2_stage": {
            "committed": False,
            "path": "results/.gross9-structural-clock-g9cb3-worker-26e64bf0a62646afad3d77e6",
            "state": "absent",
        },
    }
    for row in attempts:
        for path_text in row["permanently_absent_outputs"]:
            assert not (prereg.REPOSITORY_ROOT / path_text).exists()
        slot1 = prereg.REPOSITORY_ROOT / row["residue"]["slot1_stage"]["path"]
        slot2 = prereg.REPOSITORY_ROOT / row["residue"]["slot2_stage"]["path"]
        assert slot1.is_dir()
        assert list(slot1.iterdir()) == []
        assert slot1.stat().st_mode & 0o777 == 0o700
        assert not slot2.exists()


def test_frozen_g9cb4_preregistration_authenticates_as_opaque_metadata() -> None:
    raw = FROZEN_G9CB4_ARTIFACT.read_bytes()
    payload = json.loads(raw)
    assert raw == prereg.canonical_json_bytes(payload, trailing_lf=True)
    assert payload["protocol_version"] == (
        "gross9_structural_clock_bundle_g9cb4_preregistration_v1"
    )
    assert payload["protocol_implementation_commit"] == (
        "750c837a10c4d4ac39fbc8f6097465c82b6dc3ec"
    )
    assert payload["manifest_hash"] == FROZEN_G9CB4_MANIFEST_HASH
    assert [
        row["identity"] for row in payload["bindings"]["authority_amendments"]
    ] == EXPECTED_AUTHORITY_AMENDMENT_IDENTITIES
    assert payload["bindings"]["runtime_import_roots"] == (
        EXPECTED_RUNTIME_IMPORT_ROOTS
    )
    assert "adapter_import_roots" not in payload["bindings"]
    assert "adapter_import_closure" not in payload["bindings"]
    assert payload["output_paths"][
        "worker_capability_consumption_ledgers"
    ] == EXPECTED_CONSUMPTION_LEDGER_PATHS
    worker_environment = payload["bindings"]["environment"][
        "worker_process_environment"
    ]
    assert len(worker_environment) == 18
    current_environment = prereg.worker_process_environment()
    assert {
        key: value
        for key, value in worker_environment.items()
        if key != "PYTHONPYCACHEPREFIX"
    } == {
        key: value
        for key, value in current_environment.items()
        if key != "PYTHONPYCACHEPREFIX"
    }
    assert worker_environment["PYTHONPYCACHEPREFIX"].endswith(
        "/results/.g9cb4-bytecode-cache-disabled"
    )
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

    digest = __import__("hashlib").sha256(raw).hexdigest()
    assert digest == FROZEN_G9CB4_SHA256
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            FROZEN_G9CB4_ARTIFACT_PATH.as_posix(),
        ],
        cwd=prereg.REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert tracked.returncode == 0, "present canonical artifact must be committed"
    blob = subprocess.run(
        ["git", "show", f"HEAD:{FROZEN_G9CB4_ARTIFACT_PATH.as_posix()}"],
        cwd=prereg.REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    assert blob == raw
    blob_id = subprocess.run(
        [
            "git",
            "rev-parse",
            f"HEAD:{FROZEN_G9CB4_ARTIFACT_PATH.as_posix()}",
        ],
        cwd=prereg.REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    assert blob_id == FROZEN_G9CB4_GIT_BLOB


def test_active_g9cb5_preregistration_is_absent_before_p5_without_skip() -> None:
    assert prereg.IDENTITY == "G9CB-5"
    assert prereg.PREREGISTRATION_PATH.as_posix() == (
        "results/"
        "gross9_structural_clock_bundle_g9cb5_preregistration_2026-07-31.json"
    )
    active_paths = (
        prereg.PREREGISTRATION_PATH,
        prereg.ACCESS_CLAIM_PATH,
        prereg.ATTEMPT_SENTINEL_PATH,
        *prereg.WORKER_CAPABILITY_CONSUMPTION_LEDGER_PATHS,
        prereg.BUNDLE_PATH,
        prereg.FINAL_MANIFEST_PATH,
    )
    assert all(
        not (prereg.REPOSITORY_ROOT / path).exists()
        for path in active_paths
    )
    assert not list(
        (prereg.REPOSITORY_ROOT / "results").glob(
            ".gross9-structural-clock-g9cb5-worker-*"
        )
    )
    assert not (
        prereg.REPOSITORY_ROOT
        / "results/.g9cb5-bytecode-cache-disabled"
    ).exists()
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
    assert committed.returncode != 0


def test_frozen_g9cb4_absence_and_residue_inventory_is_complete() -> None:
    [closure] = prereg.expected_failed_predecessor_closures()
    assert closure["permanently_absent_outputs"] == sorted(
        [
            "results/gross9_structural_clock_bundle_g9cb4_2026-07-31.csv.gz",
            "results/gross9_structural_clock_bundle_g9cb4_access_claim_2026-07-31.json",
            "results/gross9_structural_clock_bundle_g9cb4_attempt_consumed_2026-07-31.json",
            "results/gross9_structural_clock_bundle_g9cb4_manifest_2026-07-31.json",
            "results/gross9_structural_clock_bundle_g9cb4_worker_capability_consumed_pass1_2026-07-31.json",
            "results/gross9_structural_clock_bundle_g9cb4_worker_capability_consumed_pass2_2026-07-31.json",
        ]
    )
    assert all(
        not (prereg.REPOSITORY_ROOT / path).exists()
        for path in closure["permanently_absent_outputs"]
    )
    assert not (
        prereg.REPOSITORY_ROOT
        / closure["residue"]["bytecode_cache"]["path"]
    ).exists()
    assert not list(
        prereg.REPOSITORY_ROOT.glob(
            closure["residue"]["publication_stages"]["glob"]
        )
    )
    assert not list(
        prereg.REPOSITORY_ROOT.glob(
            closure["residue"]["worker_stages"]["glob"]
        )
    )


def test_committed_publication_is_verified_at_d_when_present() -> None:
    manifest = prereg.REPOSITORY_ROOT / builder.MANIFEST_PATH
    if not manifest.exists():
        pytest.skip("publication artifact is absent before D")
    result = builder.validate_committed_publication()
    assert result["head"] == result["publication_commit"]
    assert result["protocol_implementation_commit"] == (
        prereg.validate_protocol_commit_topology()
    )
