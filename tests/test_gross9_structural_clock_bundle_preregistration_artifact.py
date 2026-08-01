from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
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
ACTIVE_G9CB12_ARTIFACT = (
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
EXPECTED_G9CB6_PREPUBLICATION_CLOSURE = {
    "authority_decision": {
        "authority_commit": "2695ee61fbb9b5e053dbb9da597ebe2729aad361",
        "git_blob": "eb743d9f8ecd878b83f8f8873697c58cccef9f1b",
        "git_mode": "100644",
        "path": (
            "docs/gross9-structural-clock-bundle-g9cb6-successor-authority-"
            "decision-2026-07-31.md"
        ),
        "path_type": "regular_file",
        "sha256": (
            "b64f9480741eeb4f69ac86736589fbcf8fb75565c436d76316b73f5e076acfca"
        ),
    },
    "classification": (
        "pre_preregistration_publication_bootstrap_manifest_bound_path_set_"
        "mismatch"
    ),
    "failure": {
        "bytes_opened": 105_571_805,
        "exception": "ValueError: manifest bound path set differs from bootstrap",
        "exit_status": 1,
        "manifest_constructed": True,
        "metadata_json_decoded": True,
        "normalized_invocation": (
            "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m "
            "training.preregister_gross9_structural_clock_bundle"
        ),
        "observed_preregistration_invocations": 1,
        "official_production_invocations": None,
        "paths_opened": 58,
        "preregistration_published": False,
        "publication_capability_probe_started": False,
        "runtime_python_ast_parsed": True,
        "snapshot_final_recheck_completed": False,
        "source_model_or_history_values_decoded_or_loaded": False,
        "status": "authorized_first_invocation_closed_identity",
    },
    "identity": "G9CB-6",
    "input_materialization": {
        "authority_order": "after_clean_pushed_A7_before_Q7",
        "destination": {
            "git_blob": None,
            "git_mode": None,
            "mode_octal": "0444",
            "path": (
                "data/cache_market_ext_5m_wavefull_2020-01-01_"
                "2026-06-01_oi.csv.gz"
            ),
            "path_type": "regular_file",
            "sha256": (
                "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192"
            ),
            "size_bytes": 72_898_508,
        },
        "source": {
            "absolute_path": (
                "/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_"
                "2026-06-01_oi.csv.gz"
            ),
            "expected_sha256": (
                "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192"
            ),
            "path_type": "regular_file",
            "size_bytes": 72_898_508,
        },
        "source_values_decoded": False,
        "status": "opaque_byte_identical_symlink_replaced_by_regular_file",
    },
    "permanently_absent_outputs": [
        "results/gross9_structural_clock_bundle_g9cb6_2026-07-31.csv.gz",
        "results/gross9_structural_clock_bundle_g9cb6_access_claim_2026-07-31.json",
        "results/gross9_structural_clock_bundle_g9cb6_attempt_consumed_2026-07-31.json",
        "results/gross9_structural_clock_bundle_g9cb6_manifest_2026-07-31.json",
        "results/gross9_structural_clock_bundle_g9cb6_preregistration_2026-07-31.json",
        (
            "results/gross9_structural_clock_bundle_g9cb6_worker_capability_"
            "consumed_pass1_2026-07-31.json"
        ),
        (
            "results/gross9_structural_clock_bundle_g9cb6_worker_capability_"
            "consumed_pass2_2026-07-31.json"
        ),
    ],
    "protocol_implementation": {
        "builder_git_blob": "09cb9757a230c349cd7b7df9f7ce4a20cfa9b30c",
        "builder_path": "training/build_gross9_structural_clock_bundle.py",
        "builder_sha256": (
            "4fe465368fa074536e85e2e0b54e4ff4800b4cd8a034510015bef78a66d9db93"
        ),
        "commit": "86c7076e415ed667560bfe41c942ab4a00c75a4d",
        "preregistration_git_blob": (
            "af809793347a647632f07ab1d74f5fbeabaac122"
        ),
        "preregistration_path": (
            "training/preregister_gross9_structural_clock_bundle.py"
        ),
        "preregistration_sha256": (
            "5a04a8616a7c8416e67f349f8fd4a846fda87786c0f54fb1415dcf924bb17374"
        ),
    },
    "protocol_version": "gross9_structural_clock_bundle_g9cb6_v1",
    "residue": {
        "bytecode_cache": {
            "path": "results/.g9cb6-bytecode-cache-disabled",
            "state": "absent",
        },
        "capability_probes": {
            "glob": "results/.g9cb6-otmpfile-probe-*",
            "state": "absent",
        },
        "publication_stages": {
            "glob": (
                "results/.gross9_structural_clock_bundle_g9cb6_*.stage-*"
            ),
            "state": "absent",
        },
        "worker_stages": {
            "glob": "results/.gross9-structural-clock-g9cb6-worker-*",
            "state": "absent",
        },
    },
    "root_cause": {
        "bootstrap_bound_path_count": 58,
        "bootstrap_minus_manifest": [],
        "bootstrap_missing_container": "failed_prepublication_closures",
        "manifest_bound_path_count": 59,
        "manifest_minus_bootstrap": [
            {
                "path": (
                    "data/cache_market_ext_5m_wavefull_2020-01-01_"
                    "2026-06-01_oi.csv.gz"
                ),
                "sha256": (
                    "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192"
                ),
                "size_bytes": 72_898_508,
            }
        ],
        "publication_state_validation_started": False,
        "set_comparison_location": (
            "write_once_retained_snapshot_before_results_parent_lookup"
        ),
    },
    "status": (
        "historical_prepublication_closure_no_preregistration_no_attempt_no_"
        "clock_authority"
    ),
    "topology": {
        "g9cb6_authority_commit": "2695ee61fbb9b5e053dbb9da597ebe2729aad361",
        "g9cb6_protocol_commit": "86c7076e415ed667560bfe41c942ab4a00c75a4d",
        "g9cb7_authority_commit": "ad5a7e5f6d3edeac0928c1ef93fd0fd2209a9279",
        "preregistration_commit": None,
        "terminal_evidence_commit": None,
    },
}


def test_failed_g9cb1_artifacts_remain_nonoperative_evidence() -> None:
    historical_v1 = (
        prereg.REPOSITORY_ROOT
        / prereg.HISTORICAL_PREREGISTRATION_PATH
    )
    historical_v2 = (
        prereg.REPOSITORY_ROOT
        / prereg.FAILED_V2_PREREGISTRATION_PATH
    )
    assert historical_v1 != ACTIVE_G9CB12_ARTIFACT
    assert historical_v2 != ACTIVE_G9CB12_ARTIFACT
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


def test_active_g9cb12_preregistration_is_absent_before_p12_without_skip() -> None:
    assert prereg.IDENTITY == "G9CB-12-SOURCE-SUPPORT"
    assert prereg.PREREGISTRATION_PATH.as_posix() == (
        "results/"
        "gross9_structural_clock_bundle_g9cb12_preregistration_2026-07-31.json"
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
            ".gross9-structural-clock-g9cb12-worker-*"
        )
    )
    assert not (
        prereg.REPOSITORY_ROOT
        / "results/.g9cb12-bytecode-cache-disabled"
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


def test_g9cb5_prepublication_closure_and_all_seven_outputs_remain_exact() -> None:
    closure = prereg.expected_failed_predecessor_prepublication_closures()[0]
    assert closure["identity"] == "G9CB-5"
    assert closure["topology"] == {
        "g9cb5_authority_commit": "1ca718d9dab1077b041e753f3b011fbf5b23f047",
        "g9cb5_protocol_commit": "02c3c83a5253684057f44f51ee96bcb089b40b2f",
        "g9cb6_authority_commit": "2695ee61fbb9b5e053dbb9da597ebe2729aad361",
        "preregistration_commit": None,
        "terminal_evidence_commit": None,
    }
    assert len(closure["permanently_absent_outputs"]) == 7
    assert all(
        not (prereg.REPOSITORY_ROOT / path).exists()
        for path in closure["permanently_absent_outputs"]
    )
    assert not (
        prereg.REPOSITORY_ROOT / closure["residue"]["bytecode_cache"]["path"]
    ).exists()


def test_g9cb6_prepublication_closure_and_materialization_are_exact() -> None:
    closure = prereg.expected_failed_predecessor_prepublication_closures()[1]
    assert closure == EXPECTED_G9CB6_PREPUBLICATION_CLOSURE
    destination = closure["input_materialization"]["destination"]
    source_binding = closure["input_materialization"]["source"]
    materialized = prereg.REPOSITORY_ROOT / destination["path"]
    source = Path(source_binding["absolute_path"])
    info = materialized.stat(follow_symlinks=False)
    source_info = source.stat(follow_symlinks=False)
    assert stat.S_ISREG(info.st_mode) and not materialized.is_symlink()
    assert stat.S_ISREG(source_info.st_mode) and not source.is_symlink()
    assert stat.S_IMODE(info.st_mode) == 0o444
    assert info.st_nlink == 1
    assert info.st_size == destination["size_bytes"]
    assert source_info.st_size == source_binding["size_bytes"]
    assert (info.st_dev, info.st_ino) != (
        source_info.st_dev,
        source_info.st_ino,
    )
    assert hashlib.sha256(materialized.read_bytes()).hexdigest() == destination[
        "sha256"
    ]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_binding[
        "expected_sha256"
    ]
    assert all(
        not (prereg.REPOSITORY_ROOT / path).exists()
        for path in closure["permanently_absent_outputs"]
    )
    for residue in closure["residue"].values():
        if "path" in residue:
            assert not (prereg.REPOSITORY_ROOT / residue["path"]).exists()
        else:
            assert not list(prereg.REPOSITORY_ROOT.glob(residue["glob"]))


def test_g9cb7_pre_sentinel_closure_preserves_p7_c7_and_absent_outputs() -> None:
    [closure] = prereg.expected_failed_predecessor_pre_sentinel_closures()
    assert closure["identity"] == "G9CB-7"
    assert closure["preregistration"]["seal_commit"] == (
        "ededa5df4c5b5b91588765995ed7b1c502332925"
    )
    assert closure["access_claim"]["seal_commit"] == (
        "ff1a8907d19c97beeef0bd7d2797e3bacce17617"
    )
    assert closure["failure"]["observed_production_invocations"] == 1
    assert closure["failure"]["canonical_wrapper_invocations"] == 0
    assert closure["failure"]["publication_context_constructed"] is False
    assert len(closure["permanently_absent_outputs"]) == 5
    assert all(
        not (prereg.REPOSITORY_ROOT / path).exists()
        for path in closure["permanently_absent_outputs"]
    )
    assert (
        prereg.REPOSITORY_ROOT / closure["preregistration"]["path"]
    ).is_file()
    assert (
        prereg.REPOSITORY_ROOT / closure["access_claim"]["path"]
    ).is_file()
    assert not list(prereg.REPOSITORY_ROOT.rglob("__pycache__"))
    assert not list(prereg.REPOSITORY_ROOT.rglob("*.pyc"))
    assert not list(prereg.REPOSITORY_ROOT.rglob("*.pyo"))


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


def _q13_handoff_fixture() -> tuple[list[dict[str, object]], bytes]:
    commits = {
        stage: character * 40
        for stage, character in zip(
            ("A13", "T12", "Q13", "P13", "C13", "D13"),
            "abcdef",
            strict=True,
        )
    }
    bindings: list[dict[str, object]] = []
    parent = "0" * 40
    for stage, paths in builder.H13_SUCCESSOR_STAGE_PATHS.items():
        bindings.append(
            {
                "commit": commits[stage],
                "parent_commit": parent,
                "stage": stage,
                "tracked_files": [
                    {
                        "git_blob": "1" * 40,
                        "git_mode": "100644",
                        "path": path,
                        "sha256": "2" * 64,
                        "size_bytes": 1,
                        "worktree_mode": (
                            "0644" if stage in {"A13", "Q13"} else "0444"
                        ),
                    }
                    for path in paths
                ],
            }
        )
        parent = commits[stage]
    stdout = builder._canonical_h12_json_bytes(
        {
            "claim_commit": commits["C13"],
            "claim_hash": "3" * 64,
            "csv_gzip_sha256": "4" * 64,
            "final_manifest_hash": "5" * 64,
            "head": commits["D13"],
            "identity": builder.G9CB13_IDENTITY,
            "interval_count": 0,
            "preregistration_manifest_hash": "6" * 64,
            "preregistration_seal_commit": commits["P13"],
            "protocol_implementation_commit": commits["Q13"],
            "protocol_version": builder.G9CB13_PROTOCOL_VERSION,
            "publication_commit": commits["D13"],
            "sentinel_manifest_hash": "7" * 64,
        }
    )
    return bindings, stdout


def test_q13_h13_schema_and_supervisor_leakage_exact() -> None:
    supervisor = builder._h13_supervisor_payload(
        "f" * 40, "e" * 40, "d" * 64, 1234
    )
    assert tuple(supervisor) == builder.H13_SUPERVISOR_KEYS
    assert builder.validate_g9cb13_h13_supervisor(supervisor) == supervisor
    bindings, stdout = _q13_handoff_fixture()
    handoff = builder._h13_handoff_payload(bindings, stdout)
    assert tuple(handoff) == builder.H13_TOP_LEVEL_KEYS
    assert builder.validate_g9cb13_h13_handoff(
        handoff, v13_stdout=stdout, successor_bindings=bindings
    ) == handoff
    leaked = json.loads(json.dumps(handoff))
    leaked["successor_bindings"][0]["tracked_files"][0]["path"] = (
        builder.H13_SUPERVISOR_SENTINEL_PATH.as_posix()
    )
    with pytest.raises(builder.TerminalG9CB12Failure):
        builder.validate_g9cb13_h13_handoff(
            leaked, v13_stdout=stdout, successor_bindings=bindings
        )


def test_q13_preregistration_artifact_exact() -> None:
    payload = prereg.build_g9cb13_preregistration_payload(
        protocol_implementation_commit="a" * 40
    )
    prereg.validate_g9cb13_preregistration_payload(payload)
    raw = prereg._g9cb13_canonical_bytes(payload)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert json.loads(raw)["manifest_hash"] == payload["manifest_hash"]
    stdout = (
        f"created {prereg.G9CB13_PREREGISTRATION_PATH} "
        f"{payload['manifest_hash']}\n"
    )
    assert stdout.count("\n") == 1
    assert stdout.startswith(
        "created results/gross9_structural_clock_bundle_g9cb13_"
        "preregistration_2026-08-01.json "
    )


def test_q13_h13_verified_handoff_not_approval_and_no_economics_exact() -> None:
    bindings, stdout = _q13_handoff_fixture()
    handoff = builder._h13_handoff_payload(bindings, stdout)
    assert handoff["next_workflow"] == "ralplan"
    assert handoff["active_alpha_goal"] == "incomplete"
    assert handoff["no_economics"] is True
    assert handoff["no_future_commit_prediction"] is True
    assert handoff["v13_stdout_hash"] == hashlib.sha256(stdout).hexdigest()
    assert builder.H13_STATE_TRACE == (
        "PRE_SUPERVISOR", "SUPERVISOR_LINKED", "V13_VERIFIED", "HANDOFF_LINKED"
    )
    for key in ("candidate", "comparator", "economic_result", "economic-result"):
        mutation = json.loads(json.dumps(handoff))
        mutation["successor_bindings"][0][key] = {}
        with pytest.raises(builder.TerminalG9CB12Failure):
            builder.validate_g9cb13_h13_handoff(
                mutation, v13_stdout=stdout, successor_bindings=bindings
            )
    assert "approval" not in json.dumps(handoff).lower()
