from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import time
from typing import Any

import pytest

from training import preregister_gross9_structural_clock_bundle as prereg


EXPECTED_AUTHORITY_AMENDMENTS = [
    {
        "identity": "G9CB-1A",
        "path": (
            "docs/"
            "gross9-structural-clock-bundle-rank7-authority-amendment-"
            "2026-07-31.md"
        ),
        "path_type": "regular_file",
        "sha256": (
            "a99b1a2b3d738ecc1cea8595eed2d88759c9b5fa7faf751a53b643fcc1a808cb"
        ),
        "git_blob": "0c7781ebe25178c592bb526ac51ee00c5ba840e2",
        "git_mode": "100644",
        "authority_commit": "f1ae4e68bfb0d0b861cd9979762f87e51a55f69d",
    },
    {
        "identity": "G9CB-1B",
        "path": (
            "docs/"
            "gross9-structural-clock-bundle-runtime-isolation-amendment-"
            "2026-07-31.md"
        ),
        "path_type": "regular_file",
        "sha256": (
            "354ae3870dd6dedf738b38bdd266d85b24389fe5de10d1fa0b3dbdde18d1c2de"
        ),
        "git_blob": "c2da15ff249e46a8fac2040d67f531a683b7fd7e",
        "git_mode": "100644",
        "authority_commit": "2550e0b8ee348b4217744a73d9781dba1e1e91a3",
    },
    {
        "identity": "G9CB-1C",
        "path": (
            "docs/"
            "gross9-structural-clock-bundle-preregistration-correction-"
            "amendment-2026-07-31.md"
        ),
        "path_type": "regular_file",
        "sha256": (
            "b79151c3378960017ddb30b7c1040f3027be538acad00776315380c267c6acaf"
        ),
        "git_blob": "94c0f3e13680f9e0ebbdb07ae7646b9505891e46",
        "git_mode": "100644",
        "authority_commit": "eee3383c9b2f88f4ea28f5bfe3a5ff6a650cec0f",
    },
]

EXPECTED_PROTOCOL_PATHS = [
    (
        "docs/"
        "gross9-structural-clock-bundle-g9cb6-successor-authority-decision-"
        "2026-07-31.md"
    ),
    (
        "docs/"
        "gross9-structural-clock-bundle-g9cb5-successor-authority-decision-"
        "2026-07-31.md"
    ),
    (
        "docs/"
        "gross9-structural-clock-bundle-g9cb4-successor-authority-decision-"
        "2026-07-31.md"
    ),
    (
        "docs/"
        "gross9-structural-clock-bundle-g9cb3-successor-authority-decision-"
        "2026-07-31.md"
    ),
    (
        "docs/"
        "gross9-structural-clock-bundle-successor-authority-decision-"
        "2026-07-31.md"
    ),
    "docs/gross9-structural-clock-bundle-authority-decision-2026-07-31.md",
    (
        "docs/"
        "gross9-structural-clock-bundle-rank7-authority-amendment-"
        "2026-07-31.md"
    ),
    (
        "docs/"
        "gross9-structural-clock-bundle-runtime-isolation-amendment-"
        "2026-07-31.md"
    ),
    (
        "docs/"
        "gross9-structural-clock-bundle-preregistration-correction-"
        "amendment-2026-07-31.md"
    ),
    "training/preregister_gross9_structural_clock_bundle.py",
    "tests/test_preregister_gross9_structural_clock_bundle.py",
    "tests/test_gross9_structural_clock_bundle_preregistration_artifact.py",
    "training/build_gross9_structural_clock_bundle.py",
    "tests/test_build_gross9_structural_clock_bundle.py",
    "training/gross9_structural_clock_primitives.py",
    "tests/test_gross9_structural_clock_primitives.py",
    "execution/gross9_rank7_clock_runtime.py",
    "tests/test_gross9_rank7_clock_runtime.py",
]

EXPECTED_SUPERSEDED_PREREGISTRATION = {
    "path": (
        "results/"
        "gross9_structural_clock_bundle_preregistration_2026-07-31.json"
    ),
    "path_type": "regular_file",
    "sha256": (
        "3580a3663b54509d004dc2edac0f18ff9c79cb80b199e8de5e9b1a9feb98d472"
    ),
    "git_blob": "61992d68beff0da255b002776d0efdb4ef96ab93",
    "git_mode": "100644",
    "filesystem_mode_octal": "0444",
    "seal_commit": "3810a3b7e24b83591866f2ccf9b63167795718c5",
    "protocol_parent_commit": "05437c3d8f2a9c556fde4e950a815b9901f7fc98",
    "protocol_version": "gross9_structural_clock_bundle_preregistration_v1",
    "manifest_hash": (
        "5ddf4c5c0aef42e1fb24defa78fccbd4142c8274bc22fd0a7d7e97fa9e8bb9bb"
    ),
    "status": "historical_nonoperative_preclaim_validation_failure",
}

EXPECTED_FAILED_V2_PREREGISTRATION = {
    "path": (
        "results/"
        "gross9_structural_clock_bundle_preregistration_v2_2026-07-31.json"
    ),
    "path_type": "regular_file",
    "sha256": (
        "5e6fe5e23f78103e5e4c6a288bb12df5f6aaa4e00028a211a175221a58b48e84"
    ),
    "git_blob": "6bf7c4fd62818c639b11da943f25353946d141b6",
    "git_mode": "100644",
    "filesystem_mode_octal": "0444",
    "seal_commit": "c5c5120cb5af931294524d4833f44440f8949327",
    "protocol_implementation_commit": (
        "d4ebec8f151fc5db6d318734ca0b6a79afaad1e1"
    ),
    "protocol_version": "gross9_structural_clock_bundle_preregistration_v2",
    "manifest_hash": (
        "e83d2bec1300c34401931c2b45c6c0b8715f4237eba0ae01811c665718b11a54"
    ),
    "status": (
        "historical_nonoperative_preclaim_git_metadata_contract_failure"
    ),
}

EXPECTED_FAILED_PREDECESSOR_PREREGISTRATIONS = [
    EXPECTED_SUPERSEDED_PREREGISTRATION,
    EXPECTED_FAILED_V2_PREREGISTRATION,
]

EXPECTED_CONSUMPTION_LEDGER_PATHS = [
    (
        "results/"
        "gross9_structural_clock_bundle_g9cb6_worker_capability_consumed_pass1_"
        "2026-07-31.json"
    ),
    (
        "results/"
        "gross9_structural_clock_bundle_g9cb6_worker_capability_consumed_pass2_"
        "2026-07-31.json"
    ),
]


def _q5_git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _q5_tracked_binding(root: Path, relative: Path) -> dict[str, Any]:
    raw = (root / relative).read_bytes()
    return {
        "path": relative.as_posix(),
        "path_type": "regular_file",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob": _q5_git(root, "rev-parse", f"HEAD:{relative.as_posix()}"),
        "git_mode": "100644",
    }


def _prepare_synthetic_q5_repository(
    base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any], str, str]:
    root = base / "repo"
    remote = base / "remote.git"
    root.mkdir()
    subprocess.run(
        ["git", "init", "--bare", "-q", str(remote)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _q5_git(root, "init", "-q", "-b", prereg.EXPECTED_BRANCH)
    _q5_git(root, "config", "user.email", "q5@example.invalid")
    _q5_git(root, "config", "user.name", "Q6 Synthetic")

    protocol_paths = tuple(
        sorted(prereg.PROTOCOL_PATHS, key=lambda path: path.as_posix())
    )
    for relative in protocol_paths:
        candidate = root / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if relative == Path("training/__init__.py"):
            raw = b""
        elif relative in prereg.RUNTIME_IMPORT_ROOTS:
            raw = b"SYNTHETIC_RUNTIME_VALUE = 1\n"
        else:
            raw = f"synthetic A5 bytes: {relative.as_posix()}\n".encode()
        candidate.write_bytes(raw)

    predecessor_path = Path("results/synthetic-predecessor-metadata.json")
    (root / predecessor_path).parent.mkdir(parents=True)
    (root / predecessor_path).write_bytes(b'{"synthetic":true}\n')
    tracked_nested_path = Path(
        "results/synthetic-tracked-directory/"
        "gross9_structural_clock_bundle_g9cb6_preregistration_2026-07-31.json"
    )
    (root / tracked_nested_path).parent.mkdir(parents=True)
    (root / tracked_nested_path).write_bytes(b"synthetic nested tracked leaf\n")
    results = root / "results"
    results.mkdir(exist_ok=True)
    (root / ".gitignore").write_text(
        "\n".join(
            [
                "results/gross9_structural_clock_bundle_g9cb6_*",
                "results/.gross9-structural-clock-g9cb6-worker-*",
                "results/.g9cb6-bytecode-cache-disabled",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _q5_git(root, "add", ".")
    _q5_git(root, "commit", "-qm", "synthetic A5")
    a5 = _q5_git(root, "rev-parse", "HEAD")

    for entry in prereg.SUCCESSOR_PROTOCOL_DIFF:
        status, path_text = entry.split("\t", 1)
        assert status == "M"
        candidate = root / path_text
        candidate.write_bytes(
            f"synthetic Q6 bytes: {path_text}\n".encode()
        )
    _q5_git(root, "add", ".")
    _q5_git(root, "commit", "-qm", "synthetic Q6")
    q5 = _q5_git(root, "rev-parse", "HEAD")
    _q5_git(root, "remote", "add", "origin", str(remote))
    _q5_git(root, "push", "-qu", "origin", prereg.EXPECTED_BRANCH)

    for identity in ("g9cb2", "g9cb3"):
        stage = results / f".synthetic-{identity}-slot1"
        stage.mkdir(mode=0o700)
        stage.chmod(0o700)

    predecessor = _q5_tracked_binding(root, predecessor_path)
    attempts = [
        {
            "identity": identity.upper().replace("G9CB", "G9CB-"),
            "authority_decision": predecessor,
            "preregistration": predecessor,
            "permanently_absent_outputs": [
                f"results/synthetic_{identity}_reserved.json"
            ],
            "residue": {
                "slot1_stage": {
                    "path": f"results/.synthetic-{identity}-slot1",
                    "state": "empty_directory",
                    "filesystem_mode_octal": "0700",
                    "committed": False,
                },
                "slot2_stage": {
                    "path": f"results/.synthetic-{identity}-slot2",
                    "state": "absent",
                    "committed": False,
                },
                **(
                    {
                        "bytecode_cache": {
                            "path": "results/.synthetic-g9cb3-pycache",
                            "state": "absent",
                        }
                    }
                    if identity == "g9cb3"
                    else {}
                ),
            },
        }
        for identity in ("g9cb2", "g9cb3")
    ]
    closure = {
        "identity": "G9CB-4",
        "preregistration": predecessor,
        "permanently_absent_outputs": [
            "results/synthetic_g9cb4_reserved.json"
        ],
        "residue": {
            "bytecode_cache": {
                "path": "results/.synthetic-g9cb4-pycache",
                "state": "absent",
            },
            "publication_stages": {
                "glob": "results/.synthetic-g9cb4-publish-*",
                "state": "absent",
            },
            "worker_stages": {
                "glob": "results/.synthetic-g9cb4-worker-*",
                "state": "absent",
            },
        },
    }
    prepublication_closure = {
        "identity": "G9CB-5",
        "authority_decision": predecessor,
        "permanently_absent_outputs": [
            "results/synthetic_g9cb5_reserved.json"
        ],
        "residue": {
            "bytecode_cache": {
                "path": "results/.synthetic-g9cb5-pycache",
                "state": "absent",
            },
            "publication_stages": {
                "glob": "results/.synthetic-g9cb5-publish-*",
                "state": "absent",
            },
            "worker_stages": {
                "glob": "results/.synthetic-g9cb5-worker-*",
                "state": "absent",
            },
        },
    }
    predecessors = [predecessor]
    successors = [
        {
            "identity": row["identity"],
            "preregistration": row["preregistration"],
        }
        for row in (*attempts, closure)
    ]
    amendment_paths = (
        prereg.RANK7_AUTHORITY_AMENDMENT_PATH,
        prereg.RUNTIME_ISOLATION_AMENDMENT_PATH,
        prereg.PREREGISTRATION_CORRECTION_AMENDMENT_PATH,
    )
    amendments = [
        {
            "identity": identity,
            **_q5_tracked_binding(root, path),
            "authority_commit": a5,
        }
        for identity, path in zip(
            ("G9CB-1A", "G9CB-1B", "G9CB-1C"),
            amendment_paths,
            strict=True,
        )
    ]
    authority = _q5_tracked_binding(root, prereg.AUTHORITY_DECISION_PATH)
    authority["authority_commit"] = a5
    runtime_closure = prereg.import_closure_inventory(
        prereg.RUNTIME_IMPORT_ROOTS,
        root,
    )

    monkeypatch.setattr(prereg, "AUTHORITY_DECISION_COMMIT", a5)
    monkeypatch.setattr(
        prereg,
        "_active_authority_decision_binding",
        lambda _root=root: dict(authority),
    )
    monkeypatch.setattr(
        prereg,
        "_authority_amendment_bindings",
        lambda _root=root: [dict(row) for row in amendments],
    )
    monkeypatch.setattr(
        prereg,
        "_historical_v2_authority_amendments",
        lambda: [dict(row) for row in amendments],
    )
    monkeypatch.setattr(
        prereg,
        "_direct_authority_inventory",
        lambda _root=root: [],
    )
    monkeypatch.setattr(
        prereg,
        "import_closure_inventory",
        lambda _roots, _root=root: [dict(row) for row in runtime_closure],
    )
    monkeypatch.setattr(prereg, "validate_environment", lambda _root=root: {})
    monkeypatch.setattr(prereg, "validate_config_metadata", lambda _root=root: [])
    monkeypatch.setattr(prereg, "validate_rank7_bundle", lambda _root=root: {})
    monkeypatch.setattr(prereg, "validate_sources", lambda _root=root: [])
    monkeypatch.setattr(
        prereg,
        "source_preclaim_disclosures",
        lambda _root=root: {
            "frozen_open_interest_gzip_opaque_bytes_opened_preclaim": True,
            "frozen_open_interest_gzip_decompressed_preclaim": False,
        },
    )
    monkeypatch.setattr(
        prereg,
        "expected_failed_predecessor_preregistration_bindings",
        lambda: [dict(row) for row in predecessors],
    )
    monkeypatch.setattr(
        prereg,
        "validate_failed_predecessor_preregistrations",
        lambda _root=root: [dict(row) for row in predecessors],
    )
    monkeypatch.setattr(
        prereg,
        "expected_failed_predecessor_attempts",
        lambda: copy.deepcopy(attempts),
    )
    monkeypatch.setattr(
        prereg,
        "expected_failed_predecessor_closures",
        lambda: [copy.deepcopy(closure)],
    )
    monkeypatch.setattr(
        prereg,
        "expected_failed_predecessor_prepublication_closures",
        lambda: [copy.deepcopy(prepublication_closure)],
    )
    monkeypatch.setattr(
        prereg,
        "expected_successor_preregistration_bindings",
        lambda: copy.deepcopy(successors),
    )

    manifest = prereg._manifest_without_hash(
        root,
        require_git_seal=False,
    )
    manifest["protocol_implementation_commit"] = q5
    manifest["manifest_hash"] = prereg.canonical_hash(manifest)
    prereg.validate_manifest(
        manifest,
        repository_root=root,
        verify_files=False,
        verify_environment=False,
        verify_git_seal=False,
    )
    return root, manifest, a5, q5


def _zero_access_payload() -> dict[str, object]:
    return {
        "creation_evidence_boundary": dict(
            prereg.CREATION_EVIDENCE_BOUNDARY
        ),
        "permanent_prohibited_counters": dict(
            prereg.PERMANENT_PROHIBITED_COUNTERS
        ),
        "pre2025_anchor_boundary": {
            "pre2025_anchor_bytes_hashed": True,
            "pre2025_anchor_git_blob_authenticated": True,
            "pre2025_anchor_json_parsed": False,
            "pre2025_anchor_value_rows_opened": 0,
        },
        "candidate_independence": {
            "candidate_identity_present": False,
            "candidate_artifacts_opened": False,
            "comparator_clock_rows_opened": 0,
            "comparator_clocks_preseen_by_research_program": True,
        },
    }


def test_canonical_json_and_manifest_hash_contract() -> None:
    payload = {"z": 1, "ascii": "한", "manifest_hash": "discarded"}
    assert prereg.canonical_json_bytes(payload) == (
        b'{"ascii":"\\ud55c","manifest_hash":"discarded","z":1}'
    )
    expected = hashlib.sha256(b'{"ascii":"\\ud55c","z":1}').hexdigest()
    assert prereg.canonical_hash(payload) == expected
    assert prereg.canonical_json_bytes(payload, trailing_lf=True).endswith(b"\n")


def test_closed_zero_access_schema_accepts_only_exact_typed_objects() -> None:
    payload = _zero_access_payload()
    prereg.validate_zero_access_schema(payload)

    integer_locations = [
        *[
            ("creation_evidence_boundary", key)
            for key in prereg.CREATION_ZERO_COUNTER_NAMES
        ],
        *[
            ("permanent_prohibited_counters", key)
            for key in prereg.PERMANENT_PROHIBITED_COUNTERS
        ],
        (
            "pre2025_anchor_boundary",
            "pre2025_anchor_value_rows_opened",
        ),
        ("candidate_independence", "comparator_clock_rows_opened"),
    ]
    for section, key in integer_locations:
        for invalid in (False, 0.0, "0", None, []):
            malformed = copy.deepcopy(payload)
            malformed[section][key] = invalid  # type: ignore[index]
            with pytest.raises(ValueError, match="counter|integer zero|rows"):
                prereg.validate_zero_access_schema(malformed)

    boolean_locations = [
        *[
            ("creation_evidence_boundary", key)
            for key in prereg.CREATION_FALSE_DECLARATION_NAMES
        ],
        ("pre2025_anchor_boundary", "pre2025_anchor_json_parsed"),
        ("candidate_independence", "candidate_identity_present"),
        ("candidate_independence", "candidate_artifacts_opened"),
    ]
    for section, key in boolean_locations:
        malformed = copy.deepcopy(payload)
        malformed[section][key] = 0  # type: ignore[index]
        with pytest.raises(ValueError, match="false|declaration"):
            prereg.validate_zero_access_schema(malformed)


def test_closed_zero_access_schema_rejects_unknown_and_misplaced_keys() -> None:
    for key in (
        "invented_values_computed",
        "invented_rows_opened",
        "invented_rows_examined",
        "invented_files_loaded",
        "invented_modules_imported",
        "invented_counter",
        "invented_counters",
    ):
        nested = _zero_access_payload()
        nested["nested"] = {key: 0}
        with pytest.raises(ValueError, match="misplaced"):
            prereg.validate_zero_access_schema(nested)

        top_level = _zero_access_payload()
        top_level[key] = 0
        with pytest.raises(ValueError, match="misplaced"):
            prereg.validate_zero_access_schema(top_level)

    misplaced = _zero_access_payload()
    misplaced["nested"] = {"cagr_values_computed": 0}
    with pytest.raises(ValueError, match="misplaced"):
        prereg.validate_zero_access_schema(misplaced)

    additional = _zero_access_payload()
    additional["permanent_prohibited_counters"][  # type: ignore[index]
        "invented_values_computed"
    ] = 0
    with pytest.raises(ValueError, match="schema differs"):
        prereg.validate_zero_access_schema(additional)


def test_historical_v1_zero_counters_are_valid_but_not_v2() -> None:
    historical = json.loads(
        (
            prereg.REPOSITORY_ROOT
            / prereg.HISTORICAL_PREREGISTRATION_PATH
        ).read_bytes()
    )
    prereg.validate_zero_access_schema(historical)
    assert historical["protocol_version"] == prereg.HISTORICAL_PROTOCOL_VERSION
    assert historical["protocol_version"] != prereg.PROTOCOL_VERSION
    assert prereg.PREREGISTRATION_PATH != (
        prereg.HISTORICAL_PREREGISTRATION_PATH
    )


def test_sha256_file_hashes_opaque_bytes_without_parsing(tmp_path: Path) -> None:
    opaque = tmp_path / "opaque.bin"
    opaque.write_bytes(b"\x1f\x8b\x08\x00not-a-row\x00\xff")
    assert prereg.sha256_file(opaque) == hashlib.sha256(opaque.read_bytes()).hexdigest()


def test_validate_file_rejects_symlink_and_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"opaque")
    link = tmp_path / "link"
    link.symlink_to(source)
    digest = hashlib.sha256(b"opaque").hexdigest()
    assert prereg.validate_file(source, digest)["path_type"] == "regular_file"
    with pytest.raises(ValueError, match="expected regular_file"):
        prereg.validate_file(link, digest)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prereg.validate_file(source, "0" * 64)


def test_optional_git_metadata_classifies_tracked_untracked_and_external(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.email", "g9cb4-test@example.invalid")
    git("config", "user.name", "G9CB-4 Test")
    tracked = repository / "tracked.bin"
    tracked.write_bytes(b"tracked")
    git("add", "tracked.bin")
    git("commit", "-m", "tracked")
    tracked_blob = git("rev-parse", "HEAD:tracked.bin")
    assert prereg._optional_git_metadata(
        "tracked.bin", repository
    ) == {"git_blob": tracked_blob, "git_mode": "100644"}

    untracked = repository / "untracked.bin"
    untracked.write_bytes(b"untracked")
    assert prereg._optional_git_metadata(
        "untracked.bin", repository
    ) == {"git_blob": None, "git_mode": None}

    external = tmp_path / "external.bin"
    external.write_bytes(b"external")
    assert prereg._optional_git_metadata(
        external.as_posix(), repository
    ) == {"git_blob": None, "git_mode": None}

    with pytest.raises(ValueError, match="must be repository-relative"):
        prereg._optional_git_metadata(tracked.as_posix(), repository)

    tracked.write_bytes(b"worktree drift")
    with pytest.raises(ValueError, match="worktree Git blob differs"):
        prereg._optional_git_metadata("tracked.bin", repository)
    git("checkout", "--", "tracked.bin")

    tracked.write_bytes(b"index drift")
    git("add", "tracked.bin")
    with pytest.raises(ValueError, match="index and HEAD"):
        prereg._optional_git_metadata("tracked.bin", repository)


def test_optional_git_metadata_rejects_unborn_or_invalid_head(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(
        ["git", "init"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (repository / "untracked.bin").write_bytes(b"untracked")
    with pytest.raises(ValueError, match="untracked Git classification"):
        prereg._optional_git_metadata("untracked.bin", repository)


def test_static_import_closure_is_exact_and_does_not_import_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("MARKER = 1\n", encoding="utf-8")
    (package / "leaf.py").write_text("VALUE = 2\n", encoding="utf-8")
    (package / "entry.py").write_text(
        "from . import leaf\nimport missing_external\n", encoding="utf-8"
    )

    def synthetic_binding(
        path: str | Path,
        *,
        repository_root: Path,
        expected_sha256: str | None = None,
        expected_blob: str | None = None,
    ) -> dict[str, object]:
        candidate = repository_root / path
        return {
            "path": Path(path).as_posix(),
            "path_type": "regular_file",
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "git_blob": "a" * 40,
            "git_mode": "100644",
        }

    monkeypatch.setattr(prereg, "_tracked_binding", synthetic_binding)
    closure = prereg.import_closure_inventory([Path("pkg/entry.py")], tmp_path)
    assert [item["path"] for item in closure] == [
        "pkg/__init__.py",
        "pkg/entry.py",
        "pkg/leaf.py",
    ]
    assert closure[0]["package_initializer"] is True
    assert closure[1]["package_initializer"] is False


def test_closure_validator_rejects_new_local_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.py").write_text("VALUE = 1\n", encoding="utf-8")

    def synthetic_binding(
        path: str | Path,
        *,
        repository_root: Path,
        expected_sha256: str | None = None,
        expected_blob: str | None = None,
    ) -> dict[str, object]:
        return {
            "path": Path(path).as_posix(),
            "path_type": "regular_file",
            "sha256": prereg.sha256_file(path, repository_root),
            "git_blob": "b" * 40,
            "git_mode": "100644",
        }

    monkeypatch.setattr(prereg, "_tracked_binding", synthetic_binding)
    expected = prereg.import_closure_inventory(["a.py"], tmp_path)
    (tmp_path / "b.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("import b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="closure mismatch"):
        prereg.validate_import_closure(expected, ["a.py"], tmp_path)


def test_repository_authority_amendments_authenticate_in_canonical_order() -> None:
    decision = prereg._authority_decision_binding()
    assert type(decision) is dict
    assert decision == {
        "path": (
            "docs/gross9-structural-clock-bundle-g9cb6-successor-"
            "authority-decision-2026-07-31.md"
        ),
        "path_type": "regular_file",
        "sha256": (
            "b64f9480741eeb4f69ac86736589fbcf8fb75565c436d76316b73f5e076acfca"
        ),
        "git_blob": "eb743d9f8ecd878b83f8f8873697c58cccef9f1b",
        "git_mode": "100644",
        "authority_commit": "2695ee61fbb9b5e053dbb9da597ebe2729aad361",
    }

    assert prereg._authority_amendment_bindings() == EXPECTED_AUTHORITY_AMENDMENTS


def test_failed_predecessor_preregistrations_are_exact_nonoperative_evidence() -> None:
    assert prereg.expected_superseded_preregistration_binding() == (
        EXPECTED_SUPERSEDED_PREREGISTRATION
    )
    assert prereg.validate_superseded_preregistration() == (
        EXPECTED_SUPERSEDED_PREREGISTRATION
    )
    assert prereg.expected_failed_v2_preregistration_binding() == (
        EXPECTED_FAILED_V2_PREREGISTRATION
    )
    assert prereg.validate_failed_v2_preregistration() == (
        EXPECTED_FAILED_V2_PREREGISTRATION
    )
    assert prereg.expected_failed_predecessor_preregistration_bindings() == (
        EXPECTED_FAILED_PREDECESSOR_PREREGISTRATIONS
    )
    assert prereg.validate_failed_predecessor_preregistrations() == (
        EXPECTED_FAILED_PREDECESSOR_PREREGISTRATIONS
    )


def test_g9cb2_terminal_attempt_is_exact_failed_history() -> None:
    attempts = prereg.expected_failed_predecessor_attempts()
    assert len(attempts) == 2
    row = attempts[0]
    assert row["identity"] == "G9CB-2"
    assert row["topology"] == {
        "g9cb2_authority_commit": "0a2847c8589908def4243890727c3640f806e109",
        "g9cb2_claim_commit": "731f093eb963b9e7213778ed4f259ee5466cd893",
        "g9cb2_preregistration_commit": "04550a47686ee039f82dfdb412d3c3eec4b5d6a1",
        "g9cb2_protocol_commit": "f48634af22dcad84ffde885fa970635d133cc126",
        "g9cb3_authority_commit": "a97576c050cf7cdf08738ddb755e63cc92484428",
        "terminal_evidence_commit": "edad4de5cf5524c4646c64b0581e47c914e31425",
    }
    assert len(row["protocol_implementation"]["files"]) == 5
    assert [item["path"] for item in row["protocol_implementation"]["files"]] == [
        "tests/test_build_gross9_structural_clock_bundle.py",
        "tests/test_gross9_structural_clock_bundle_preregistration_artifact.py",
        "tests/test_preregister_gross9_structural_clock_bundle.py",
        "training/build_gross9_structural_clock_bundle.py",
        "training/preregister_gross9_structural_clock_bundle.py",
    ]
    assert len(row["permanently_absent_outputs"]) == 4
    assert row["failure_counters"] == {
        "candidate_rows_opened": 0,
        "comparator_clock_rows_opened": 0,
        "generic_runtime_modules_imported": 0,
        "pre2025_anchor_value_rows_opened": 0,
        "source_value_rows_opened": 0,
        "worker_capabilities_consumed": 0,
        "worker_git_children_launched": 0,
        "worker_ledgers_published": 0,
    }
    assert prereg.validate_failed_predecessor_attempts() == attempts


def test_g9cb3_terminal_attempt_binds_atomic_sentinel_and_pass1_ledger() -> None:
    row = prereg.expected_failed_predecessor_attempts()[1]
    assert row["identity"] == "G9CB-3"
    assert set(row["terminal_evidence"]) == {
        "attempt_sentinel",
        "pass1_worker_ledger",
    }
    assert {
        binding["seal_commit"]
        for binding in row["terminal_evidence"].values()
    } == {prereg.G9CB3_TERMINAL_EVIDENCE_COMMIT}
    assert row["exposure"]["decoded_and_handed_off"] == [
        "market",
        "funding",
        "premium",
        "open_interest",
    ]
    assert row["exposure"]["exact_decoded_and_handoff_counts_recoverable"] is False
    assert row["root_cause"]["domain_end_is_exclusive_boundary"] is True
    assert row["root_cause"]["fabricated_boundary_value_authorized"] is False


@pytest.mark.parametrize(
    "missing_key",
    ("attempt_sentinel", "pass1_worker_ledger"),
)
def test_g9cb3_terminal_evidence_rejects_single_file_schema_variants(
    monkeypatch: pytest.MonkeyPatch,
    missing_key: str,
) -> None:
    monkeypatch.setattr(prereg, "_protocol_inventory", lambda *_a, **_k: [])
    monkeypatch.setattr(prereg, "_direct_authority_inventory", lambda *_a: [])
    monkeypatch.setattr(prereg, "import_closure_inventory", lambda *_a: [])
    monkeypatch.setattr(prereg, "validate_environment", lambda *_a: {})
    monkeypatch.setattr(prereg, "validate_config_metadata", lambda *_a: {})
    monkeypatch.setattr(prereg, "validate_rank7_bundle", lambda *_a: {})
    monkeypatch.setattr(prereg, "validate_sources", lambda *_a: [])
    manifest = prereg._manifest_without_hash(
        prereg.REPOSITORY_ROOT, require_git_seal=False
    )
    row = manifest["bindings"]["failed_predecessor_attempts"][1]
    del row["terminal_evidence"][missing_key]
    manifest["manifest_hash"] = prereg.canonical_hash(manifest)
    with pytest.raises(ValueError, match="failed predecessor attempt"):
        prereg.validate_manifest(
            manifest,
            verify_files=False,
            verify_environment=False,
            verify_git_seal=False,
        )


def test_g9cb3_terminal_evidence_cannot_be_reclassified_as_d3_products() -> None:
    row = prereg.expected_failed_predecessor_attempts()[1]
    sentinel = row["terminal_evidence"]["attempt_sentinel"]
    ledger = row["terminal_evidence"]["pass1_worker_ledger"]
    assert sentinel["seal_commit"] == ledger["seal_commit"]
    assert sentinel["seal_commit"] == prereg.G9CB3_TERMINAL_EVIDENCE_COMMIT
    assert row["status"] == (
        "historical_terminal_attempt_consumed_no_clock_authority"
    )
    forbidden_roles = {
        "pass_receipt",
        "per_pass_core",
        "canonical_csv_gzip",
        "final_manifest",
        "D3",
    }
    assert forbidden_roles.isdisjoint(row)
    assert forbidden_roles.isdisjoint(row["terminal_evidence"])


@pytest.mark.parametrize(
    "omitted",
    ("attempt_sentinel", "pass1_worker_ledger"),
)
def test_g9cb3_terminal_evidence_rejects_single_file_repository_variants(
    tmp_path: Path,
    omitted: str,
) -> None:
    attempts = prereg.expected_failed_predecessor_attempts()
    bindings = [
        attempts[0][key]
        for key in (
            "authority_decision",
            "preregistration",
            "access_claim",
            "attempt_sentinel",
        )
    ]
    bindings.extend(
        attempts[1][key]
        for key in ("authority_decision", "preregistration", "access_claim")
    )
    bindings.extend(
        binding
        for key, binding in attempts[1]["terminal_evidence"].items()
        if key != omitted
    )
    for binding in bindings:
        source = prereg.REPOSITORY_ROOT / binding["path"]
        target = tmp_path / binding["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        if "filesystem_mode_octal" in binding:
            target.chmod(int(binding["filesystem_mode_octal"], 8))

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.email", "g9cb4-test@example.invalid")
    git("config", "user.name", "G9CB-4 Test")
    git("add", ".")
    git("commit", "-m", "single evidence variant")
    with pytest.raises(ValueError, match="Git classification"):
        prereg._validate_failed_attempt_current_files(tmp_path)


def test_producer_preclassifies_g9cb2_git_pairs_before_single_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []
    original_git = prereg._git_result
    original_read = prereg._read_no_follow_once

    def recorded_git(
        arguments: object,
        root: Path,
    ) -> subprocess.CompletedProcess[str]:
        values = list(arguments)  # type: ignore[arg-type]
        if values and values[0] in {"ls-files", "ls-tree"}:
            events.append(("git", values[-1]))
        return original_git(values, root)

    def recorded_read(path: Path) -> tuple[bytes, object]:
        events.append(
            (
                "read",
                path.relative_to(prereg.REPOSITORY_ROOT).as_posix(),
            )
        )
        return original_read(path)

    monkeypatch.setattr(prereg, "_git_result", recorded_git)
    monkeypatch.setattr(prereg, "_read_no_follow_once", recorded_read)
    prereg.validate_failed_predecessor_attempts()
    attempts = prereg.expected_failed_predecessor_attempts()
    current_paths = [
        attempts[0][key]["path"]
        for key in (
            "authority_decision",
            "preregistration",
            "access_claim",
            "attempt_sentinel",
        )
    ] + [
        attempts[1][key]["path"]
        for key in ("authority_decision", "preregistration", "access_claim")
    ] + [
        binding["path"]
        for binding in attempts[1]["terminal_evidence"].values()
    ]
    first_read = next(
        index for index, event in enumerate(events) if event[0] == "read"
    )
    for path_text in current_paths:
        git_events = [
            index
            for index, event in enumerate(events)
            if event == ("git", path_text)
        ]
        assert len(git_events) == 3
        assert max(git_events) < first_read
        assert events.count(("read", path_text)) == 1


def test_protocol_commit_topology_accepts_exact_synthetic_a5_q5_p5_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _q5_git(tmp_path, "init", "-q", "-b", prereg.EXPECTED_BRANCH)
    _q5_git(tmp_path, "config", "user.email", "q5-topology@example.invalid")
    _q5_git(tmp_path, "config", "user.name", "Q6 Topology")

    def paths(diff: tuple[str, ...], expected_status: str) -> tuple[Path, ...]:
        parsed = []
        for entry in diff:
            status, path_text = entry.split("\t", 1)
            assert status == expected_status
            parsed.append(Path(path_text))
        return tuple(parsed)

    protocol_paths = paths(prereg.G9CB1_CORRECTION_PROTOCOL_DIFF, "M")
    for relative in protocol_paths:
        candidate = tmp_path / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(f"synthetic base {relative}\n".encode())
    (tmp_path / "synthetic-base").write_bytes(b"synthetic topology root\n")
    _q5_git(tmp_path, "add", ".")
    _q5_git(tmp_path, "commit", "-qm", "synthetic historical seal")
    commits = {
        "HISTORICAL_PREREGISTRATION_SEAL_COMMIT": _q5_git(
            tmp_path, "rev-parse", "HEAD"
        )
    }

    def commit_delta(
        key: str,
        label: str,
        diff: tuple[str, ...],
    ) -> str:
        for entry in diff:
            status, path_text = entry.split("\t", 1)
            candidate = tmp_path / path_text
            candidate.parent.mkdir(parents=True, exist_ok=True)
            if status == "A":
                assert not candidate.exists()
            else:
                assert status == "M"
                assert candidate.is_file()
            candidate.write_bytes(f"{label}: {path_text}\n".encode())
        _q5_git(tmp_path, "add", ".")
        _q5_git(tmp_path, "commit", "-qm", label)
        commit = _q5_git(tmp_path, "rev-parse", "HEAD")
        commits[key] = commit
        return commit

    chain = (
        (
            "PREREGISTRATION_CORRECTION_AMENDMENT_COMMIT",
            "synthetic G9CB-1C authority",
            prereg.G9CB1_CORRECTION_AUTHORITY_DIFF,
        ),
        (
            "FAILED_V2_PROTOCOL_IMPLEMENTATION_COMMIT",
            "synthetic failed-v2 protocol",
            prereg.G9CB1_CORRECTION_PROTOCOL_DIFF,
        ),
        (
            "FAILED_V2_PREREGISTRATION_SEAL_COMMIT",
            "synthetic failed-v2 preregistration",
            prereg.FAILED_V2_PREREGISTRATION_DIFF,
        ),
        (
            "G9CB2_AUTHORITY_DECISION_COMMIT",
            "synthetic G9CB-2 authority",
            prereg.G9CB2_SUCCESSOR_AUTHORITY_DIFF,
        ),
        (
            "G9CB2_PROTOCOL_IMPLEMENTATION_COMMIT",
            "synthetic G9CB-2 protocol",
            prereg.G9CB2_SUCCESSOR_PROTOCOL_DIFF,
        ),
        (
            "G9CB2_PREREGISTRATION_SEAL_COMMIT",
            "synthetic G9CB-2 preregistration",
            prereg.G9CB2_ACTIVE_PREREGISTRATION_DIFF,
        ),
        (
            "G9CB2_CLAIM_COMMIT",
            "synthetic G9CB-2 claim",
            prereg.G9CB2_CLAIM_DIFF,
        ),
        (
            "G9CB3_AUTHORITY_DECISION_COMMIT",
            "synthetic G9CB-3 authority",
            prereg.G9CB3_SUCCESSOR_AUTHORITY_DIFF,
        ),
        (
            "G9CB2_TERMINAL_EVIDENCE_COMMIT",
            "synthetic G9CB-2 terminal evidence",
            prereg.G9CB2_TERMINAL_EVIDENCE_DIFF,
        ),
        (
            "G9CB3_PROTOCOL_IMPLEMENTATION_COMMIT",
            "synthetic G9CB-3 protocol",
            prereg.G9CB3_PROTOCOL_DIFF,
        ),
        (
            "G9CB3_PREREGISTRATION_SEAL_COMMIT",
            "synthetic G9CB-3 preregistration",
            prereg.G9CB3_ACTIVE_PREREGISTRATION_DIFF,
        ),
        (
            "G9CB3_CLAIM_COMMIT",
            "synthetic G9CB-3 claim",
            prereg.G9CB3_CLAIM_DIFF,
        ),
        (
            "G9CB4_AUTHORITY_DECISION_COMMIT",
            "synthetic G9CB-4 authority",
            prereg.G9CB4_SUCCESSOR_AUTHORITY_DIFF,
        ),
        (
            "G9CB3_TERMINAL_EVIDENCE_COMMIT",
            "synthetic G9CB-3 terminal evidence",
            prereg.TERMINAL_EVIDENCE_DIFF,
        ),
        (
            "G9CB4_PROTOCOL_IMPLEMENTATION_COMMIT",
            "synthetic G9CB-4 protocol",
            prereg.SUCCESSOR_PROTOCOL_DIFF,
        ),
        (
            "G9CB4_PREREGISTRATION_SEAL_COMMIT",
            "synthetic G9CB-4 preregistration",
            prereg.G9CB4_ACTIVE_PREREGISTRATION_DIFF,
        ),
        (
            "G9CB5_AUTHORITY_DECISION_COMMIT",
            "synthetic A5 authority",
            prereg.G9CB5_SUCCESSOR_AUTHORITY_DIFF,
        ),
        (
            "G9CB5_PROTOCOL_IMPLEMENTATION_COMMIT",
            "synthetic Q5 protocol",
            prereg.SUCCESSOR_PROTOCOL_DIFF,
        ),
        (
            "AUTHORITY_DECISION_COMMIT",
            "synthetic A6 authority",
            prereg.SUCCESSOR_AUTHORITY_DIFF,
        ),
    )
    for key, label, diff in chain:
        commit_delta(key, label, diff)
    q6 = commit_delta(
        "Q6_PROTOCOL_IMPLEMENTATION_COMMIT",
        "synthetic Q6 protocol",
        prereg.SUCCESSOR_PROTOCOL_DIFF,
    )
    commit_delta(
        "P6_PREREGISTRATION_SEAL_COMMIT",
        "synthetic P6 preregistration",
        prereg.ACTIVE_PREREGISTRATION_DIFF,
    )

    for name, commit in commits.items():
        if hasattr(prereg, name):
            monkeypatch.setattr(prereg, name, commit)

    assert prereg.validate_protocol_commit_topology(tmp_path) == q6

    a6 = commits["AUTHORITY_DECISION_COMMIT"]
    _q5_git(tmp_path, "checkout", "-q", "--detach", a6)
    for relative in paths(prereg.SUCCESSOR_PROTOCOL_DIFF, "M"):
        (tmp_path / relative).write_bytes(
            f"malformed Q6 delta: {relative}\n".encode()
        )
    (tmp_path / "unexpected-q6-path").write_bytes(b"unexpected\n")
    _q5_git(tmp_path, "add", ".")
    _q5_git(tmp_path, "commit", "-qm", "malformed Q6 protocol")
    with pytest.raises(ValueError, match="implementation diff"):
        prereg.validate_protocol_commit_topology(tmp_path)


def test_protocol_paths_include_exact_sorted_unique_g9cb6_inventory() -> None:
    assert [path.as_posix() for path in prereg.PROTOCOL_PATHS] == (
        EXPECTED_PROTOCOL_PATHS
    )
    assert len(prereg.PROTOCOL_PATHS) == 18
    assert len(set(prereg.PROTOCOL_PATHS)) == 18
    assert sorted(path.as_posix() for path in prereg.PROTOCOL_PATHS) == sorted(
        EXPECTED_PROTOCOL_PATHS
    )


def test_tracked_results_projection_uses_first_component_not_descendant_basename() -> None:
    assert prereg._tracked_results_top_level_entries(
        "results/direct.json\n"
        "results/nested/child/direct.json\n"
        "results/collision/child/nested\n"
    ) == {"direct.json", "nested", "collision"}


def test_binary_git_reader_preserves_blob_bytes_without_newline_normalization(
    tmp_path: Path,
) -> None:
    _q5_git(tmp_path, "init", "-q")
    _q5_git(tmp_path, "config", "user.email", "binary@example.invalid")
    _q5_git(tmp_path, "config", "user.name", "Binary Git Reader")
    raw = b"exact historical bytes\r\n\n"
    (tmp_path / "historical.py").write_bytes(raw)
    _q5_git(tmp_path, "add", "historical.py")
    _q5_git(tmp_path, "commit", "-qm", "historical bytes")

    assert prereg._run_git_bytes(
        ["show", "HEAD:historical.py"], tmp_path
    ) == raw


@pytest.mark.parametrize(
    "path_text",
    ["other/file", "results", "results/", "results/../leaf", "/results/leaf"],
)
def test_tracked_results_projection_rejects_malformed_paths(path_text: str) -> None:
    with pytest.raises(ValueError, match="malformed tracked results path"):
        prereg._tracked_results_top_level_entries(path_text)


@pytest.mark.parametrize("drift", ["missing-tracked-top-level", "extra-untracked"])
def test_q6_results_inventory_rejects_missing_tracked_or_extra_untracked_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    root, manifest, _a6, _q6 = _prepare_synthetic_q5_repository(
        tmp_path, monkeypatch
    )
    if drift == "missing-tracked-top-level":
        target = root / "results/synthetic-tracked-directory"
        for child in target.iterdir():
            child.unlink()
        target.rmdir()
    else:
        (root / "results/untracked-extra").write_bytes(b"extra\n")
    snapshot, _pairs = prereg._prepare_preregistration_snapshot(manifest, root)
    try:
        results_fd, _leaf = snapshot._parent(
            prereg.PREREGISTRATION_PATH.as_posix()
        )
        with pytest.raises(FileExistsError, match="exact results inventory"):
            prereg._validate_closed_path_state(
                results_fd,
                prereg.Q6_PREREGISTRATION_PUBLICATION,
                snapshot=snapshot,
                preregistration=False,
                claim=False,
                worker_stage=False,
                fixed_pycache=False,
            )
    finally:
        snapshot.close()


def test_runtime_import_inventory_has_only_isolated_facade_and_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert [path.as_posix() for path in prereg.RUNTIME_IMPORT_ROOTS] == [
        "execution/gross9_rank7_clock_runtime.py",
        "training/gross9_structural_clock_primitives.py",
    ]
    direct = prereg._direct_authority_inventory()
    assert len(direct) == len(prereg.DIRECT_AUTHORITY_BINDINGS)

    def synthetic_binding(
        path: str | Path,
        *,
        repository_root: Path,
        expected_sha256: str | None = None,
        expected_blob: str | None = None,
    ) -> dict[str, object]:
        candidate = repository_root / path
        return {
            "path": Path(path).as_posix(),
            "path_type": "regular_file",
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "git_blob": "c" * 40,
            "git_mode": "100644",
        }

    monkeypatch.setattr(prereg, "_tracked_binding", synthetic_binding)
    closure = prereg.import_closure_inventory(prereg.RUNTIME_IMPORT_ROOTS)
    assert {path.as_posix() for path in prereg.RUNTIME_IMPORT_ROOTS}.issubset(
        {item["path"] for item in closure}
    )
    assert all(item["git_blob"] and item["git_mode"] == "100644" for item in closure)
    assert {
        "execution/portfolio_live.py",
        "execution/rank7_runtime.py",
        "execution/rex_llm_live.py",
    }.isdisjoint({item["path"] for item in closure})


def test_worker_process_environment_substitutes_canonical_synthetic_root(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical-repository"
    canonical_root.mkdir()
    synthetic_alias = tmp_path / "synthetic-repository"
    synthetic_alias.symlink_to(canonical_root, target_is_directory=True)

    environment = prereg.worker_process_environment(synthetic_alias)

    assert list(environment) == [
        "BLIS_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LC_ALL",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONHASHSEED",
        "PYTHONIOENCODING",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
        "PYTHONPYCACHEPREFIX",
        "PYTHONUNBUFFERED",
        "PYTHONUTF8",
        "TZ",
        "VECLIB_MAXIMUM_THREADS",
    ]
    assert environment == {
        "BLIS_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": canonical_root.as_posix(),
        "PYTHONPYCACHEPREFIX": (
            canonical_root / "results/.g9cb6-bytecode-cache-disabled"
        ).as_posix(),
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "TZ": "UTC",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def test_manifest_binds_g9cb_1b_contract_and_exact_rank7_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prereg,
        "_protocol_inventory",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(prereg, "_direct_authority_inventory", lambda *_args: [])
    monkeypatch.setattr(prereg, "import_closure_inventory", lambda *_args: [])
    monkeypatch.setattr(prereg, "validate_environment", lambda *_args: {})
    monkeypatch.setattr(prereg, "validate_config_metadata", lambda *_args: {})
    monkeypatch.setattr(prereg, "validate_rank7_bundle", lambda *_args: {})
    monkeypatch.setattr(prereg, "validate_sources", lambda *_args: [])
    manifest = prereg._manifest_without_hash(
        prereg.REPOSITORY_ROOT, require_git_seal=False
    )
    assert manifest["bindings"]["authority_amendments"] == (
        EXPECTED_AUTHORITY_AMENDMENTS
    )
    assert manifest["bindings"]["failed_predecessor_preregistrations"] == (
        EXPECTED_FAILED_PREDECESSOR_PREREGISTRATIONS
    )
    assert manifest["bindings"]["failed_predecessor_attempts"] == (
        prereg.expected_failed_predecessor_attempts()
    )
    assert manifest["protocol_implementation_commit"] == "0" * 40
    assert manifest["protocol_version"] == (
        "gross9_structural_clock_bundle_g9cb6_preregistration_v1"
    )
    assert manifest["output_paths"]["preregistration"] == (
        "results/"
        "gross9_structural_clock_bundle_g9cb6_preregistration_2026-07-31.json"
    )
    assert manifest["bindings"]["runtime_import_roots"] == [
        "execution/gross9_rank7_clock_runtime.py",
        "training/gross9_structural_clock_primitives.py",
    ]
    assert "adapter_import_roots" not in manifest["bindings"]
    assert "adapter_import_closure" not in manifest["bindings"]
    assert manifest["output_paths"]["worker_capability_consumption_ledgers"] == (
        EXPECTED_CONSUMPTION_LEDGER_PATHS
    )
    assert manifest["access_counter_names"]["rows_used"] == [
        "causal_feature_rows_by_source",
        "prediction_rows_scored",
        "outcome_dependent_ohlc_rows_examined",
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
    manifest["manifest_hash"] = prereg.canonical_hash(manifest)
    prereg.validate_manifest(
        manifest,
        verify_files=False,
        verify_environment=False,
        verify_git_seal=False,
    )
    manifest["bindings"]["authority_amendments"].reverse()
    manifest["manifest_hash"] = prereg.canonical_hash(manifest)
    with pytest.raises(ValueError, match="authority amendment bindings mismatch"):
        prereg.validate_manifest(
            manifest,
            verify_files=False,
            verify_environment=False,
            verify_git_seal=False,
        )


def test_permitted_manifests_declare_exact_rank7_and_source_inventories() -> None:
    assert prereg._declared_sources() == list(prereg.SOURCE_BINDINGS)
    assert prereg._rank7_declared_files() == [
        (path, digest)
        for path, digest, _blob in prereg.RANK7_FILE_BINDINGS
    ]
    config = prereg.validate_config_metadata()
    assert config["gross_weight"] == 9.0
    assert config["portfolio_weights"] == {
        sleeve["name"]: sleeve["configured_weight"] for sleeve in prereg.SLEEVES
    }


def test_frozen_environment_inventory_authenticates_under_project_runner() -> None:
    environment = prereg.validate_environment()
    assert environment["distribution_count"] == 108
    assert environment["distribution_inventory_sha256"] == (
        prereg.FROZEN_ENVIRONMENT["distribution_inventory_sha256"]
    )
    assert environment["selected_distributions"]["sqlalchemy"] == "absent"


def test_source_preclaim_disclosures_are_exact_without_row_counts() -> None:
    disclosures = prereg.source_preclaim_disclosures()
    assert disclosures == {
        "frozen_open_interest_gzip_logical_path": (
            "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
        ),
        "frozen_open_interest_gzip_resolved_path": (
            "/home/pakchu/rllm/data/"
            "cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
        ),
        "frozen_open_interest_gzip_size_bytes": 72_898_508,
        "frozen_open_interest_gzip_sha256": (
            "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192"
        ),
        "frozen_open_interest_gzip_opaque_bytes_opened_preclaim": True,
        "frozen_open_interest_gzip_decompressed_preclaim": False,
        "frozen_open_interest_gzip_headers_decoded_preclaim": 0,
        "frozen_open_interest_gzip_rows_decoded_preclaim": 0,
        "frozen_open_interest_gzip_fields_or_values_opened_preclaim": 0,
        "open_interest_logical_path": (
            "/tmp/btcusdt_open_interest_5m_2020_2026.csv"
        ),
        "open_interest_artifact_size_bytes": 19_657_777,
        "open_interest_artifact_bytes_read_for_sha256_preclaim": 19_657_777,
        "open_interest_sha256_preclaim": (
            "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31"
        ),
        "open_interest_headers_decoded_preclaim": 0,
        "open_interest_rows_decoded_preclaim": 0,
        "open_interest_fields_or_values_opened_preclaim": 0,
    }


def test_anchor_loader_guard_forbids_value_parsing() -> None:
    anchor = Path("results/gross9_pre2025_authoritative_anchor_2026-07-28.json")
    with pytest.raises(ValueError, match="hash-only"):
        prereg._load_json_metadata(anchor, prereg.REPOSITORY_ROOT)


def test_write_once_is_singleton_create_only_and_verifies_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _q5_git(tmp_path, "init", "-q")
    _q5_git(tmp_path, "config", "user.email", "q5-tests@example.invalid")
    _q5_git(tmp_path, "config", "user.name", "Q6 tests")
    (tmp_path / "host-marker").write_bytes(b"synthetic host repository\n")
    _q5_git(tmp_path, "add", "host-marker")
    _q5_git(tmp_path, "commit", "-qm", "synthetic host")
    root = tmp_path / "synthetic-root"
    results = root / "results"
    results.mkdir(parents=True)
    target = results / prereg.PREREGISTRATION_PATH.name
    retained = results / "synthetic-retained-input.bin"
    retained.write_bytes(b"retained preregistration input\n")
    monkeypatch.setattr(prereg, "PREREGISTRATION_PATH", Path("results") / target.name)
    monkeypatch.setattr(
        prereg,
        "expected_failed_predecessor_attempts",
        lambda: [],
    )
    monkeypatch.setattr(
        prereg,
        "expected_failed_predecessor_closures",
        lambda: [],
    )

    manifest = {
        "protocol_version": prereg.PROTOCOL_VERSION,
        "identity": prereg.IDENTITY,
        "candidate_independence": {
            "candidate_identity_present": False,
            "candidate_artifacts_opened": False,
            "comparator_clock_rows_opened": 0,
            "comparator_clocks_preseen_by_research_program": True,
        },
        "creation_evidence_boundary": dict(prereg.CREATION_EVIDENCE_BOUNDARY),
        "source_preclaim_disclosures": {
            "frozen_open_interest_gzip_opaque_bytes_opened_preclaim": True,
            "frozen_open_interest_gzip_decompressed_preclaim": False,
        },
        "synthetic_retained_binding": {
            "path": retained.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(retained.read_bytes()).hexdigest(),
        },
    }
    manifest["manifest_hash"] = prereg.canonical_hash(manifest)

    monkeypatch.setattr(prereg, "validate_manifest", lambda *_args, **_kwargs: None)
    assert prereg.write_once(manifest, repository_root=root) is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    assert target.read_bytes() == prereg.canonical_json_bytes(
        manifest, trailing_lf=True
    )
    assert prereg.write_once(manifest, repository_root=root) is False

    target.chmod(0o644)
    target.write_bytes(b"other\n")
    with pytest.raises(FileExistsError, match="other bytes"):
        prereg.write_once(manifest, repository_root=root)

    target.unlink()
    referent = root / "referent.json"
    referent.write_bytes(
        prereg.canonical_json_bytes(manifest, trailing_lf=True)
    )
    target.symlink_to(referent)
    with pytest.raises(FileExistsError, match="regular file"):
        prereg.write_once(manifest, repository_root=root)


def test_output_path_rejects_noncanonical_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prereg,
        "PREREGISTRATION_PATH",
        Path("results") / prereg.PREREGISTRATION_PATH.name,
    )
    with pytest.raises(ValueError, match="only canonical"):
        prereg._validate_output_path(tmp_path / "other.json", tmp_path)


def test_module_is_stdlib_metadata_only_and_has_no_forbidden_dependency() -> None:
    source_path = Path(prereg.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    allowed = {
        "__future__",
        "argparse",
        "ast",
        "hashlib",
        "importlib",
        "json",
        "os",
        "pathlib",
        "platform",
        "re",
        "stat",
        "subprocess",
        "sys",
        "tempfile",
        "typing",
        "zlib",
    }
    assert imported_roots <= allowed
    assert not imported_roots.intersection(
        {"numpy", "pandas", "scipy", "sklearn", "torch", "transformers"}
    )
    lowered = source.lower()
    for forbidden in (
        "read_parquet",
        "gzip.open",
        "numpy.load",
        "_reconstruct_gross9_runtime_clocks",
        "settlement_demand_impulse",
    ):
        assert forbidden not in lowered
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_csv"
        for node in ast.walk(tree)
    )
    subprocess_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.value
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert len(subprocess_calls) == 1
    call = subprocess_calls[0]
    assert isinstance(call.args[0], ast.List)
    assert isinstance(call.args[0].elts[0], ast.Constant)
    assert call.args[0].elts[0].value == "git"


def test_g9cb6_identity_paths_and_terminal_literals_are_exact() -> None:
    assert prereg.IDENTITY == "G9CB-6"
    assert prereg.PROTOCOL_VERSION == (
        "gross9_structural_clock_bundle_g9cb6_preregistration_v1"
    )
    assert prereg.PREREGISTRATION_PATH.as_posix() == (
        "results/"
        "gross9_structural_clock_bundle_g9cb6_preregistration_2026-07-31.json"
    )
    assert prereg.ACCESS_CLAIM_PATH.as_posix() == (
        "results/"
        "gross9_structural_clock_bundle_g9cb6_access_claim_2026-07-31.json"
    )
    assert prereg.ATTEMPT_SENTINEL_PATH.as_posix() == (
        "results/"
        "gross9_structural_clock_bundle_g9cb6_attempt_consumed_2026-07-31.json"
    )


def test_g9cb6_authority_is_add_only_child_of_frozen_q5() -> None:
    assert prereg.AUTHORITY_DECISION_COMMIT == (
        "2695ee61fbb9b5e053dbb9da597ebe2729aad361"
    )
    assert prereg.G9CB5_PROTOCOL_IMPLEMENTATION_COMMIT == (
        "02c3c83a5253684057f44f51ee96bcb089b40b2f"
    )
    assert prereg._single_parent(
        prereg.AUTHORITY_DECISION_COMMIT,
        prereg.REPOSITORY_ROOT,
    ) == prereg.G9CB5_PROTOCOL_IMPLEMENTATION_COMMIT


def test_g9cb5_prepublication_closure_authenticates_exact_git_bytes() -> None:
    assert prereg.validate_failed_predecessor_prepublication_closures() == (
        prereg.expected_failed_predecessor_prepublication_closures()
    )


@pytest.mark.parametrize("reserved_kind", ["output", "bytecode"])
def test_g9cb5_prepublication_closure_rejects_dangling_reserved_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reserved_kind: str,
) -> None:
    [closure] = prereg.expected_failed_predecessor_prepublication_closures()
    historical_raw = {
        path.as_posix(): subprocess.run(
            [
                "git",
                "show",
                (
                    f"{prereg.G9CB5_PROTOCOL_IMPLEMENTATION_COMMIT}:"
                    f"{path.as_posix()}"
                ),
            ],
            cwd=prereg.REPOSITORY_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        for path in (prereg.PREREGISTRATION_SOURCE, prereg.BUILDER_SOURCE)
    }
    authority = closure["authority_decision"]
    monkeypatch.setattr(
        prereg,
        "_tracked_binding",
        lambda *_args, **_kwargs: {
            key: authority[key]
            for key in ("path", "path_type", "sha256", "git_blob", "git_mode")
        },
    )
    monkeypatch.setattr(
        prereg,
        "_run_git_bytes",
        lambda arguments, _root: historical_raw[
            str(arguments[-1]).split(":", 1)[1]
        ],
    )

    results = tmp_path / "results"
    results.mkdir()
    path_text = (
        closure["permanently_absent_outputs"][0]
        if reserved_kind == "output"
        else closure["residue"]["bytecode_cache"]["path"]
    )
    (results / Path(path_text).name).symlink_to("missing-target")

    with pytest.raises(ValueError, match="reserved G9CB-5|bytecode residue"):
        prereg.validate_failed_predecessor_prepublication_closures(tmp_path)


def test_successor_schema_has_three_preregistrations_two_attempts_one_closure() -> None:
    preregistrations = prereg.expected_successor_preregistration_bindings()
    attempts = prereg.expected_failed_predecessor_attempts()
    closures = prereg.expected_failed_predecessor_closures()
    assert [row["identity"] for row in preregistrations] == [
        "G9CB-2",
        "G9CB-3",
        "G9CB-4",
    ]
    assert [row["identity"] for row in attempts] == ["G9CB-2", "G9CB-3"]
    assert [row["identity"] for row in closures] == ["G9CB-4"]


def test_g9cb4_closure_binds_exact_frozen_opaque_evidence() -> None:
    [closure] = prereg.expected_failed_predecessor_closures()
    assert closure["classification"] == (
        "pre_access_claim_pre_sentinel_keyword_only_call_contract_failure"
    )
    assert closure["preregistration"]["seal_commit"] == (
        "01de73258902d754905319b906345c865a016558"
    )
    assert closure["preregistration"]["sha256"] == (
        "f65aaf5fd2219f90421912e6fc9065ddffb54f5adf881196986f25185fe7342e"
    )
    assert closure["preregistration"]["manifest_hash"] == (
        "fa3dab6f7e6ab86428c03fc5c3d7b005e0a165cd76662bba9a7c3cd5941beeed"
    )
    assert closure["protocol_implementation"]["builder_sha256"] == (
        "c7c3bf1f9971e058e719139b50379c356f45a0fcc8f62c12aab100f70fa64c63"
    )
    assert closure["exposure"]["source_values_opened"] == 0
    assert closure["exposure"]["economics_or_overlap_computed"] is False


def test_g9cb4_closure_has_no_attempt_claim_publication_or_terminal_commit() -> None:
    [closure] = prereg.expected_failed_predecessor_closures()
    assert closure["failure"]["claim_payload_constructed"] is False
    assert closure["failure"]["claim_write_attempted"] is False
    assert closure["failure"]["sentinel_published"] is False
    assert closure["topology"]["terminal_evidence_commit"] is None
    forbidden = {"attempt", "C4", "D4", "T4", "sentinel"}
    assert forbidden.isdisjoint(closure)


def test_g9cb4_closure_permanent_absences_and_residue_are_exact() -> None:
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
    assert closure["residue"] == {
        "bytecode_cache": {
            "path": "results/.g9cb4-bytecode-cache-disabled",
            "state": "absent",
        },
        "publication_stages": {
            "glob": "results/.gross9_structural_clock_bundle_g9cb4_*.stage-*",
            "state": "absent",
        },
        "worker_stages": {
            "glob": "results/.gross9-structural-clock-g9cb4-worker-*",
            "state": "absent",
        },
    }


def test_manifest_carries_closed_predecessor_classes_without_reclassification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prereg, "_protocol_inventory", lambda *_a, **_k: [])
    monkeypatch.setattr(prereg, "_direct_authority_inventory", lambda *_a: [])
    monkeypatch.setattr(prereg, "import_closure_inventory", lambda *_a: [])
    monkeypatch.setattr(prereg, "validate_environment", lambda *_a: {})
    monkeypatch.setattr(prereg, "validate_config_metadata", lambda *_a: {})
    monkeypatch.setattr(prereg, "validate_rank7_bundle", lambda *_a: {})
    monkeypatch.setattr(prereg, "validate_sources", lambda *_a: [])
    manifest = prereg._manifest_without_hash(
        prereg.REPOSITORY_ROOT,
        require_git_seal=False,
    )
    assert [row["identity"] for row in manifest["bindings"][
        "successor_preregistrations"
    ]] == ["G9CB-2", "G9CB-3", "G9CB-4"]
    assert [row["identity"] for row in manifest["bindings"][
        "failed_predecessor_attempts"
    ]] == ["G9CB-2", "G9CB-3"]
    assert [row["identity"] for row in manifest["bindings"][
        "failed_predecessor_closures"
    ]] == ["G9CB-4"]
    assert [row["identity"] for row in manifest["bindings"][
        "failed_predecessor_prepublication_closures"
    ]] == ["G9CB-5"]


def test_g9cb6_stage_and_pycache_prefixes_do_not_alias_predecessor_residue() -> None:
    active = {
        "results/.gross9-structural-clock-g9cb6-worker-",
        "results/.g9cb6-bytecode-cache-disabled",
    }
    predecessor = {
        "results/.gross9-structural-clock-worker-",
        "results/.gross9-structural-clock-g9cb3-worker-",
        "results/.gross9-structural-clock-g9cb4-worker-",
        "results/.g9cb3-bytecode-cache-disabled",
        "results/.g9cb4-bytecode-cache-disabled",
    }
    assert all(
        not current.startswith(old) and not old.startswith(current)
        for current in active
        for old in predecessor
    )


def test_actual_synthetic_q5_publication_is_pair_first_two_read_and_link_last(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, manifest, a5, q5 = _prepare_synthetic_q5_repository(
        tmp_path,
        monkeypatch,
    )
    synthetic_marker = root / "results/.synthetic-baseline"
    assert not synthetic_marker.exists()
    assert _q5_git(
        root,
        "ls-files",
        "--",
        synthetic_marker.relative_to(root).as_posix(),
    ) == ""
    events: list[tuple[str, str | int]] = []
    snapshot_fds: dict[str, int] = {}
    opening: list[str] = []
    git_commands: list[tuple[str, ...]] = []
    original_pair = prereg._classify_git_pair_only
    original_open = prereg._PreregistrationSnapshot.open_initial
    original_verify = prereg._PreregistrationSnapshot.verify_final
    original_prepare = prereg._prepare_unnamed_publication
    original_link = prereg._link_prepared_publication
    original_pread = prereg.os.pread
    original_git_result = prereg._git_result

    def recorded_pair(*args: Any, **kwargs: Any):
        events.append(("pair", str(args[1])))
        return original_pair(*args, **kwargs)

    def recorded_open(
        snapshot: prereg._PreregistrationSnapshot,
        path_text: str,
    ):
        events.append(("open", path_text))
        opening.append(path_text)
        try:
            result = original_open(snapshot, path_text)
        finally:
            opening.clear()
        snapshot_fds[path_text] = snapshot.files[path_text]
        return result

    def recorded_verify(
        snapshot: prereg._PreregistrationSnapshot,
    ) -> None:
        events.append(("verify", "bound-snapshot"))
        original_verify(snapshot)

    def recorded_prepare(results_fd: int, raw: bytes):
        events.append(("prepare", len(raw)))
        return original_prepare(results_fd, raw)

    def recorded_link(unnamed_fd: int, results_fd: int, leaf: str) -> None:
        events.append(("link", leaf))
        original_link(unnamed_fd, results_fd, leaf)

    def recorded_pread(fd: int, size: int, offset: int) -> bytes:
        if opening:
            events.append(("snapshot-read", opening[0]))
        else:
            for path_text, descriptor in snapshot_fds.items():
                if descriptor == fd:
                    events.append(("snapshot-read", path_text))
                    break
        return original_pread(fd, size, offset)

    def recorded_git_result(
        arguments: Any,
        repository_root: Path,
    ) -> subprocess.CompletedProcess[str]:
        git_commands.append(tuple(str(value) for value in arguments))
        return original_git_result(arguments, repository_root)

    monkeypatch.setattr(prereg, "_classify_git_pair_only", recorded_pair)
    monkeypatch.setattr(
        prereg._PreregistrationSnapshot,
        "open_initial",
        recorded_open,
    )
    monkeypatch.setattr(
        prereg._PreregistrationSnapshot,
        "verify_final",
        recorded_verify,
    )
    monkeypatch.setattr(
        prereg,
        "_prepare_unnamed_publication",
        recorded_prepare,
    )
    monkeypatch.setattr(prereg, "_link_prepared_publication", recorded_link)
    monkeypatch.setattr(prereg.os, "pread", recorded_pread)
    monkeypatch.setattr(prereg, "_git_result", recorded_git_result)

    assert prereg._validate_q6_publication_topology(root) == q5
    assert prereg._single_parent(q5, root) == a5
    assert prereg._commit_diff(a5, q5, root) == prereg.SUCCESSOR_PROTOCOL_DIFF
    assert prereg.write_once(manifest, repository_root=root) is True
    assert not synthetic_marker.exists()
    assert _q5_git(
        root,
        "ls-files",
        "--",
        synthetic_marker.relative_to(root).as_posix(),
    ) == ""

    first_open = next(
        index for index, event in enumerate(events) if event[0] == "open"
    )
    initial_pairs = {
        value for kind, value in events[:first_open] if kind == "pair"
    }
    assert initial_pairs == set(snapshot_fds)
    assert [
        value for kind, value in events if kind == "open"
    ] == sorted(snapshot_fds)
    for path_text in snapshot_fds:
        assert events.count(("snapshot-read", path_text)) == 2
    verify_indices = [
        index for index, event in enumerate(events) if event[0] == "verify"
    ]
    link_index = next(
        index
        for index, event in enumerate(events)
        if event == ("link", prereg.PREREGISTRATION_PATH.name)
    )
    prepare_index = max(
        index
        for index, event in enumerate(events[:link_index])
        if event[0] == "prepare"
    )
    assert len(verify_indices) == 1
    assert prepare_index < verify_indices[0] < link_index
    assert not any(
        kind in {"verify", "snapshot-read"}
        for kind, _value in events[link_index + 1 :]
    )
    assert all(
        not command or command[0] != "hash-object"
        for command in git_commands
    )

    target = root / prereg.PREREGISTRATION_PATH
    assert target.read_bytes() == prereg.canonical_json_bytes(
        manifest,
        trailing_lf=True,
    )
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    active = (
        prereg.ACCESS_CLAIM_PATH,
        prereg.ATTEMPT_SENTINEL_PATH,
        *prereg.WORKER_CAPABILITY_CONSUMPTION_LEDGER_PATHS,
        prereg.BUNDLE_PATH,
        prereg.FINAL_MANIFEST_PATH,
    )
    assert all(not (root / path).exists() for path in active)
    assert not list(
        (root / "results").glob(
            ".gross9-structural-clock-g9cb6-worker-*"
        )
    )
    assert not (
        root / "results/.g9cb6-bytecode-cache-disabled"
    ).exists()


def test_existing_preregistration_public_path_is_full_retained_verification_or_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, manifest, _a5, _q5 = _prepare_synthetic_q5_repository(
        tmp_path,
        monkeypatch,
    )
    assert prereg.write_once(manifest, repository_root=root) is True

    loader = getattr(prereg, "load_existing_artifact", None)
    if loader is None:
        with pytest.raises(SystemExit):
            prereg.main(["--verify-existing"])
        return

    events: list[tuple[str, str]] = []
    active_path = ""
    descriptor_paths: dict[int, str] = {}
    reads: dict[str, list[int]] = {}
    original_topology = prereg._validate_q6_publication_topology
    original_pair = prereg._classify_git_pair_only
    original_open = prereg._PreregistrationSnapshot.open_initial
    original_verify = prereg._PreregistrationSnapshot.verify_final
    original_pread = prereg.os.pread

    def recorded_topology(repository_root: Path) -> str:
        events.append(("topology", repository_root.as_posix()))
        return original_topology(repository_root)

    def recorded_pair(
        repository_root: Path,
        path_text: str,
        declaration: dict[str, Any],
    ) -> tuple[str, str] | None:
        events.append(("pair", path_text))
        return original_pair(repository_root, path_text, declaration)

    def recorded_open(
        snapshot: prereg._PreregistrationSnapshot,
        path_text: str,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal active_path
        active_path = path_text
        events.append(("open", path_text))
        result = original_open(snapshot, path_text)
        descriptor_paths[snapshot.files[path_text]] = path_text
        return result

    def recorded_verify(snapshot: prereg._PreregistrationSnapshot) -> None:
        events.append(("verify", "snapshot"))
        original_verify(snapshot)

    def recorded_pread(fd: int, size: int, offset: int) -> bytes:
        path_text = descriptor_paths.get(fd, active_path)
        if path_text:
            reads.setdefault(path_text, []).append(fd)
        return original_pread(fd, size, offset)

    monkeypatch.setattr(
        prereg,
        "_validate_q6_publication_topology",
        recorded_topology,
    )
    monkeypatch.setattr(prereg, "_classify_git_pair_only", recorded_pair)
    monkeypatch.setattr(
        prereg._PreregistrationSnapshot,
        "open_initial",
        recorded_open,
    )
    monkeypatch.setattr(
        prereg._PreregistrationSnapshot,
        "verify_final",
        recorded_verify,
    )
    monkeypatch.setattr(prereg.os, "pread", recorded_pread)

    loader_arguments: dict[str, Any] = {"repository_root": root}
    if "verify_files" in inspect.signature(loader).parameters:
        loader_arguments["verify_files"] = False
    assert loader(**loader_arguments) == manifest
    assert events.count(("topology", root.as_posix())) == 1
    opened = [value for kind, value in events if kind == "open"]
    paired = [value for kind, value in events if kind == "pair"]
    assert opened
    assert set(paired) == set(opened)
    assert prereg.PREREGISTRATION_PATH.as_posix() in opened
    first_open = next(
        index for index, (kind, _value) in enumerate(events) if kind == "open"
    )
    assert all(kind == "pair" for kind, _value in events[1:first_open])
    assert events.count(("verify", "snapshot")) == 1
    for path_text in opened:
        assert len(reads[path_text]) == 2
        assert len(set(reads[path_text])) == 1


@pytest.mark.parametrize("identity", ["G9CB-2", "G9CB-3"])
@pytest.mark.parametrize("replacement_state", ["empty", "nonempty"])
def test_preregistration_retains_predecessor_residue_edge_through_final_recheck(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    identity: str,
    replacement_state: str,
) -> None:
    root, manifest, _a5, _q5 = _prepare_synthetic_q5_repository(
        tmp_path,
        monkeypatch,
    )
    snapshot, _pairs = prereg._prepare_preregistration_snapshot(
        manifest,
        root,
    )
    results_fd, _leaf = snapshot._parent(prereg.PREREGISTRATION_PATH.as_posix())
    attempts = prereg.expected_failed_predecessor_attempts()
    closures = prereg.expected_failed_predecessor_closures()
    stage_path = Path(
        next(row for row in attempts if row["identity"] == identity)[
            "residue"
        ]["slot1_stage"]["path"]
    )
    stage_leaf = stage_path.name
    detached_leaf = f".detached-{stage_leaf}"
    try:
        prereg._validate_closed_path_state(
            results_fd,
            prereg.Q6_PREREGISTRATION_PUBLICATION,
            snapshot=snapshot,
            preregistration=False,
            claim=False,
            worker_stage=False,
            fixed_pycache=False,
        )
        original = os.stat(
            stage_leaf,
            dir_fd=results_fd,
            follow_symlinks=False,
        )
        os.rename(
            stage_leaf,
            detached_leaf,
            src_dir_fd=results_fd,
            dst_dir_fd=results_fd,
        )
        os.rmdir(detached_leaf, dir_fd=results_fd)
        os.mkdir(stage_leaf, 0o700, dir_fd=results_fd)
        replacement = os.stat(
            stage_leaf,
            dir_fd=results_fd,
            follow_symlinks=False,
        )
        assert (replacement.st_dev, replacement.st_ino) != (
            original.st_dev,
            original.st_ino,
        )
        assert stat.S_IMODE(replacement.st_mode) == 0o700
        if replacement_state == "nonempty":
            replacement_fd = os.open(
                stage_leaf,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=results_fd,
            )
            try:
                marker_fd = os.open(
                    "replacement-marker",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o600,
                    dir_fd=replacement_fd,
                )
                os.close(marker_fd)
            finally:
                os.close(replacement_fd)
        results_info = os.fstat(results_fd)
        snapshot.rebaseline_directory(
            (results_info.st_dev, results_info.st_ino)
        )
        with pytest.raises(
            (ValueError, RuntimeError),
            match="residue|directory|parent|component|changed|graph",
        ):
            prereg._validate_predecessor_inventory_from_snapshot(
                snapshot,
                attempts,
                closures,
            )
            snapshot.verify_final()
    finally:
        snapshot.close()


@pytest.mark.parametrize("identity", ["G9CB-2", "G9CB-3"])
def test_preregistration_retained_residue_rejects_restored_timestamp_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    identity: str,
) -> None:
    root, manifest, _a5, _q5 = _prepare_synthetic_q5_repository(
        tmp_path,
        monkeypatch,
    )
    snapshot, _pairs = prereg._prepare_preregistration_snapshot(
        manifest,
        root,
    )
    results_fd, _leaf = snapshot._parent(prereg.PREREGISTRATION_PATH.as_posix())
    attempts = prereg.expected_failed_predecessor_attempts()
    stage_path = Path(
        next(row for row in attempts if row["identity"] == identity)[
            "residue"
        ]["slot1_stage"]["path"]
    )
    try:
        prereg._validate_closed_path_state(
            results_fd,
            prereg.Q6_PREREGISTRATION_PUBLICATION,
            snapshot=snapshot,
            preregistration=False,
            claim=False,
            worker_stage=False,
            fixed_pycache=False,
        )
        key = ("repo", tuple(stage_path.parts))
        stage_fd = snapshot.directories[key]
        before = os.fstat(stage_fd)
        time.sleep(0.01)
        marker_fd = os.open(
            "ephemeral-marker",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
            dir_fd=stage_fd,
        )
        os.close(marker_fd)
        os.unlink("ephemeral-marker", dir_fd=stage_fd)
        after = os.fstat(stage_fd)
        assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
        assert after.st_mtime_ns != before.st_mtime_ns
        assert not os.listdir(stage_fd)
        with pytest.raises(
            (ValueError, RuntimeError),
            match="timestamp|residue|directory|changed|graph",
        ):
            prereg._validate_predecessor_inventory_from_snapshot(
                snapshot,
                attempts,
                prereg.expected_failed_predecessor_closures(),
            )
            snapshot.verify_final()
    finally:
        snapshot.close()


@pytest.mark.parametrize(
    "closed_state",
    ["claim", "ledger", "worker-stage", "fixed-pycache"],
)
def test_q5_publication_rejects_every_preexisting_active_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    closed_state: str,
) -> None:
    root, manifest, _a5, _q5 = _prepare_synthetic_q5_repository(
        tmp_path,
        monkeypatch,
    )
    candidates = {
        "claim": root / prereg.ACCESS_CLAIM_PATH,
        "ledger": (
            root / prereg.WORKER_CAPABILITY_CONSUMPTION_LEDGER_PATHS[0]
        ),
        "worker-stage": (
            root
            / "results/.gross9-structural-clock-g9cb6-worker-existing"
        ),
        "fixed-pycache": (
            root / "results/.g9cb6-bytecode-cache-disabled"
        ),
    }
    candidate = candidates[closed_state]
    if closed_state in {"worker-stage", "fixed-pycache"}:
        candidate.mkdir()
    else:
        candidate.write_bytes(b"synthetic residue\n")
    with pytest.raises(FileExistsError, match="path-state|closed"):
        prereg.write_once(manifest, repository_root=root)
    assert not (root / prereg.PREREGISTRATION_PATH).exists()


@pytest.mark.parametrize("failure", ["dirty", "wrong-parent"])
def test_q5_publication_rejects_dirty_or_wrong_real_git_topology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    root, manifest, _a5, _q5 = _prepare_synthetic_q5_repository(
        tmp_path,
        monkeypatch,
    )
    if failure == "dirty":
        protocol = root / prereg.PROTOCOL_PATHS[0]
        protocol.write_bytes(protocol.read_bytes() + b"dirty\n")
    else:
        marker = root / "synthetic/intervening.txt"
        marker.parent.mkdir()
        marker.write_bytes(b"intervening\n")
        _q5_git(root, "add", marker.relative_to(root).as_posix())
        _q5_git(root, "commit", "-qm", "intervening")
        _q5_git(root, "push", "-q")
    with pytest.raises(ValueError, match="clean|direct child|diff"):
        prereg.write_once(manifest, repository_root=root)
    assert not (root / prereg.PREREGISTRATION_PATH).exists()


def test_q6_protocol_topology_constants_bind_a5_q5_a6_q6_p6_chain() -> None:
    assert prereg.G9CB4_PREREGISTRATION_SEAL_COMMIT == (
        "01de73258902d754905319b906345c865a016558"
    )
    assert prereg.G9CB5_AUTHORITY_DECISION_COMMIT == (
        "1ca718d9dab1077b041e753f3b011fbf5b23f047"
    )
    assert prereg.G9CB5_PROTOCOL_IMPLEMENTATION_COMMIT == (
        "02c3c83a5253684057f44f51ee96bcb089b40b2f"
    )
    assert prereg.AUTHORITY_DECISION_COMMIT == (
        "2695ee61fbb9b5e053dbb9da597ebe2729aad361"
    )
    assert prereg.SUCCESSOR_PROTOCOL_DIFF == (
        "M\ttests/test_build_gross9_structural_clock_bundle.py",
        "M\ttests/test_gross9_structural_clock_bundle_preregistration_artifact.py",
        "M\ttests/test_preregister_gross9_structural_clock_bundle.py",
        "M\ttraining/build_gross9_structural_clock_bundle.py",
        "M\ttraining/preregister_gross9_structural_clock_bundle.py",
    )


def test_preregistration_snapshot_rejects_symlink_and_retained_parent_swap(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "leaf").write_bytes(b"synthetic\n")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    symlink_snapshot = prereg._PreregistrationSnapshot(tmp_path)
    try:
        with pytest.raises(OSError):
            symlink_snapshot.open_initial("alias/leaf")
    finally:
        symlink_snapshot.close()

    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "leaf").write_bytes(b"initial\n")
    snapshot = prereg._PreregistrationSnapshot(tmp_path)
    try:
        snapshot.open_initial("parent/leaf")
        parent.rename(tmp_path / "retained-parent")
        parent.mkdir()
        (parent / "leaf").write_bytes(b"replacement\n")
        with pytest.raises(RuntimeError, match="directory graph|component"):
            snapshot.verify_final()
    finally:
        snapshot.close()
