from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import stat
import subprocess

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
        "gross9_structural_clock_bundle_g9cb3_worker_capability_consumed_pass1_"
        "2026-07-31.json"
    ),
    (
        "results/"
        "gross9_structural_clock_bundle_g9cb3_worker_capability_consumed_pass2_"
        "2026-07-31.json"
    ),
]


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
    git("config", "user.email", "g9cb3-test@example.invalid")
    git("config", "user.name", "G9CB-3 Test")
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
    assert decision["path"] == prereg.AUTHORITY_DECISION_PATH.as_posix()
    assert decision["sha256"] == prereg.AUTHORITY_DECISION_SHA256
    assert decision["git_blob"] == prereg.AUTHORITY_DECISION_GIT_BLOB
    assert decision["authority_commit"] == prereg.AUTHORITY_DECISION_COMMIT
    assert decision == {
        "path": (
            "docs/gross9-structural-clock-bundle-g9cb3-successor-"
            "authority-decision-2026-07-31.md"
        ),
        "path_type": "regular_file",
        "sha256": (
            "1df555c5149bfe269d2cc2c87375d54032809f13ac36f4e92b5ba00dd6e87cc7"
        ),
        "git_blob": "43d68f6b7407c19b3b52ef8b7bb7010797dbf3b3",
        "git_mode": "100644",
        "authority_commit": "a97576c050cf7cdf08738ddb755e63cc92484428",
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
    assert len(attempts) == 1
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
    row = prereg.expected_failed_predecessor_attempts()[0]
    current_paths = [
        row[key]["path"]
        for key in (
            "authority_decision",
            "preregistration",
            "access_claim",
            "attempt_sentinel",
        )
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


def test_protocol_commit_topology_accepts_exact_c2_a3_t2_q3_p3_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = "1" * 40
    seal = "2" * 40
    parents = {
        prereg.G9CB2_AUTHORITY_DECISION_COMMIT: (
            prereg.FAILED_V2_PREREGISTRATION_SEAL_COMMIT
        ),
        prereg.G9CB2_PROTOCOL_IMPLEMENTATION_COMMIT: (
            prereg.G9CB2_AUTHORITY_DECISION_COMMIT
        ),
        prereg.G9CB2_PREREGISTRATION_SEAL_COMMIT: (
            prereg.G9CB2_PROTOCOL_IMPLEMENTATION_COMMIT
        ),
        prereg.G9CB2_CLAIM_COMMIT: prereg.G9CB2_PREREGISTRATION_SEAL_COMMIT,
        prereg.AUTHORITY_DECISION_COMMIT: prereg.G9CB2_CLAIM_COMMIT,
        prereg.G9CB2_TERMINAL_EVIDENCE_COMMIT: (
            prereg.AUTHORITY_DECISION_COMMIT
        ),
        implementation: prereg.G9CB2_TERMINAL_EVIDENCE_COMMIT,
        seal: implementation,
    }
    diffs = {
        (
            prereg.FAILED_V2_PREREGISTRATION_SEAL_COMMIT,
            prereg.G9CB2_AUTHORITY_DECISION_COMMIT,
        ): prereg.G9CB2_SUCCESSOR_AUTHORITY_DIFF,
        (
            prereg.G9CB2_AUTHORITY_DECISION_COMMIT,
            prereg.G9CB2_PROTOCOL_IMPLEMENTATION_COMMIT,
        ): prereg.G9CB2_SUCCESSOR_PROTOCOL_DIFF,
        (
            prereg.G9CB2_PROTOCOL_IMPLEMENTATION_COMMIT,
            prereg.G9CB2_PREREGISTRATION_SEAL_COMMIT,
        ): prereg.G9CB2_ACTIVE_PREREGISTRATION_DIFF,
        (
            prereg.G9CB2_PREREGISTRATION_SEAL_COMMIT,
            prereg.G9CB2_CLAIM_COMMIT,
        ): prereg.G9CB2_CLAIM_DIFF,
        (
            prereg.G9CB2_CLAIM_COMMIT,
            prereg.AUTHORITY_DECISION_COMMIT,
        ): prereg.SUCCESSOR_AUTHORITY_DIFF,
        (
            prereg.AUTHORITY_DECISION_COMMIT,
            prereg.G9CB2_TERMINAL_EVIDENCE_COMMIT,
        ): prereg.TERMINAL_EVIDENCE_DIFF,
        (
            prereg.G9CB2_TERMINAL_EVIDENCE_COMMIT,
            implementation,
        ): prereg.SUCCESSOR_PROTOCOL_DIFF,
        (implementation, seal): prereg.ACTIVE_PREREGISTRATION_DIFF,
    }
    monkeypatch.setattr(
        prereg, "validate_failed_v2_preregistration_topology", lambda *_: None
    )
    monkeypatch.setattr(prereg, "_single_parent", lambda commit, *_: parents[commit])
    monkeypatch.setattr(prereg, "_commit_diff", lambda parent, child, *_: diffs[(parent, child)])
    monkeypatch.setattr(
        prereg,
        "_addition_commits",
        lambda path, *_: (
            (prereg.AUTHORITY_DECISION_COMMIT,)
            if path == prereg.AUTHORITY_DECISION_PATH
            else (
                (prereg.G9CB2_AUTHORITY_DECISION_COMMIT,)
                if path == prereg.G9CB2_AUTHORITY_DECISION_PATH
                else (seal,)
            )
        ),
    )
    monkeypatch.setattr(
        prereg,
        "_run_git",
        lambda arguments, *_: seal
        if arguments == ["rev-parse", "HEAD"]
        else "",
    )
    monkeypatch.setattr(prereg, "_require_ancestor", lambda *_: None)
    assert prereg.validate_protocol_commit_topology(Path("/synthetic")) == (
        implementation
    )

    diffs[(prereg.G9CB2_TERMINAL_EVIDENCE_COMMIT, implementation)] = (
        "M\ttraining/gross9_structural_clock_primitives.py",
    )
    with pytest.raises(ValueError, match="implementation diff"):
        prereg.validate_protocol_commit_topology(Path("/synthetic"))


def test_protocol_paths_include_exact_g9cb3_authority_and_modules() -> None:
    assert [path.as_posix() for path in prereg.PROTOCOL_PATHS] == (
        EXPECTED_PROTOCOL_PATHS
    )
    assert len(prereg.PROTOCOL_PATHS) == 15
    assert sorted(path.as_posix() for path in prereg.PROTOCOL_PATHS) == sorted(
        EXPECTED_PROTOCOL_PATHS
    )


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
            canonical_root / "results/.g9cb3-bytecode-cache-disabled"
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
        "gross9_structural_clock_bundle_g9cb3_preregistration_v1"
    )
    assert manifest["output_paths"]["preregistration"] == (
        "results/"
        "gross9_structural_clock_bundle_g9cb3_preregistration_2026-07-31.json"
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
    results = tmp_path / "results"
    results.mkdir()
    target = results / prereg.PREREGISTRATION_PATH.name
    monkeypatch.setattr(prereg, "PREREGISTRATION_PATH", Path("results") / target.name)

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
    }
    manifest["manifest_hash"] = prereg.canonical_hash(manifest)

    monkeypatch.setattr(prereg, "validate_manifest", lambda *_args, **_kwargs: None)
    assert prereg.write_once(manifest, repository_root=tmp_path) is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    assert target.read_bytes() == prereg.canonical_json_bytes(
        manifest, trailing_lf=True
    )
    assert prereg.write_once(manifest, repository_root=tmp_path) is False

    target.chmod(0o644)
    target.write_bytes(b"other\n")
    with pytest.raises(FileExistsError, match="other bytes"):
        prereg.write_once(manifest, repository_root=tmp_path)

    target.unlink()
    referent = tmp_path / "referent.json"
    referent.write_bytes(
        prereg.canonical_json_bytes(manifest, trailing_lf=True)
    )
    target.symlink_to(referent)
    with pytest.raises(FileExistsError, match="regular file"):
        prereg.write_once(manifest, repository_root=tmp_path)


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
        "read_csv",
        "read_parquet",
        "gzip.open",
        "numpy.load",
        "_reconstruct_gross9_runtime_clocks",
        "settlement_demand_impulse",
    ):
        assert forbidden not in lowered
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
