from __future__ import annotations

import ast
import copy
import dataclasses
import json
import os
import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import (
    build_protocol_specification_intent_maturity_d2_source_support as runner,
)
from training import (
    build_protocol_specification_intent_maturity_source_support as core,
)
from training import (
    preregister_protocol_specification_intent_maturity_d2 as prereg,
)


def _run(
    *arguments: str,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _synthetic_origin(tmp_path: Path) -> tuple[Path, str, str]:
    work = tmp_path / "work"
    work.mkdir()
    _run("git", "init", "-b", "master", cwd=work)
    _run("git", "config", "user.name", "PSIM D2 Test", cwd=work)
    _run(
        "git",
        "config",
        "user.email",
        "psim-d2@example.test",
        cwd=work,
    )
    proposal = work / "EIPS" / "eip-1.md"
    proposal.parent.mkdir()
    proposal.write_text(
        "---\neip: 1\ntitle: First\n---\n# Abstract\none\n",
        encoding="utf-8",
    )
    _run("git", "add", ".", cwd=work)
    _run(
        "git",
        "commit",
        "-m",
        "first",
        "--date=2020-01-02T00:00:00Z",
        cwd=work,
    )
    sealed_tip = _run("git", "rev-parse", "HEAD", cwd=work).stdout.strip()
    proposal.write_text(
        "---\neip: 1\ntitle: Second\n---\n# Abstract\ntwo\n",
        encoding="utf-8",
    )
    _run("git", "add", ".", cwd=work)
    _run(
        "git",
        "commit",
        "-m",
        "second",
        "--date=2020-01-03T00:00:00Z",
        cwd=work,
    )
    branch_tip = _run("git", "rev-parse", "HEAD", cwd=work).stdout.strip()
    origin = tmp_path / "origin.git"
    _run("git", "clone", "--bare", str(work), str(origin), cwd=tmp_path)
    return origin, sealed_tip, branch_tip


def _local_spec(origin: Path, sealed_tip: str) -> SimpleNamespace:
    return SimpleNamespace(
        protocol="ethereum",
        remote=origin.resolve().as_uri(),
        branch="master",
        remote_head_symref="refs/heads/master",
        sealed_tip=sealed_tip,
        object_format="sha1",
    )


def _synthetic_cards_events():
    events = runner.synthetic_events()
    cards = runner.build_daily_cards(events)
    return events, cards


def test_self_check_is_inherited_synthetic_only_and_opens_no_git_or_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):  # pragma: no cover - regression only
        raise AssertionError("PSIM-D2 self-check must not invoke Git/source")

    for module in (runner, core):
        monkeypatch.setattr(module, "_run_git", fail)
        monkeypatch.setattr(module, "_git_text", fail)
    monkeypatch.setattr(core, "_cat_file_batch", fail)

    payload = runner.build_self_check_manifest()

    assert payload["failed"] == []
    assert payload["policy_id"] == "PSIM-D2"
    assert payload["network_calls"] == 0
    assert payload["git_commands"] == 0
    assert payload["source_event_rows_opened"] == 0
    assert payload["official_source_opened"] is False
    assert payload["outcomes_opened"] is False
    assert payload["forbidden_access"] == (
        runner.AccessLedger.zero().snapshot()
    )
    assert payload["inherited_core"] == {
        "runner_path": runner.D1_CORE_RUNNER_PATH.as_posix(),
        "runner_commit": runner.D1_CORE_COMMIT,
        "runner_sha256": runner.D1_CORE_RUNNER_SHA256,
        "manifest_hash": runner.D1_CORE_SELF_CHECK_MANIFEST_HASH,
        "stdout_sha256": runner.D1_CORE_SELF_CHECK_STDOUT_SHA256,
    }


def test_runner_has_no_model_market_or_transport_client_imports() -> None:
    source = (runner.REPO_ROOT / runner.RUNNER_PATH).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint(
        {
            "aiohttp",
            "ccxt",
            "httpx",
            "models",
            "pandas",
            "requests",
            "sklearn",
            "torch",
            "transformers",
            "urllib",
            "yfinance",
        }
    )


def test_authority_constants_bind_d2_preregistration_and_d1_core() -> None:
    assert runner.POLICY_ID == "PSIM-D2"
    assert runner.PREREGISTRATION_COMMIT == (
        "e853f7688a484b323c024115e3ef4af07e6a5896"
    )
    assert runner.PREREGISTRATION_SHA256 == (
        "3b405de2bcdc1979855e8505148f7de3fbee366cb126e78b1b23e10f84cf470a"
    )
    assert runner.PREREGISTRATION_MANIFEST_HASH == (
        "917d2f318b268b01621c9e969237d76fc82d7e6aff408269842e660cc155d915"
    )
    assert runner.D1_CORE_COMMIT == (
        "80b656994f17548a7a599a548e23e9f1cd01302d"
    )
    assert runner.D1_CORE_RUNNER_SHA256 == (
        "414e83256b3ea489a9e1cd0995f6061e5fab550cd12c795ef7e88eff8998d9fb"
    )
    assert runner.D1_CORE_TEST_SHA256 == (
        "343aa1a72cfbca23d9756988ced042b5c61a6e8fc5a21a0b6d18e45870e906e9"
    )
    assert runner.sha256_file(runner.PREREGISTRATION_PATH) == (
        runner.PREREGISTRATION_SHA256
    )
    assert runner.sha256_file(runner.D1_CORE_RUNNER_PATH) == (
        runner.D1_CORE_RUNNER_SHA256
    )
    assert runner.sha256_file(runner.D1_CORE_TEST_PATH) == (
        runner.D1_CORE_TEST_SHA256
    )


def test_preregistration_loader_replays_exact_frozen_manifest() -> None:
    payload = runner._load_preregistration()
    assert payload == prereg.build_preregistration()
    assert payload["manifest_hash"] == runner.PREREGISTRATION_MANIFEST_HASH
    assert payload["candidate"]["id"] == runner.POLICY_ID
    assert payload["inheritance_proof"][
        "all_other_paths_byte_equal"
    ] is True


def test_git_environment_removes_all_inherited_git_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_DIR", "/tmp/hostile")
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "/tmp/shared")
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", "/tmp/alt")
    environment = runner._git_environment()
    assert "GIT_DIR" not in environment
    assert "GIT_OBJECT_DIRECTORY" not in environment
    assert "GIT_ALTERNATE_OBJECT_DIRECTORIES" not in environment
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_TERMINAL_PROMPT"] == "0"


def test_local_bare_acquisition_uses_exact_shape_and_never_git_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    origin, sealed_tip, branch_tip = _synthetic_origin(tmp_path)
    spec = _local_spec(origin, sealed_tip)
    monkeypatch.setattr(runner, "_repository_spec", lambda protocol: spec)
    config = runner.Config(source_root=tmp_path / "source")
    ledger = runner.AccessLedger.zero()
    calls: list[tuple[str, ...]] = []
    original = runner._run_git

    def record(arguments, **kwargs):
        calls.append(tuple(arguments))
        return original(arguments, **kwargs)

    monkeypatch.setattr(runner, "_run_git", record)
    receipt = runner.prepare_source_repository(
        config,
        "ethereum",
        "a",
        ledger,
    )

    assert calls[0] == (
        "clone",
        "--bare",
        "--filter=blob:none",
        "--single-branch",
        "--branch",
        "master",
        "--no-tags",
        spec.remote,
        str(config.source_root / "ethereum-a.git"),
    )
    assert not any("status" in call for call in calls)
    assert receipt["local_branch_oid"] == branch_tip
    assert receipt["sealed_tip"] == sealed_tip
    assert receipt["sealed_ref"] == "refs/psim-d2/sealed-tip"
    assert receipt["ref_roster"] == [
        "refs/heads/master",
        "refs/psim-d2/sealed-tip",
    ]
    assert receipt["is_bare_repository"] is True
    assert receipt["is_inside_work_tree"] is False
    assert receipt["forbidden_paths_absent"] is True
    assert receipt["git_status_invoked"] is False
    assert receipt["checkout_created"] is False
    assert receipt["object_store"]["regular_files"] > 0
    assert receipt["object_store"]["symlinks"] == 0
    assert receipt["object_store"]["multiple_link_files"] == 0
    assert ledger.network_commands == 3


def test_collect_chain_uses_sealed_ref_not_moving_master(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    origin, sealed_tip, branch_tip = _synthetic_origin(tmp_path)
    spec = _local_spec(origin, sealed_tip)
    monkeypatch.setattr(runner, "_repository_spec", lambda protocol: spec)
    config = runner.Config(source_root=tmp_path / "source")
    ledger = runner.AccessLedger.zero()
    runner.prepare_source_repository(
        config,
        "ethereum",
        "a",
        ledger,
    )

    records = runner.collect_commit_chain(
        runner.clone_path(config, "ethereum", "a"),
        "ethereum",
        ledger,
    )

    assert branch_tip != sealed_tip
    assert [row.oid for row in records] == [sealed_tip]
    assert records[0].parent_oid is None


def test_existing_clone_root_fails_before_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "source" / "ethereum-a.git"
    destination.mkdir(parents=True)
    monkeypatch.setattr(
        runner,
        "_repository_spec",
        lambda protocol: SimpleNamespace(
            remote="file:///unused",
            branch="master",
            sealed_tip="1" * 40,
            object_format="sha1",
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_git",
        lambda *args, **kwargs: pytest.fail(
            "existing root must fail before Git"
        ),
    )
    with pytest.raises(RuntimeError, match="fresh clone root already exists"):
        runner.prepare_source_repository(
            runner.Config(source_root=tmp_path / "source"),
            "ethereum",
            "a",
            runner.AccessLedger.zero(),
        )


def test_gate_one_rejects_status_checkout_shared_object_and_identity_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for protocol in ("ethereum", "bitcoin"):
        spec = runner._repository_spec(protocol)
        for replica in ("a", "b"):
            rows.append(
                {
                    "protocol": protocol,
                    "replica": replica,
                    "root_name": f"{protocol}-{replica}.git",
                    "remote": spec.remote,
                    "remote_head_symref": "refs/heads/master",
                    "remote_head_oid": "f" * 40,
                    "local_branch_ref": "refs/heads/master",
                    "local_branch_oid": "e" * 40,
                    "sealed_ref": runner.SEALED_REF,
                    "sealed_tip": spec.sealed_tip,
                    "object_format": "sha1",
                    "object_type": "commit",
                    "is_bare_repository": True,
                    "is_inside_work_tree": False,
                    "absolute_git_dir_matches_root": True,
                    "git_common_dir": ".",
                    "symbolic_head": "refs/heads/master",
                    "ref_roster": [
                        "refs/heads/master",
                        runner.SEALED_REF,
                    ],
                    "git_fsck_no_dangling": True,
                    "forbidden_paths_absent": True,
                    "shared_object_alternates": False,
                    "checkout_created": False,
                    "git_status_invoked": False,
                    "is_shallow_repository": False,
                    "object_store": {
                        "regular_files": 1,
                        "symlinks": 0,
                        "multiple_link_files": 0,
                    },
                    "disk_used_gib": 1,
                }
            )
    assert runner.gate_git_identity(rows).passed

    for mutation in (
        ("git_status_invoked", True),
        ("checkout_created", True),
        ("shared_object_alternates", True),
        ("remote", "invalid"),
    ):
        changed = copy.deepcopy(rows)
        changed[0][mutation[0]] = mutation[1]
        assert not runner.gate_git_identity(changed).passed

    changed = copy.deepcopy(rows)
    changed[0]["object_store"]["multiple_link_files"] = 1
    assert not runner.gate_git_identity(changed).passed


def test_d2_control_and_result_artifacts_never_claim_profitability() -> None:
    events, cards = _synthetic_cards_events()
    metrics = runner.build_control_metrics(events, cards)
    gate = runner.gate_control_sensitivity(metrics)
    control = runner.build_control_report(cards, metrics, gate)
    assert control["policy_id"] == "PSIM-D2"
    assert control["protocol_version"] == runner.CONTROL_REPORT_PROTOCOL
    assert control["profitability_result"] is False
    assert control["outcomes_opened"] is False

    failed_gate = runner.GateResult(
        runner.GATE_NAMES[0],
        False,
        {},
        "synthetic",
    )
    report = runner.build_result_report(
        decision="reject",
        authority={"synthetic": True},
        gates=[failed_gate],
        source_audit={"proposal_path_incidence_opened": False},
        event_count=0,
        card_count=0,
        artifacts=None,
        ledger=runner.AccessLedger.zero(),
    )
    assert report["policy_id"] == "PSIM-D2"
    assert report["terminal_action"] == runner.FAILURE_ACTION
    assert report["profitability_result"] is False
    assert report["outcomes_opened"] is False
    assert report["source_incidence_opened"] is False


def test_safe_write_rejects_escape_symlink_and_conflict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(runner, "REPO_ROOT", repo_root)
    artifact = runner._write_once_bytes("results/a.json", b"{}\n")
    assert runner._write_once_bytes("results/a.json", b"{}\n") == artifact
    with pytest.raises(RuntimeError, match="artifact differs"):
        runner._write_once_bytes("results/a.json", b"{\"x\":1}\n")
    with pytest.raises(RuntimeError, match="escapes repository"):
        runner._safe_destination(tmp_path / "outside.json")
    link = repo_root / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinked"):
        runner._safe_destination("link/out.json")


def test_source_configuration_requires_exact_root_outputs_and_no_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source_root = tmp_path / "source"
    monkeypatch.setattr(runner, "REPO_ROOT", repo_root)
    monkeypatch.setattr(runner, "DEFAULT_SOURCE_ROOT", source_root)
    runner._validate_source_configuration(
        runner.Config(source_root=source_root)
    )
    with pytest.raises(ValueError, match="source root is frozen"):
        runner._validate_source_configuration(
            runner.Config(source_root=tmp_path / "other")
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "source-link"
    link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(runner, "DEFAULT_SOURCE_ROOT", link / "nested")
    with pytest.raises(ValueError, match="symlink ancestor"):
        runner._validate_source_configuration(
            runner.Config(source_root=link / "nested")
        )


def _mock_official_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    valid_receipts: bool,
) -> runner.Config:
    repo_root = tmp_path / "repo-root"
    repo_root.mkdir()
    source_root = tmp_path / "source-root"
    monkeypatch.setattr(runner, "REPO_ROOT", repo_root)
    monkeypatch.setattr(runner, "DEFAULT_SOURCE_ROOT", source_root)
    monkeypatch.setattr(
        runner,
        "validate_execution_seal",
        lambda: {
            "seal_hash": "a" * 64,
            "shared_commit": "b" * 40,
            "runner": {},
            "tests": {},
        },
    )
    monkeypatch.setattr(
        runner,
        "static_authority",
        lambda: {"source_authority_hash": "c" * 64},
    )
    monkeypatch.setattr(
        runner,
        "_authority_report",
        lambda seal, authority: {
            "seal_hash": seal["seal_hash"],
            **authority,
        },
    )
    monkeypatch.setattr(runner, "_worktree_clean", lambda: True)
    monkeypatch.setattr(runner, "enforce_disk_guard", lambda path: 1)

    def prepare(config, protocol, replica, ledger):
        spec = runner._repository_spec(protocol)
        return {
            "protocol": protocol,
            "replica": replica,
            "root_name": f"{protocol}-{replica}.git",
            "remote": spec.remote if valid_receipts else "invalid",
            "remote_head_symref": "refs/heads/master",
            "remote_head_oid": "f" * 40,
            "local_branch_ref": "refs/heads/master",
            "local_branch_oid": "e" * 40,
            "sealed_ref": runner.SEALED_REF,
            "sealed_tip": spec.sealed_tip,
            "object_format": "sha1",
            "object_type": "commit",
            "is_bare_repository": True,
            "is_inside_work_tree": False,
            "absolute_git_dir_matches_root": True,
            "git_common_dir": ".",
            "symbolic_head": "refs/heads/master",
            "ref_roster": [
                "refs/heads/master",
                runner.SEALED_REF,
            ],
            "git_fsck_no_dangling": True,
            "forbidden_paths_absent": True,
            "shared_object_alternates": False,
            "checkout_created": False,
            "git_status_invoked": False,
            "is_shallow_repository": False,
            "object_store": {
                "regular_files": 1,
                "symlinks": 0,
                "multiple_link_files": 0,
            },
            "disk_used_gib": 1,
        }

    monkeypatch.setattr(runner, "prepare_source_repository", prepare)
    if not valid_receipts:
        monkeypatch.setattr(
            runner,
            "collect_commit_chain",
            lambda *args, **kwargs: pytest.fail(
                "first failed gate must stop without repair"
            ),
        )
        return runner.Config(source_root=source_root)

    synthetic = runner.synthetic_events()

    def chain(repo, protocol, ledger):
        spec = runner._repository_spec(protocol)
        return [
            runner.CommitRecord(
                protocol=protocol,
                oid=spec.sealed_tip,
                tree_oid="1" * 40,
                parent_oid=None,
                first_parent_index=0,
                committer_epoch=1_578_614_400,
                committer_day=date(2020, 1, 10),
                effective_day=date(2020, 1, 10),
            )
        ]

    def proposal_groups(repo, records, ledger):
        protocol = records[0].protocol
        event = next(
            row for row in synthetic if row.protocol == protocol
        )
        return (
            [
                runner.ProposalGroup(
                    protocol=protocol,
                    proposal_number=event.proposal_number,
                    commit_oid=event.commit_oid,
                    first_parent_index=event.first_parent_index,
                    committer_day=event.committer_day,
                    effective_day=event.effective_day,
                    old_path=event.old_path,
                    new_path=event.new_path,
                    old_blob_oid=event.old_blob_oid,
                    new_blob_oid=event.new_blob_oid,
                    event_type=event.event_type,
                    event_id=event.event_id,
                )
            ],
            [],
        )

    def materialize(repo, groups, ledger):
        protocol = groups[0].protocol
        return [row for row in synthetic if row.protocol == protocol]

    monkeypatch.setattr(runner, "collect_commit_chain", chain)
    monkeypatch.setattr(runner, "collect_proposal_groups", proposal_groups)
    monkeypatch.setattr(runner, "materialize_events", materialize)
    return runner.Config(source_root=source_root)


def test_official_run_stops_at_gate_one_and_publishes_only_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _mock_official_source(
        monkeypatch,
        tmp_path,
        valid_receipts=False,
    )
    report = runner.run_official(config)
    assert report["decision"] == "reject"
    assert report["first_failure"] == {
        "gate_id": 1,
        "name": runner.GATE_NAMES[0],
    }
    assert report["source_incidence_opened"] is False
    assert report["source_audit"]["git_status_invoked"] is False
    assert report["source_audit"]["checkout_created"] is False
    assert (runner.REPO_ROOT / config.rejection_path).is_file()
    assert not any(
        (runner.REPO_ROOT / path).exists()
        for path in (
            config.result_path,
            config.events_path,
            config.cards_path,
            config.controls_path,
            runner.RUN_LOCK_PATH,
        )
    )


def test_official_run_replays_all_gates_and_atomically_publishes_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _mock_official_source(
        monkeypatch,
        tmp_path,
        valid_receipts=True,
    )
    report = runner.run_official(config)
    replay = runner.run_official(config)
    assert report["decision"] == "pass"
    assert replay == report
    assert [row["name"] for row in report["gates"]] == list(
        runner.GATE_NAMES
    )
    assert all(row["passed"] for row in report["gates"])
    assert report["profitability_result"] is False
    assert report["outcomes_opened"] is False
    assert report["source_audit"]["source_traversal_ref"] == (
        runner.SEALED_REF
    )
    assert not any(
        report["access_ledger"][name]
        for name in runner.FORBIDDEN_ACCESS_FIELDS
    )
    for entry in report["artifacts"].values():
        artifact = runner.REPO_ROOT / entry["path"]
        assert artifact.is_file()
        assert runner.sha256_file(artifact) == entry["sha256"]
    assert not (runner.REPO_ROOT / runner.RUN_LOCK_PATH).exists()


def test_publication_failure_becomes_gate_thirteen_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _mock_official_source(
        monkeypatch,
        tmp_path,
        valid_receipts=True,
    )
    monkeypatch.setattr(
        runner,
        "_publish_pass_group",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FileExistsError("synthetic publication race")
        ),
    )
    report = runner.run_official(config)
    assert report["decision"] == "reject"
    assert report["first_failure"] == {
        "gate_id": 13,
        "name": runner.GATE_NAMES[12],
    }
    assert report["gates"][-1]["passed"] is False
    assert report["error"] == {"type": "FileExistsError"}
    assert (runner.REPO_ROOT / config.rejection_path).is_file()
    assert not (runner.REPO_ROOT / runner.RUN_LOCK_PATH).exists()


def test_gate_identity_detects_card_hash_tampering() -> None:
    _events, cards = _synthetic_cards_events()
    tampered = dataclasses.replace(
        cards[0],
        local_payload_sha256="0" * 64,
    )
    gate = runner.gate_daily_cards([tampered, *cards[1:]])
    assert not gate.passed


def test_execution_seal_creation_fails_closed_while_worktree_is_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_worktree_clean", lambda: False)
    with pytest.raises(RuntimeError, match="clean worktree"):
        runner.build_execution_seal()
