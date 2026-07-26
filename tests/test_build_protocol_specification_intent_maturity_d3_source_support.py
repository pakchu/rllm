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
    build_protocol_specification_intent_maturity_d3_source_support as runner,
)
from training import (
    build_protocol_specification_intent_maturity_source_support as core,
)
from training import (
    preregister_protocol_specification_intent_maturity_d3 as prereg,
)


def _run(
    *arguments: str,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if environment:
        env.update(environment)
    return subprocess.run(
        list(arguments),
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _synthetic_origin(tmp_path: Path) -> tuple[Path, str, str]:
    work = tmp_path / "work"
    work.mkdir()
    git = str(runner.GIT_BINARY)
    _run(git, "init", "-b", "master", cwd=work)
    _run(git, "config", "user.name", "PSIM D3 Test", cwd=work)
    _run(
        git,
        "config",
        "user.email",
        "psim-d3@example.test",
        cwd=work,
    )
    proposal = work / "EIPS" / "eip-1.md"
    proposal.parent.mkdir()
    proposal.write_text(
        "---\neip: 1\ntitle: First\n---\n# Abstract\none\n",
        encoding="utf-8",
    )
    _run(git, "add", ".", cwd=work)
    _run(
        git,
        "commit",
        "-m",
        "first",
        cwd=work,
        environment={
            "GIT_AUTHOR_DATE": "2020-01-02T00:00:00Z",
            "GIT_COMMITTER_DATE": "2020-01-02T00:00:00Z",
        },
    )
    sealed_tip = _run(git, "rev-parse", "HEAD", cwd=work).stdout.strip()
    proposal.write_text(
        "---\neip: 1\ntitle: Second\n---\n# Abstract\ntwo\n",
        encoding="utf-8",
    )
    _run(git, "add", ".", cwd=work)
    _run(
        git,
        "commit",
        "-m",
        "second",
        cwd=work,
        environment={
            "GIT_AUTHOR_DATE": "2020-01-03T00:00:00Z",
            "GIT_COMMITTER_DATE": "2020-01-03T00:00:00Z",
        },
    )
    branch_tip = _run(git, "rev-parse", "HEAD", cwd=work).stdout.strip()
    origin = tmp_path / "origin.git"
    _run(git, "clone", "--bare", str(work), str(origin), cwd=tmp_path)
    _run(git, "config", "uploadpack.allowFilter", "true", cwd=origin)
    _run(
        git,
        "config",
        "uploadpack.allowAnySHA1InWant",
        "true",
        cwd=origin,
    )
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
        raise AssertionError("PSIM-D3 self-check must not invoke Git/source")

    for module in (runner, core):
        monkeypatch.setattr(module, "_run_git", fail)
        monkeypatch.setattr(module, "_git_text", fail)
    monkeypatch.setattr(core, "_cat_file_batch", fail)

    payload = runner.build_self_check_manifest()

    assert payload["failed"] == []
    assert payload["policy_id"] == "PSIM-D3"
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
    assert payload["transport_probe"][
        "no_lazy_fetch_semantic_probe_passed"
    ] is True
    assert payload["transport_probe"]["single_fetch_invocations"] == 1
    assert payload["transport_probe"]["access_boundary"] == {
        "official_eip_bip_source_accessed": False,
        "market_data_accessed": False,
        "model_accessed": False,
        "outcomes_accessed": False,
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
    assert 'subprocess.run(\\n        ["git"' not in source
    assert "core._cat_file_batch(" not in source
    assert "core.collect_proposal_groups" not in source
    assert "core.materialize_events" not in source


def test_authority_constants_bind_d3_preregistration_and_d1_core() -> None:
    assert runner.POLICY_ID == "PSIM-D3"
    assert runner.PREREGISTRATION_COMMIT == (
        "1760d5945f0c8adc90ea667a21cbf6201eb5567e"
    )
    assert runner.PREREGISTRATION_SHA256 == (
        "332743f25d5be45ce4d022c67758051c01297f4cc18ccdf2138be75b5ef159ab"
    )
    assert runner.PREREGISTRATION_MANIFEST_HASH == (
        "d87358780df573bde11a317bf2e56f0ce044b3fc2fad3a28ef6e154d64023d86"
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
    assert runner.GIT_BINARY == Path("/usr/bin/git")
    assert runner.sha256_file(runner.GIT_BINARY) == prereg.GIT_BINARY_SHA256
    assert runner.sha256_file(runner.TRANSPORT_PROBE_PATH) == (
        runner.TRANSPORT_PROBE_SHA256
    )


def test_preregistration_loader_replays_exact_frozen_manifest() -> None:
    payload = runner._load_preregistration()
    assert payload == prereg.build_preregistration()
    assert payload["manifest_hash"] == runner.PREREGISTRATION_MANIFEST_HASH
    assert payload["candidate"]["id"] == runner.POLICY_ID
    assert payload["inheritance_proof"][
        "all_other_contract_paths_byte_equal"
    ] is True
    probe = runner._load_transport_probe()
    assert probe["result_hash"] == runner.TRANSPORT_PROBE_RESULT_HASH


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


def test_all_local_git_executors_use_exact_binary_and_no_lazy_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []
    original = subprocess.run

    def record(arguments, **kwargs):
        calls.append((list(arguments), dict(kwargs["env"])))
        return original(arguments, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", record)
    completed = runner._run_git(
        ["--version"],
        ledger=runner.AccessLedger.zero(),
    )
    assert completed.returncode == 0
    assert runner._git_output("--version") == prereg.GIT_VERSION
    assert runner._git_blob_sha256(
        runner.PREREGISTRATION_COMMIT,
        runner.PREREGISTRATION_PATH,
    ) == runner.PREREGISTRATION_SHA256
    assert [row[0] for row in calls] == [
        ["/usr/bin/git", "--version"],
        ["/usr/bin/git", "--version"],
        [
            "/usr/bin/git",
            "show",
            (
                f"{runner.PREREGISTRATION_COMMIT}:"
                f"{runner.PREREGISTRATION_PATH.as_posix()}"
            ),
        ],
    ]
    assert all(
        environment["GIT_NO_LAZY_FETCH"] == "1"
        for _arguments, environment in calls
    )


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
    assert receipt["sealed_ref"] == "refs/psim-d3/sealed-tip"
    assert receipt["ref_roster"] == [
        "refs/heads/master",
        "refs/psim-d3/sealed-tip",
    ]
    assert receipt["is_bare_repository"] is True
    assert receipt["is_inside_work_tree"] is False
    assert receipt["forbidden_paths_absent"] is True
    assert receipt["git_status_invoked"] is False
    assert receipt["checkout_created"] is False
    assert receipt["fetch_head_absent"] is True
    assert receipt["object_store"]["regular_files"] > 0
    assert receipt["object_store"]["symlinks"] == 0
    assert receipt["object_store"]["multiple_link_files"] == 0
    assert ledger.network_commands == 3
    sealed_fetch = next(
        call
        for call in calls
        if "fetch" in call and spec.sealed_tip in call
    )
    assert "--no-write-fetch-head" in sealed_fetch


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


def test_gate_three_and_four_use_exact_git_and_one_batch_hydration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    origin, sealed_tip, _branch_tip = _synthetic_origin(tmp_path)
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
    repo = runner.clone_path(config, "ethereum", "a")
    tree_oid = _run(
        str(runner.GIT_BINARY),
        "rev-parse",
        f"{sealed_tip}^{{tree}}",
        cwd=tmp_path / "work",
    ).stdout.strip()
    record = runner.CommitRecord(
        protocol="ethereum",
        oid=sealed_tip,
        tree_oid=tree_oid,
        parent_oid=None,
        first_parent_index=0,
        committer_epoch=1_577_923_200,
        committer_day=date(2020, 1, 2),
        effective_day=date(2020, 1, 2),
    )
    groups, issues = runner.proposal_groups_for_commit(
        repo,
        record,
        ledger,
    )
    assert issues == []
    assert len(groups) == 1
    requested = {
        oid
        for group in groups
        for oid in (group.old_blob_oid, group.new_blob_oid)
        if oid is not None
    }
    assert requested
    assert not requested.intersection(runner._local_objects(repo, ledger))

    network_before = ledger.network_commands
    receipts: list[dict] = []
    events = runner.materialize_events(
        repo,
        groups,
        ledger,
        receipts,
    )
    assert ledger.network_commands == network_before + 1
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["fetch_invocations"] == 1
    assert receipt["requested_blob_count"] == len(requested)
    assert receipt["new_pack_count"] >= 1
    assert receipt["new_promisor_count"] == receipt["new_pack_count"]
    assert receipt["new_loose_object_count"] == 0
    assert receipt["new_total_object_count"] == len(requested)
    assert receipt["maintenance_child_processes"] == 0
    assert receipt["post_read_fetch_child_processes"] == 0
    assert receipt["post_read_object_store_unchanged"] is True
    assert receipt["fetch_head_absent"] is True

    reference_ledger = core.AccessLedger.zero()
    expected = core.materialize_events(
        tmp_path / "work",
        groups,
        reference_ledger,
    )
    assert [runner.event_row(row) for row in events] == [
        core.event_row(row) for row in expected
    ]


def test_post_hydration_event_builder_is_structurally_d1_identical() -> None:
    d1_tree = ast.parse(
        (core.REPO_ROOT / core.RUNNER_PATH).read_text(encoding="utf-8")
    )
    d3_tree = ast.parse(
        (runner.REPO_ROOT / runner.RUNNER_PATH).read_text(encoding="utf-8")
    )

    def function(tree: ast.Module, name: str) -> ast.FunctionDef:
        return next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    d1_body = function(d1_tree, "materialize_events").body
    feature_index = next(
        index
        for index, statement in enumerate(d1_body)
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == "features"
    )
    d1_semantics = d1_body[feature_index:]
    d3_semantics = function(
        d3_tree,
        "_materialize_events_from_raw",
    ).body

    class NormalizeCoreReferences(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute):
            node = self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == "core":
                if node.attr == "prereg":
                    return ast.copy_location(
                        ast.Name(id="prereg", ctx=ast.Load()),
                        node,
                    )
                return ast.copy_location(
                    ast.Name(id=node.attr, ctx=ast.Load()),
                    node,
                )
            return node

    normalized = NormalizeCoreReferences()
    assert ast.dump(
        ast.Module(
            body=[normalized.visit(node) for node in d3_semantics],
            type_ignores=[],
        ),
        include_attributes=False,
    ) == ast.dump(
        ast.Module(
            body=d1_semantics,
            type_ignores=[],
        ),
        include_attributes=False,
    )


def test_materializer_rejects_out_of_window_before_hydration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    event = runner.synthetic_events()[0]
    group = runner.ProposalGroup(
        protocol=event.protocol,
        proposal_number=event.proposal_number,
        commit_oid=event.commit_oid,
        first_parent_index=event.first_parent_index,
        committer_day=date(2019, 12, 31),
        effective_day=date(2019, 12, 31),
        old_path=event.old_path,
        new_path=event.new_path,
        old_blob_oid=event.old_blob_oid,
        new_blob_oid=event.new_blob_oid,
        event_type=event.event_type,
        event_id=event.event_id,
    )
    monkeypatch.setattr(
        runner,
        "_hydrate_blob_batch",
        lambda *args, **kwargs: pytest.fail(
            "out-of-window groups must fail before hydration"
        ),
    )
    ledger = runner.AccessLedger.zero()
    with pytest.raises(RuntimeError, match="pre-2020"):
        runner.materialize_events(tmp_path, [group], ledger)
    assert ledger.pre_2020_proposal_blobs_opened == 1
    assert ledger.network_commands == 0


def test_hydration_failure_emits_hashed_forensic_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    event = next(
        row
        for row in runner.synthetic_events()
        if row.effective_day >= date(2020, 1, 1)
    )
    group = runner.ProposalGroup(
        protocol=event.protocol,
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

    def fail(repo, object_ids, ledger, progress):
        progress.update(
            {
                "repository_root_name": "ethereum-a.git",
                "fetch_invocations": 1,
                "stage": "post_hydration_inventory",
            }
        )
        raise RuntimeError("synthetic object delta mismatch")

    monkeypatch.setattr(runner, "_hydrate_blob_batch", fail)
    receipts: list[dict] = []
    with pytest.raises(RuntimeError, match="object delta mismatch"):
        runner.materialize_events(
            tmp_path,
            [group],
            runner.AccessLedger.zero(),
            receipts,
        )
    assert len(receipts) == 1
    receipt = receipts[0]
    core_receipt = {
        key: value for key, value in receipt.items() if key != "receipt_hash"
    }
    assert receipt["fetch_invocations"] == 1
    assert receipt["stage"] == "post_hydration_inventory"
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["receipt_hash"] == runner.canonical_hash(core_receipt)


def test_hydration_delta_accepts_multiple_promisor_packs() -> None:
    before = {
        "packs": ["pack-base.pack"],
        "promisors": ["pack-base.promisor"],
        "loose_objects": [],
        "objects": {"0" * 40: "commit"},
    }
    after = {
        "packs": ["pack-a.pack", "pack-b.pack", "pack-base.pack"],
        "promisors": [
            "pack-a.promisor",
            "pack-b.promisor",
            "pack-base.promisor",
        ],
        "loose_objects": [],
        "objects": {
            "0" * 40: "commit",
            "1" * 40: "blob",
            "2" * 40: "blob",
        },
    }
    result = runner._validate_hydration_delta(
        before=before,
        after=after,
        new_pack_objects={
            "pack-a.pack": {"1" * 40: "blob"},
            "pack-b.pack": {"2" * 40: "blob"},
        },
        requested=("1" * 40, "2" * 40),
    )
    assert result["new_pack_count"] == 2
    assert result["new_promisor_count"] == 2
    assert result["new_total_object_count"] == 2


@pytest.mark.parametrize(
    ("after_objects", "new_pack_objects", "promisors", "loose"),
    [
        (
            {
                "0" * 40: "commit",
                "1" * 40: "blob",
                "3" * 40: "blob",
            },
            {
                "pack-a.pack": {
                    "1" * 40: "blob",
                    "3" * 40: "blob",
                }
            },
            ["pack-a.promisor", "pack-base.promisor"],
            [],
        ),
        (
            {"0" * 40: "commit"},
            {"pack-a.pack": {}},
            ["pack-a.promisor", "pack-base.promisor"],
            [],
        ),
        (
            {"0" * 40: "commit", "1" * 40: "blob"},
            {"pack-a.pack": {"1" * 40: "blob"}},
            ["pack-base.promisor"],
            [],
        ),
        (
            {"0" * 40: "commit", "1" * 40: "blob"},
            {"pack-a.pack": {"1" * 40: "blob"}},
            ["pack-a.promisor", "pack-base.promisor"],
            ["11/" + "1" * 38],
        ),
    ],
)
def test_hydration_delta_fails_closed_on_boundary_changes(
    after_objects: dict[str, str],
    new_pack_objects: dict[str, dict[str, str]],
    promisors: list[str],
    loose: list[str],
) -> None:
    with pytest.raises(RuntimeError):
        runner._validate_hydration_delta(
            before={
                "packs": ["pack-base.pack"],
                "promisors": ["pack-base.promisor"],
                "loose_objects": [],
                "objects": {"0" * 40: "commit"},
            },
            after={
                "packs": ["pack-a.pack", "pack-base.pack"],
                "promisors": promisors,
                "loose_objects": loose,
                "objects": after_objects,
            },
            new_pack_objects=new_pack_objects,
            requested=("1" * 40,),
        )


def test_trace_and_post_read_boundaries_fail_closed(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps({"event": "child_start", "child_id": 1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="trace child argv is ambiguous"):
        runner._trace_child_arguments(trace)

    snapshot = {
        "fetch_head_absent": True,
        "loose_objects": [],
        "objects": {"1" * 40: "blob"},
        "packs": ["pack-a.pack"],
        "promisors": ["pack-a.promisor"],
        "refs": ["refs/heads/master " + "0" * 40],
    }
    runner._assert_post_read_invariant(snapshot, dict(snapshot))
    changed = dict(snapshot)
    changed["packs"] = ["pack-a.pack", "pack-b.pack"]
    with pytest.raises(
        RuntimeError,
        match="post-hydration object store changed",
    ):
        runner._assert_post_read_invariant(snapshot, changed)


def test_cat_file_parser_is_length_delimited_and_fails_closed() -> None:
    oid = "1" * 40
    raw = f"{oid} blob 4\n".encode("ascii") + b"a\nb\n\n"
    assert runner._parse_cat_file_batch(
        raw,
        [oid],
        expected_type="blob",
    ) == [(oid, b"a\nb\n")]
    for malformed in (
        raw[:-1],
        raw + b"x",
        raw.replace(b" blob ", b" tree "),
    ):
        with pytest.raises(RuntimeError):
            runner._parse_cat_file_batch(
                malformed,
                [oid],
                expected_type="blob",
            )


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
                    "fetch_head_absent": True,
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
        ("fetch_head_absent", False),
        ("shared_object_alternates", True),
        ("remote", "invalid"),
    ):
        changed = copy.deepcopy(rows)
        changed[0][mutation[0]] = mutation[1]
        assert not runner.gate_git_identity(changed).passed

    changed = copy.deepcopy(rows)
    changed[0]["object_store"]["multiple_link_files"] = 1
    assert not runner.gate_git_identity(changed).passed


def test_d3_control_and_result_artifacts_never_claim_profitability() -> None:
    events, cards = _synthetic_cards_events()
    metrics = runner.build_control_metrics(events, cards)
    gate = runner.gate_control_sensitivity(metrics)
    control = runner.build_control_report(cards, metrics, gate)
    assert control["policy_id"] == "PSIM-D3"
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
    assert report["policy_id"] == "PSIM-D3"
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
            "fetch_head_absent": True,
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

    def materialize(repo, groups, ledger, receipt_sink=None):
        protocol = groups[0].protocol
        assert receipt_sink is not None
        receipt = {
            "repository_root_name": Path(repo).name,
            "fetch_invocations": 1,
        }
        receipt_sink.append(
            {
                **receipt,
                "receipt_hash": runner.canonical_hash(receipt),
            }
        )
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
    assert len(
        report["source_audit"]["batch_hydration_receipts"]
    ) == 4
    assert (
        report["source_audit"]["batch_hydration_receipts_sha256"]
        == runner.canonical_hash(
            report["source_audit"]["batch_hydration_receipts"]
        )
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
