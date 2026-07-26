from __future__ import annotations

import ast
import dataclasses
import gzip
import json
import os
import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import (
    build_protocol_specification_intent_maturity_d4_source_support as d4_runner,
)
from training import (
    build_protocol_specification_intent_maturity_d5_source_support as runner,
)
from training import (
    preregister_protocol_specification_intent_maturity_d5 as prereg,
)
from training import (
    probe_protocol_specification_intent_maturity_d5_event_semantics
    as semantics_probe,
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
    _run(git, "config", "user.name", "PSIM D5 Test", cwd=work)
    _run(
        git,
        "config",
        "user.email",
        "psim-d5@example.test",
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


def _blob(raw: bytes, proposal_number: int = 20):
    oid = runner.core.git_object_sha1("blob", raw)
    return semantics_probe.decode_blob_d5(
        "ethereum", proposal_number, oid, raw
    )


def _group(
    *,
    old_raw: bytes | None,
    new_raw: bytes | None,
    proposal_number: int = 20,
    event_type: str = "UPDATE",
) -> tuple[runner.ProposalGroup, dict[str, bytes]]:
    old_oid = (
        None
        if old_raw is None
        else runner.core.git_object_sha1("blob", old_raw)
    )
    new_oid = (
        None
        if new_raw is None
        else runner.core.git_object_sha1("blob", new_raw)
    )
    commit_oid = "1" * 40
    old_path = None if old_raw is None else f"EIPS/eip-{proposal_number}.md"
    new_path = None if new_raw is None else f"EIPS/eip-{proposal_number}.md"
    group = runner.ProposalGroup(
        protocol="ethereum",
        proposal_number=proposal_number,
        commit_oid=commit_oid,
        first_parent_index=0,
        committer_day=date(2023, 10, 25),
        effective_day=date(2023, 10, 25),
        old_path=old_path,
        new_path=new_path,
        old_blob_oid=old_oid,
        new_blob_oid=new_oid,
        event_type=event_type,
        event_id=runner._event_id(
            "ethereum", commit_oid, proposal_number, old_oid, new_oid
        ),
    )
    raw = {}
    if old_oid is not None and old_raw is not None:
        raw[old_oid] = old_raw
    if new_oid is not None and new_raw is not None:
        raw[new_oid] = new_raw
    return group, raw


def _semantics_receipt(
    protocol: str,
    counts: dict[str, int],
    *,
    oid_manifest_sha256: str = "a" * 64,
) -> dict[str, object]:
    core = {
        "protocol_version": "psim_d5_blob_semantics_receipt_v1",
        "protocol": protocol,
        "decoded_blob_count": sum(counts.values()),
        "oid_manifest_sha256": oid_manifest_sha256,
        "class_counts": counts,
        "total_fraction": "1.0",
        "unknown_class_count": 0,
    }
    return {**core, "receipt_hash": runner.canonical_hash(core)}


def _gate4_hydration_receipts(
    semantics: dict[tuple[str, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (protocol, replica), semantic in sorted(semantics.items()):
        core = {
            "repository_root_name": f"{protocol}-{replica}.git",
            "requested_blob_count": semantic["decoded_blob_count"],
            "oid_manifest_sha256": semantic["oid_manifest_sha256"],
        }
        rows.append({**core, "receipt_hash": runner.canonical_hash(core)})
    return rows


def _gate4_ledger(
    semantics: dict[tuple[str, str], dict[str, object]],
) -> runner.AccessLedger:
    total = sum(int(row["decoded_blob_count"]) for row in semantics.values())
    ledger = runner.AccessLedger.zero()
    ledger.proposal_blobs_opened = total
    ledger.proposal_text_rows_opened = total
    return ledger


def _rehash_receipt(receipt: dict[str, object]) -> dict[str, object]:
    core = {
        key: value for key, value in receipt.items() if key != "receipt_hash"
    }
    return {**core, "receipt_hash": runner.canonical_hash(core)}


def test_self_check_is_synthetic_only_and_opens_no_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("self-check crossed a source or transport boundary")

    for name in (
        "prepare_source_repository",
        "collect_commit_chain",
        "collect_proposal_groups",
        "materialize_events",
        "_run_git",
        "_git_text",
    ):
        monkeypatch.setattr(runner, name, forbidden)

    payload = runner.build_self_check_manifest()
    assert payload["failed"] == []
    assert payload["policy_id"] == "PSIM-D5"
    assert payload["network_calls"] == 0
    assert payload["git_commands"] == 0
    assert payload["source_event_rows_opened"] == 0
    assert payload["official_source_opened"] is False
    assert payload["outcomes_opened"] is False
    assert payload["forbidden_access"] == runner.AccessLedger.zero().snapshot()


def test_runner_import_surface_is_outcome_blind() -> None:
    tree = ast.parse(runner.repository_path(runner.RUNNER_PATH).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden = {
        "torch",
        "transformers",
        "ccxt",
        "pandas",
        "numpy",
        "requests",
        "httpx",
        "aiohttp",
    }
    assert imported.isdisjoint(forbidden)
    assert runner.DEFAULT_SOURCE_ROOT == Path("/tmp/psim-d5-source")
    assert runner.SEALED_REF == "refs/psim-d5/sealed-tip"
    assert runner.GATE_NAMES == tuple(runner.core.GATE_NAMES)


def test_authority_constants_bind_exact_d5_prereg_and_d4_predecessor() -> None:
    assert runner.PREREGISTRATION_COMMIT == (
        "4e2b403c1f369bf2e76b5edeb1e4166b9d2f8779"
    )
    assert runner.PREREGISTRATION_SHA256 == (
        "11465540d59181bc48ea28c5164579847cbd936bf005c69d874ec2c873c949b9"
    )
    assert runner.PREREGISTRATION_MANIFEST_HASH == (
        "f08eeb300fceb906cdcde485b4bce184c48d4cb14a1cd9028046e0c21a287309"
    )
    assert runner.D4_IMPLEMENTATION_COMMIT == (
        "2d3216d5a144ba8eb694270301231850f0e015ca"
    )
    assert runner.D4_RUNNER_SHA256 == runner.sha256_file(
        runner.D4_RUNNER_PATH
    )
    assert runner.D4_TEST_SHA256 == runner.sha256_file(runner.D4_TEST_PATH)
    assert runner.SEMANTICS_PROBE_RESULT_HASH == (
        "467f4272bc7276879c0087662a70d99c57d9cef421647f1a679e2fce65de4871"
    )


def test_preregistration_and_probe_loaders_replay_frozen_authority() -> None:
    registration = runner._load_preregistration()
    semantics = runner._load_semantics_probe()
    transport = runner._load_transport_probe()
    assert registration == prereg.build_preregistration()
    assert registration["manifest_hash"] == runner.PREREGISTRATION_MANIFEST_HASH
    assert semantics == semantics_probe.build_probe()
    assert semantics["result_hash"] == runner.SEMANTICS_PROBE_RESULT_HASH
    assert transport["result_hash"] == runner.TRANSPORT_PROBE_RESULT_HASH


def test_d5_transport_surface_tracks_d4_contract() -> None:
    assert runner.CLONE_ARGUMENTS == d4_runner.CLONE_ARGUMENTS
    assert runner.GIT_BINARY == d4_runner.GIT_BINARY
    assert runner.HYDRATION_TIMEOUT_SECONDS == 1800
    assert runner.BARE_REPOSITORY_CONTRACT["checkout_allowed"] is False
    assert runner.BARE_REPOSITORY_CONTRACT[
        "shared_objects_or_cache_allowed"
    ] is False
    for name in (
        "prepare_source_repository",
        "collect_commit_chain",
        "proposal_groups_for_commit",
        "collect_proposal_groups",
        "_parse_cat_file_batch",
        "_cat_file_batch_local",
        "_hydrate_blob_batch",
        "_validate_hydration_delta",
    ):
        assert callable(getattr(runner, name))


def test_hydration_trace_namespace_is_d5_local(tmp_path: Path) -> None:
    repo = tmp_path / "ethereum-a.git"
    repo.mkdir()
    trace = runner._fresh_trace_path(repo, "fetch")
    assert trace.parent == tmp_path / ".psim-d5-traces"
    assert not (tmp_path / ".psim-d3-traces").exists()
    assert not (tmp_path / ".psim-d4-traces").exists()


def test_valid_and_known_invalid_events_use_d5_model_field() -> None:
    valid = (
        b"---\neip: 20\nstatus: Draft\nrequires: 1\n---\n"
        b"# Abstract\nOld.\n"
    )
    invalid = (
        b"---\neip: 20\nstatus: Draft\nstatus: Draft\n---\n"
        b"# Abstract\nNew.\n"
    )
    group, raw = _group(old_raw=valid, new_raw=invalid)
    sink: list[dict] = []
    events = runner._materialize_events_from_raw(
        [group], raw, runner.AccessLedger.zero(), sink
    )
    assert len(events) == 1
    event = events[0]
    row = runner.event_row(event)
    assert "intent_text" not in row
    assert event.model_visibility == "MODEL_VISIBLE"
    assert event.new_metadata_state == "INVALID_DUPLICATE_IDENTICAL"
    assert event.invalid_metadata_present is True
    assert event.invalid_metadata_states == ("INVALID_DUPLICATE_IDENTICAL",)
    assert event.dependency_delta_state == "UNKNOWN_INVALID_METADATA"
    assert event.dependency_edge_delta_count is None
    assert event.normalized_text_delta
    cards = runner.build_daily_cards([event])
    model_rows = [
        value
        for card in cards
        for value in runner._walk_mappings(card.local_payload)
        if "normalized_text_delta" in value
    ]
    assert model_rows
    assert all(set(value) == runner.MODEL_EVENT_PAYLOAD_FIELDS for value in model_rows)
    assert all("invalid_metadata_states" not in value for value in model_rows)
    assert sink[0]["class_counts"] == {
        "D4_DUPLICATE_IDENTICAL_HEADER": 1,
        "D4_VALID": 1,
    }
    assert sink[0]["receipt_hash"] == runner.canonical_hash(
        {key: value for key, value in sink[0].items() if key != "receipt_hash"}
    )


def test_exact_administrative_redirect_is_source_retained_but_card_hidden() -> None:
    valid = (
        b"---\neip: 20\ntitle: Prior\n---\n"
        b"# Specification\nPrior.\n"
    )
    redirect = (
        b"This file was moved to "
        b"https://github.com/ethereum/ercs/blob/master/ercs/erc-20.md\n"
    )
    group, raw = _group(old_raw=valid, new_raw=redirect)
    events = runner._materialize_events_from_raw(
        [group], raw, runner.AccessLedger.zero()
    )
    event = events[0]
    row = runner.event_row(event)
    assert row["model_visibility"] == "ADMINISTRATIVE_QUARANTINE"
    assert row["administrative_quarantined"] is True
    assert row["normalized_text_delta"] == ""
    assert row["audit_line_change_count"] > 0
    assert row["dependency_edge_delta_count"] is None
    assert row["memorization_excluded"] is True

    control = runner.synthetic_events()[0]
    with_admin = runner.build_daily_cards([event, control])
    without_admin = runner.build_daily_cards([control])
    assert [runner.card_row(card) for card in with_admin] == [
        runner.card_row(card) for card in without_admin
    ]
    assert runner.split_support_metrics(
        [event, control], with_admin
    ) == runner.split_support_metrics([control], without_admin)
    serialized = json.dumps([card.local_payload for card in with_admin])
    assert event.event_id not in {
        event_id for card in with_admin for event_id in card.event_ids
    }
    assert "intent_text" not in serialized
    assert "path_identity_hash" not in serialized
    assert "old_path" not in serialized
    assert "new_path" not in serialized


def test_unknown_grammar_fails_closed_before_event_materialization() -> None:
    valid = b"---\neip: 20\ntitle: Prior\n---\n# Abstract\nPrior.\n"
    unknown = b"not an eip preamble\n"
    group, raw = _group(old_raw=valid, new_raw=unknown)
    ledger = runner.AccessLedger.zero()
    with pytest.raises(ValueError, match="unregistered blob grammar"):
        runner._materialize_events_from_raw([group], raw, ledger)
    assert ledger.models_loaded == 0
    assert ledger.future_return_rows_read == 0
    assert ledger.trade_rows_built == 0


def test_reverse_administrative_transition_fails_closed() -> None:
    redirect = (
        b"This file was moved to "
        b"https://github.com/ethereum/ercs/blob/master/ercs/erc-20.md\n"
    )
    valid = b"---\neip: 20\ntitle: Restored\n---\n# Abstract\nRestored.\n"
    group, raw = _group(old_raw=redirect, new_raw=valid)
    with pytest.raises(ValueError, match="reverse administrative migration"):
        runner._materialize_events_from_raw(
            [group], raw, runner.AccessLedger.zero()
        )


def test_controls_preserve_model_boundary_and_pass_synthetic_battery() -> None:
    events = runner.synthetic_events()
    cards = runner.build_daily_cards(events)
    controls = runner.build_control_metrics(events, cards)
    assert tuple(controls) == tuple(runner.core.prereg.RELATION_CONTROLS)
    assert all(row["passed"] for row in controls.values())
    assert runner.gate_control_sensitivity(controls).passed
    for control in runner.core.prereg.RELATION_CONTROLS:
        transformed = runner.transform_events(events, control)
        assert len(transformed) == len(events)
        assert all("intent_text" not in runner.event_row(row) for row in transformed)


def test_gate_four_requires_exact_class_counts_and_receipt_integrity() -> None:
    ethereum = [row for row in runner.synthetic_events() if row.protocol == "ethereum"]
    bitcoin = [row for row in runner.synthetic_events() if row.protocol == "bitcoin"]
    events = {
        ("ethereum", "a"): ethereum,
        ("ethereum", "b"): ethereum,
        ("bitcoin", "a"): bitcoin,
        ("bitcoin", "b"): bitcoin,
    }
    expected = prereg.build_preregistration()["source_support_contract"][
        "ethereum_historical_blob_class_counts"
    ]
    receipts = {
        ("ethereum", "a"): _semantics_receipt("ethereum", expected),
        ("ethereum", "b"): _semantics_receipt("ethereum", expected),
        ("bitcoin", "a"): _semantics_receipt("bitcoin", {"D4_VALID": 1}),
        ("bitcoin", "b"): _semantics_receipt("bitcoin", {"D4_VALID": 1}),
    }
    hydration = _gate4_hydration_receipts(receipts)
    ledger = _gate4_ledger(receipts)
    assert runner.gate_event_parser_replay(
        events, receipts, hydration, ledger
    ).passed

    tampered = dict(receipts)
    broken = dict(tampered[("ethereum", "a")])
    broken["decoded_blob_count"] = 1
    tampered[("ethereum", "a")] = _rehash_receipt(broken)
    gate = runner.gate_event_parser_replay(
        events, tampered, hydration, ledger
    )
    assert not gate.passed
    assert "ethereum" in gate.failure

    forged_manifest = dict(receipts)
    forged = dict(forged_manifest[("bitcoin", "a")])
    forged["oid_manifest_sha256"] = "f" * 64
    forged_manifest[("bitcoin", "a")] = _rehash_receipt(forged)
    gate = runner.gate_event_parser_replay(
        events, forged_manifest, hydration, ledger
    )
    assert not gate.passed
    assert gate.metrics["bitcoin"]["hydration_manifest_binding"] is False

    extra_class = dict(receipts)
    bitcoin = dict(extra_class[("bitcoin", "a")])
    bitcoin["class_counts"] = {"D4_VALID": 1, "D5_UNKNOWN_INVALID": 0}
    extra_class[("bitcoin", "a")] = _rehash_receipt(bitcoin)
    gate = runner.gate_event_parser_replay(
        events, extra_class, hydration, ledger
    )
    assert not gate.passed
    assert gate.metrics["bitcoin"]["registered_class_roster_exact"] is False

    short_ledger = _gate4_ledger(receipts)
    short_ledger.proposal_text_rows_opened -= 1
    gate = runner.gate_event_parser_replay(
        events, receipts, hydration, short_ledger
    )
    assert not gate.passed
    assert gate.metrics["bitcoin"]["decode_ledger_matches_receipts"] is False


def test_gate_and_artifact_rows_never_emit_legacy_intent_alias() -> None:
    events = runner.synthetic_events()
    cards = runner.build_daily_cards(events)
    controls = runner.build_control_metrics(events, cards)
    control_gate = runner.gate_control_sensitivity(controls)
    config = runner.Config()
    raw, manifest = runner.build_pass_artifacts(
        config, events, cards, controls, control_gate
    )
    assert manifest["events"]["rows"] == len(events)
    event_bytes = gzip.decompress(raw[config.events_path])
    card_bytes = gzip.decompress(raw[config.cards_path])
    assert b'"intent_text"' not in event_bytes
    assert b'"intent_text"' not in card_bytes
    assert b'"normalized_text_delta"' in event_bytes
    assert b'"normalized_text_delta"' in card_bytes


def test_real_local_clone_uses_one_batch_hydration_and_d5_decode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    origin, sealed_tip, branch_tip = _synthetic_origin(tmp_path)
    spec = _local_spec(origin, sealed_tip)
    monkeypatch.setattr(runner, "_repository_spec", lambda protocol: spec)
    config = runner.Config(source_root=tmp_path / "source")
    ledger = runner.AccessLedger.zero()
    receipt = runner.prepare_source_repository(
        config, "ethereum", "a", ledger
    )
    assert receipt["sealed_tip"] == sealed_tip
    assert branch_tip != sealed_tip
    repo = runner.clone_path(config, "ethereum", "a")
    records = runner.collect_commit_chain(repo, "ethereum", ledger)
    assert [row.oid for row in records] == [sealed_tip]
    groups, issues = runner.collect_proposal_groups(repo, records, ledger)
    assert issues == []
    hydration: list[dict] = []
    semantics: list[dict] = []
    events = runner.materialize_events(
        repo, groups, ledger, hydration, semantics
    )
    assert len(hydration) == 1
    assert hydration[0]["fetch_invocations"] == 1
    assert hydration[0]["post_read_fetch_child_processes"] == 0
    assert hydration[0]["post_read_object_store_unchanged"] is True
    assert len(semantics) == 1
    assert semantics[0]["class_counts"] == {"D4_VALID": 1}
    assert semantics[0]["oid_manifest_sha256"] == hydration[0][
        "oid_manifest_sha256"
    ]
    assert len(events) == 1
    assert "ABSTRACT|ADD|one" in events[0].normalized_text_delta.splitlines()
    assert events[0].model_visibility == "MODEL_VISIBLE"


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
    event = runner.synthetic_events()[0]
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
            tmp_path, [group], runner.AccessLedger.zero(), receipts
        )
    receipt = receipts[0]
    assert receipt["receipt_hash"] == runner.canonical_hash(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    )


def test_source_configuration_is_frozen_and_symlink_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="official source root is frozen"):
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
            "ref_roster": ["refs/heads/master", runner.SEALED_REF],
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
        event = next(row for row in synthetic if row.protocol == protocol)
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

    expected = prereg.build_preregistration()["source_support_contract"][
        "ethereum_historical_blob_class_counts"
    ]

    def materialize(
        repo,
        groups,
        ledger,
        receipt_sink=None,
        semantics_sink=None,
    ):
        protocol = groups[0].protocol
        assert receipt_sink is not None
        assert semantics_sink is not None
        counts = expected if protocol == "ethereum" else {"D4_VALID": 1}
        requested = sum(counts.values())
        manifest_hash = runner.sha256_bytes(
            f"{protocol}:synthetic-manifest".encode()
        )
        receipt = {
            "repository_root_name": Path(repo).name,
            "fetch_invocations": 1,
            "requested_blob_count": requested,
            "oid_manifest_sha256": manifest_hash,
        }
        receipt_sink.append(
            {**receipt, "receipt_hash": runner.canonical_hash(receipt)}
        )
        semantics_sink.append(
            _semantics_receipt(
                protocol,
                counts,
                oid_manifest_sha256=manifest_hash,
            )
        )
        ledger.proposal_blobs_opened += requested
        ledger.proposal_text_rows_opened += requested
        return [row for row in synthetic if row.protocol == protocol]

    monkeypatch.setattr(runner, "collect_commit_chain", chain)
    monkeypatch.setattr(runner, "collect_proposal_groups", proposal_groups)
    monkeypatch.setattr(runner, "materialize_events", materialize)
    return runner.Config(source_root=source_root)


def test_official_run_stops_at_first_gate_and_publishes_only_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _mock_official_source(
        monkeypatch, tmp_path, valid_receipts=False
    )
    report = runner.run_official(config)
    assert report["decision"] == "reject"
    assert report["first_failure"] == {
        "gate_id": 1,
        "name": runner.GATE_NAMES[0],
    }
    assert report["outcomes_opened"] is False
    assert (runner.REPO_ROOT / config.rejection_path).is_file()
    assert not (runner.REPO_ROOT / runner.RUN_LOCK_PATH).exists()


def test_official_run_replays_all_gates_and_publishes_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _mock_official_source(
        monkeypatch, tmp_path, valid_receipts=True
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
    assert len(report["source_audit"]["batch_hydration_receipts"]) == 4
    assert len(report["source_audit"]["blob_semantics_receipts"]) == 4
    assert report["source_audit"]["blob_semantics_receipts_sha256"] == (
        runner.canonical_hash(
            report["source_audit"]["blob_semantics_receipts"]
        )
    )
    assert not any(
        report["access_ledger"][name]
        for name in runner.FORBIDDEN_ACCESS_FIELDS
    )
    assert not (runner.REPO_ROOT / runner.RUN_LOCK_PATH).exists()


def test_gate_identity_detects_card_hash_tampering() -> None:
    cards = runner.build_daily_cards(runner.synthetic_events())
    tampered = dataclasses.replace(cards[0], local_payload_sha256="0" * 64)
    assert not runner.gate_daily_cards([tampered, *cards[1:]]).passed


def test_execution_seal_creation_fails_closed_while_worktree_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_worktree_clean", lambda: False)
    with pytest.raises(RuntimeError, match="clean worktree"):
        runner.build_execution_seal()


def test_post_seal_verification_uses_recursion_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout="7 passed in 0.01s\n",
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    payload = runner._run_post_seal_test_verification()
    assert payload["passed"] == 7
    assert observed["argv"] == [
        ".venv/bin/pytest",
        "-q",
        runner.SEAL_TEST_PATH.as_posix(),
    ]
    assert observed["env"][runner.SEAL_TEST_CHILD_ENV] == "1"
