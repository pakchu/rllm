from __future__ import annotations

import ast
import copy
import dataclasses
import gzip
import inspect
import json
import os
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from training import (
    build_protocol_specification_intent_maturity_d6_source_support
    as d6_runner,
)
from training import (
    build_protocol_specification_intent_maturity_d7_source_support as runner,
)
from training import (
    preregister_protocol_specification_intent_maturity_d7 as prereg,
)
from training import (
    probe_protocol_specification_intent_maturity_d5_event_semantics
    as semantics_probe,
)
from training import (
    probe_protocol_specification_intent_maturity_d6_mechanism as mechanism,
)
from training import (
    probe_protocol_specification_intent_maturity_d7_bitcoin_grammar_mechanism
    as d7_mechanism,
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
    _run(git, "config", "user.name", "PSIM D7 Test", cwd=work)
    _run(
        git,
        "config",
        "user.email",
        "psim-d7@example.test",
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


def _group(
    *,
    old_raw: bytes | None,
    new_raw: bytes | None,
    proposal_number: int,
    protocol: str = "ethereum",
    first_parent_index: int = 0,
    event_type: str = "UPDATE",
    effective_day: date = date(2023, 10, 25),
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
    commit_oid = runner.sha256_bytes(
        f"commit:{proposal_number}:{first_parent_index}".encode()
    )[:40]
    proposal_path = (
        f"EIPS/eip-{proposal_number}.md"
        if protocol == "ethereum"
        else f"bip-{proposal_number:04d}.mediawiki"
    )
    old_path = (
        None
        if old_raw is None
        else proposal_path
    )
    new_path = (
        None
        if new_raw is None
        else proposal_path
    )
    group = runner.ProposalGroup(
        protocol=protocol,
        proposal_number=proposal_number,
        commit_oid=commit_oid,
        first_parent_index=first_parent_index,
        committer_day=effective_day,
        effective_day=effective_day,
        old_path=old_path,
        new_path=new_path,
        old_blob_oid=old_oid,
        new_blob_oid=new_oid,
        event_type=event_type,
        event_id=runner._event_id(
            protocol,
            commit_oid,
            proposal_number,
            old_oid,
            new_oid,
        ),
    )
    raw: dict[str, bytes] = {}
    if old_oid is not None and old_raw is not None:
        raw[old_oid] = old_raw
    if new_oid is not None and new_raw is not None:
        raw[new_oid] = new_raw
    return group, raw


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(nested))
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for nested in value:
            keys.update(_nested_keys(nested))
    return keys


def _representative_migration_episode() -> dict[str, Any]:
    census = mechanism._read_canonical_json(mechanism.D5_CENSUS_PATH)
    return copy.deepcopy(
        census["census"]["administrative_episode_census"]["representative"]
    )


def _gate_events() -> tuple[
    list[runner.ProposalEvent],
    list[runner.ProposalEvent],
    dict[int, str],
    str,
]:
    episodes: list[dict[str, Any]] = []
    authority: dict[int, str] = {}
    for proposal in range(1, 366):
        episode = mechanism._synthetic_episode(proposal)
        for index, step in enumerate(episode["steps"]):
            step["event_id"] = runner.sha256_bytes(
                f"migration:{proposal}:{index}".encode()
            )
        restoration = episode["steps"][-1]
        restoration["event_id"] = runner._event_id(
            "ethereum",
            restoration["commit_oid"],
            proposal,
            restoration["old_blob_oid"],
            restoration["new_blob_oid"],
        )
        episodes.append(episode)
        authority[proposal] = runner.canonical_hash(episode)
    manifest_hash = runner.canonical_hash(
        mechanism.migration_receipt_manifest_d6(authority)
    )
    ethereum: list[runner.ProposalEvent] = []
    for index, episode in enumerate(episodes):
        proposal = episode["proposal"]
        receipt_hash = authority[proposal]
        restoration = episode["steps"][-1]
        event = runner._synthetic_event(
            protocol="ethereum",
            proposal_number=proposal,
            effective_day=date(2023, 10, 26),
            event_type="UPDATE",
            revision=2,
            first_parent_index=index,
        )
        ethereum.append(
            dataclasses.replace(
                event,
                commit_oid=restoration["commit_oid"],
                committer_day=date.fromisoformat(
                    restoration["effective_day"]
                ),
                effective_day=date.fromisoformat(
                    restoration["effective_day"]
                ),
                event_id=restoration["event_id"],
                old_path=restoration["old_path"],
                new_path=restoration["new_path"],
                old_blob_oid=restoration["old_blob_oid"],
                new_blob_oid=restoration["new_blob_oid"],
                old_blob_sha256=restoration["old_blob_sha256"],
                new_blob_sha256=restoration["new_blob_sha256"],
                normalized_text_delta_source="",
                model_line_change_count=0,
                line_change_count=0,
                changed_sections=(),
                changed_section_count=0,
                dependency_delta_state="ADMINISTRATIVE_QUARANTINE",
                dependency_edge_delta_count=None,
                old_metadata_state="ADMINISTRATIVE_REDIRECT",
                new_metadata_state="VALID",
                model_visibility="ADMINISTRATIVE_QUARANTINE",
                administrative_quarantined=True,
                quarantine_reason=(
                    "EXACT_D5_RECEIPT_BOUND_2023_ERC_RESTORATION"
                ),
                migration_restoration_receipt={
                    "authority_receipt_hash": receipt_hash,
                    "authority_receipt_manifest_hash": manifest_hash,
                    "causal_episode": episode,
                    "causal_episode_steps": 3,
                    "protocol_version": (
                        "psim_d6_exact_migration_restoration_receipt_v1"
                    ),
                    "quarantine_reason": (
                        "EXACT_2023_ETHEREUM_ERC_MIGRATION_"
                        "EPISODE_RESTORATION"
                    ),
                },
                semantic_outcome_id="PASS_EXACT_MIGRATION_RESTORATION",
                memorization_excluded=True,
            )
        )
    for index in range(190):
        event = runner._synthetic_event(
            protocol="ethereum",
            proposal_number=100_000 + index,
            effective_day=date(2023, 11, 1),
            event_type="UPDATE",
            revision=1,
            first_parent_index=10_000 + index,
        )
        ethereum.append(
            dataclasses.replace(
                event,
                normalized_text_delta_source=(
                    "ABSTRACT|ADD|" + "x" * 8_192
                ),
            )
        )
    bitcoin = [
        runner._synthetic_event(
            protocol="bitcoin",
            proposal_number=9_001,
            effective_day=date(2023, 11, 1),
            event_type="UPDATE",
            revision=1,
            first_parent_index=20_000,
        )
    ]
    return ethereum, bitcoin, authority, manifest_hash


def _semantics_receipt(
    protocol: str,
    events: Sequence[runner.ProposalEvent],
    class_counts: Mapping[str, int],
    *,
    oid_manifest_sha256: str,
    error_outcomes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    event_rows = [runner.event_row(row) for row in events]
    outcomes = [
        {
            "event_id": row["event_id"],
            "outcome_id": row["semantic_outcome_id"],
            "passed": True,
        }
        for row in event_rows
    ] + [dict(row) for row in error_outcomes]
    errors = [row for row in outcomes if row["passed"] is False]
    transport_roster = runner._event_transport_roster(event_rows)
    migration_roster = runner._event_migration_roster(event_rows)
    assert transport_roster is not None
    assert migration_roster is not None
    core = {
        "protocol_version": "psim_d7_blob_semantics_receipt_v1",
        "protocol": protocol,
        "requested_blob_count": sum(class_counts.values()),
        "decoded_blob_count": sum(class_counts.values()),
        "decode_error_blob_count": 0,
        "oid_manifest_sha256": oid_manifest_sha256,
        "class_counts": dict(class_counts),
        "total_fraction": "1.0",
        "event_count_expected": len(outcomes),
        "event_count_materialized": len(events),
        "event_outcomes": outcomes,
        "event_outcome_counts": dict(
            sorted(Counter(row["outcome_id"] for row in outcomes).items())
        ),
        "event_outcome_roster_hash": runner.canonical_hash(outcomes),
        "semantic_error_count": len(errors),
        "semantic_error_roster_hash": runner.canonical_hash(errors),
        "model_text_transport_receipt_roster_hash": runner.canonical_hash(
            transport_roster
        ),
        "model_text_multi_chunk_event_count": sum(
            row["chunk_count"] > 1 for row in transport_roster
        ),
        "model_text_multi_chunk_event_roster_hash": runner.canonical_hash(
            sorted(
                row["event_id"]
                for row in transport_roster
                if row["chunk_count"] > 1
            )
        ),
        "max_model_text_chunk_count": max(
            (row["chunk_count"] for row in transport_roster),
            default=0,
        ),
        "migration_restoration_count": len(migration_roster),
        "migration_restoration_receipts": migration_roster,
        "migration_restoration_receipt_roster_hash": runner.canonical_hash(
            migration_roster
        ),
        "complete_event_outcome_roster": True,
        "raw_or_normalized_text_published": False,
    }
    return {**core, "receipt_hash": runner.canonical_hash(core)}


def _gate_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    error_outcomes: Sequence[Mapping[str, Any]] = (),
) -> tuple[
    dict[tuple[str, str], list[runner.ProposalEvent]],
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
    runner.AccessLedger,
]:
    ethereum, bitcoin, authority, manifest_hash = _gate_events()
    monkeypatch.setattr(
        runner,
        "_frozen_migration_authority",
        lambda: (authority, manifest_hash),
    )
    monkeypatch.setattr(
        prereg.d6,
        "D5_EPISODE_RECEIPT_MANIFEST_HASH",
        manifest_hash,
    )
    events = {
        ("ethereum", "a"): ethereum,
        ("ethereum", "b"): ethereum,
        ("bitcoin", "a"): bitcoin,
        ("bitcoin", "b"): bitcoin,
    }
    ethereum_counts = prereg.build_preregistration()[
        "source_support_contract"
    ]["ethereum_historical_blob_class_counts"]
    manifests = {
        ("ethereum", "a"): "a" * 64,
        ("ethereum", "b"): "a" * 64,
        ("bitcoin", "a"): "c" * 64,
        ("bitcoin", "b"): "c" * 64,
    }
    semantics: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in events.items():
        protocol, _replica = key
        errors = error_outcomes if key == ("ethereum", "a") else ()
        semantics[key] = _semantics_receipt(
            protocol,
            rows,
            (
                ethereum_counts
                if protocol == "ethereum"
                else runner.D7_BITCOIN_CLASS_COUNTS
            ),
            oid_manifest_sha256=manifests[key],
            error_outcomes=errors,
        )
    monkeypatch.setattr(
        prereg.d6,
        "D5_TEXT_BOUND_EVENT_ROSTER_HASH",
        semantics[("ethereum", "a")][
            "model_text_multi_chunk_event_roster_hash"
        ],
    )
    hydration: list[dict[str, Any]] = []
    for (protocol, replica), receipt in sorted(semantics.items()):
        core = {
            "repository_root_name": f"{protocol}-{replica}.git",
            "requested_blob_count": receipt["requested_blob_count"],
            "oid_manifest_sha256": receipt["oid_manifest_sha256"],
        }
        hydration.append(
            {**core, "receipt_hash": runner.canonical_hash(core)}
        )
    total = sum(row["requested_blob_count"] for row in semantics.values())
    ledger = runner.AccessLedger.zero()
    ledger.proposal_blobs_opened = total
    ledger.proposal_text_rows_opened = total
    return events, semantics, hydration, ledger


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
    assert payload["policy_id"] == "PSIM-D7"
    assert payload["network_calls"] == 0
    assert payload["git_commands"] == 0
    assert payload["source_event_rows_opened"] == 0
    assert payload["official_source_opened"] is False
    assert payload["outcomes_opened"] is False
    assert payload["forbidden_access"] == runner.AccessLedger.zero().snapshot()


def test_runner_namespace_authority_and_import_surface_are_frozen() -> None:
    tree = ast.parse(runner.repository_path(runner.RUNNER_PATH).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint(
        {
            "aiohttp",
            "ccxt",
            "httpx",
            "numpy",
            "pandas",
            "requests",
            "torch",
            "transformers",
        }
    )
    assert runner.DEFAULT_SOURCE_ROOT == Path("/tmp/psim-d7-source")
    assert runner.DEFAULT_SOURCE_ROOT != d6_runner.DEFAULT_SOURCE_ROOT
    assert runner.SEALED_REF == "refs/psim-d7/sealed-tip"
    assert runner.PREREGISTRATION_COMMIT == (
        "b26c92acf053553c5f4b02eee6b6229229d7e737"
    )
    assert runner.D6_IMPLEMENTATION_COMMIT == (
        "ff7aea465190779eaf6e79619f7756171054710d"
    )
    assert runner._load_preregistration() == prereg.build_preregistration()
    assert runner._load_mechanism_probe() == d7_mechanism.build_probe()
    assert runner.D7_BITCOIN_CLASS_COUNTS == {
        "D4_VALID": 426,
        "D7_BIP_LATER_HEADER": 7,
        "D7_BIP_PREFIXED_DEPENDENCY": 1,
    }
    assert d7_mechanism._load_d6_census_binding()[
        "grammar_class_counts"
    ] == {
        "BIP_LATER_EXACT_PRE_HEADER_AFTER_NONHEADER_PREFIX": 7,
        "BIP_PREFIXED_DECIMAL_DEPENDENCY_TOKEN": 1,
        "D4_VALID": 426,
    }
    assert (
        runner._load_transport_probe()["result_hash"]
        == runner.TRANSPORT_PROBE_RESULT_HASH
    )


def test_bitcoin_overlay_preserves_d6_initial_and_supports_exact_d7_delta(
) -> None:
    proposal = 900
    initial_raw = d7_mechanism._synthetic_bip(
        proposal,
        dependency="899",
    )
    initial_oid = runner.core.git_object_sha1("blob", initial_raw)
    inherited = semantics_probe.decode_blob_d5(
        "bitcoin",
        proposal,
        initial_oid,
        initial_raw,
    )
    observed = runner.parse_blob_features(
        "bitcoin",
        proposal,
        initial_oid,
        initial_raw,
    )
    assert isinstance(observed, runner.D7BitcoinBlobSemantics)
    for name in semantics_probe.BlobSemantics.__dataclass_fields__:
        assert getattr(observed, name) == getattr(inherited, name)
    assert observed.grammar_anchor == d7_mechanism.INITIAL_ANCHOR
    assert observed.dependency_token_mode == "BARE_ONLY"

    later_raw = d7_mechanism._synthetic_bip(
        proposal,
        prefix="Retained preface\n\n",
        dependency="899",
    )
    later = runner.parse_blob_features(
        "bitcoin",
        proposal,
        runner.core.git_object_sha1("blob", later_raw),
        later_raw,
    )
    assert isinstance(later, runner.D7BitcoinBlobSemantics)
    assert later.grammar_anchor == d7_mechanism.LATER_PRE_ANCHOR
    assert later.normalized_lines[0] == "Retained preface"
    assert later.grammar_receipt_hash == later.classification_detail_hash

    prefixed_raw = d7_mechanism._synthetic_bip(
        proposal,
        dependency="BIP-899",
    )
    prefixed = runner.parse_blob_features(
        "bitcoin",
        proposal,
        runner.core.git_object_sha1("blob", prefixed_raw),
        prefixed_raw,
    )
    assert isinstance(prefixed, runner.D7BitcoinBlobSemantics)
    assert prefixed.dependency_token_mode == "PREFIXED_PRESENT"
    assert dict(prefixed.dependency_edges)["requires"] == (899,)
    assert any("BIP-899" in line for line in prefixed.normalized_lines)

    with pytest.raises(ValueError, match="object SHA-1 mismatch"):
        runner.parse_blob_features(
            "bitcoin",
            proposal,
            "0" * 40,
            prefixed_raw,
        )


def test_bitcoin_overlay_collects_complete_roster_before_typed_reject(
) -> None:
    fixtures = (
        d7_mechanism._synthetic_bip(900, dependency="899"),
        d7_mechanism._synthetic_bip(
            901,
            prefix="Retained preface\n\n",
            dependency="899",
        ),
        d7_mechanism._synthetic_bip(902, dependency="BIP-899"),
        d7_mechanism._synthetic_bip(
            903,
            prefix="Retained preface\n\n",
            second_header_proposal=903,
        ),
    )
    groups: list[runner.ProposalGroup] = []
    raw: dict[str, bytes] = {}
    for index, fixture in enumerate(fixtures):
        group, blobs = _group(
            old_raw=None,
            new_raw=fixture,
            proposal_number=900 + index,
            protocol="bitcoin",
            first_parent_index=index,
            event_type="CREATE",
        )
        groups.append(group)
        raw.update(blobs)

    sink: list[dict[str, Any]] = []
    events = runner._materialize_events_from_raw(
        groups,
        raw,
        runner.AccessLedger.zero(),
        sink,
    )
    assert len(sink) == 1
    receipt = sink[0]
    assert len(events) == 3
    assert receipt["complete_event_outcome_roster"] is True
    assert receipt["event_count_expected"] == 4
    assert receipt["event_count_materialized"] == 3
    assert receipt["semantic_error_count"] == 1
    assert receipt["class_counts"] == {
        "D4_VALID": 1,
        "D7_BITCOIN_GRAMMAR_ERROR": 1,
        "D7_BIP_LATER_HEADER": 1,
        "D7_BIP_PREFIXED_DEPENDENCY": 1,
    }
    assert receipt["event_outcome_counts"] == {
        "ERROR_UNKNOWN_GRAMMAR": 1,
        "PASS_MODEL_VISIBLE": 3,
    }
    serialized = json.dumps(receipt, sort_keys=True)
    assert "Retained preface" not in serialized
    assert "BIP-899" not in serialized


def test_public_migration_authorizer_is_exact_frozen_and_noninjectable() -> None:
    assert tuple(inspect.signature(runner.authorize_migration_restoration).parameters) == (
        "episode",
    )
    authority, manifest_hash = runner._frozen_migration_authority()
    assert len(authority) == 365
    assert manifest_hash == prereg.d6.D5_EPISODE_RECEIPT_MANIFEST_HASH
    with pytest.raises(TypeError):
        authority[next(iter(authority))] = "0" * 64  # type: ignore[index]

    episode = _representative_migration_episode()
    decision = runner.authorize_migration_restoration(episode)
    assert decision["model"] == {
        "administrative_quarantined": True,
        "model_visibility": "ADMINISTRATIVE_QUARANTINE",
        "normalized_text_delta_chunks": [],
    }
    assert (
        decision["audit"]["authority_receipt_hash"]
        == authority[episode["proposal"]]
    )
    for mutation in ("commit", "path", "proposal", "receipt"):
        candidate = copy.deepcopy(episode)
        if mutation == "commit":
            candidate["steps"][2]["commit_oid"] = "0" * 40
        elif mutation == "path":
            candidate["path"] = "EIPS/eip-999999.md"
        elif mutation == "proposal":
            candidate["proposal"] = 999_999
        else:
            candidate["steps"][2]["event_id"] = "0" * 64
        with pytest.raises(ValueError):
            runner.authorize_migration_restoration(candidate)

    proposal = int(episode["proposal"])
    redirect = (
        "This file was moved to "
        f"https://github.com/ethereum/ercs/blob/master/ERCS/erc-{proposal}.md\n"
    ).encode()
    restored = (
        f"---\neip: {proposal}\ntitle: Restored\n---\n"
        "# Abstract\nrestored\n"
    ).encode()
    old_oid = runner.core.git_object_sha1("blob", redirect)
    new_oid = runner.core.git_object_sha1("blob", restored)
    old = runner.parse_blob_features(
        "ethereum",
        proposal,
        old_oid,
        redirect,
    )
    new = runner.parse_blob_features(
        "ethereum",
        proposal,
        new_oid,
        restored,
    )
    with pytest.raises(
        runner.D6SemanticError,
        match="ERROR_UNAUTHORIZED_MIGRATION_RESTORATION",
    ):
        runner.build_event_semantics_d6(
            "ethereum",
            proposal,
            old_path=f"EIPS/eip-{proposal}.md",
            new_path=f"EIPS/eip-{proposal}.md",
            old=old,
            new=new,
            migration_decision={
                "model": decision["model"],
                "audit": {},
            },
        )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a" * 8_192,
        "b" * 8_193,
        "c" * 8_191 + "한" + "d",
        "e" * 8_191 + "\n" + "f",
        "g" * 20_000,
        "h" * 58_416,
        "i" * 65_536,
        "ABSTRACT|ADD|opaque|line|payload",
    ],
    ids=(
        "empty",
        "8192",
        "8193",
        "utf8-boundary",
        "lf-boundary",
        "20000",
        "58416",
        "65536",
        "opaque-pipes",
    ),
)
def test_event_and_model_payload_use_lossless_canonical_chunks(
    text: str,
) -> None:
    event = dataclasses.replace(
        runner.synthetic_events()[0],
        normalized_text_delta_source=text,
    )
    row = runner.event_row(event)
    chunks = row["normalized_text_delta_chunks"]
    assert "".join(
        chunk["normalized_text_delta_chunk"] for chunk in chunks
    ) == text
    assert len(chunks) <= 8
    assert all(
        len(chunk["normalized_text_delta_chunk"].encode("utf-8")) <= 8_192
        for chunk in chunks
    )
    assert "normalized_text_delta" not in row
    schedule = next(iter(event.available_at))
    payload = runner._event_payload(
        event,
        event.available_at[schedule],
        schedule,
    )
    assert payload["normalized_text_delta_chunks"] == chunks
    assert set(payload) == runner.MODEL_EVENT_PAYLOAD_FIELDS
    assert set(payload).isdisjoint(runner.FORBIDDEN_MODEL_PAYLOAD_FIELDS)
    if text == "ABSTRACT|ADD|opaque|line|payload":
        assert runner._swap_add_remove_text(text) == (
            "ABSTRACT|REMOVE|opaque|line|payload"
        )


def test_chunk_tamper_and_ninth_chunk_fail_closed() -> None:
    event = dataclasses.replace(
        runner.synthetic_events()[0],
        normalized_text_delta_source="x" * 20_000,
    )
    row = runner.event_row(event)
    assert runner._event_transport_valid(row)
    for mutation in ("delete", "swap", "text", "receipt"):
        candidate = copy.deepcopy(row)
        if mutation == "delete":
            candidate["normalized_text_delta_chunks"].pop()
        elif mutation == "swap":
            candidate["normalized_text_delta_chunks"][0], candidate[
                "normalized_text_delta_chunks"
            ][1] = (
                candidate["normalized_text_delta_chunks"][1],
                candidate["normalized_text_delta_chunks"][0],
            )
        elif mutation == "text":
            candidate["normalized_text_delta_chunks"][0][
                "normalized_text_delta_chunk"
            ] += "tamper"
        else:
            candidate["model_text_transport_receipt"]["receipt_hash"] = (
                "0" * 64
            )
        assert not runner._event_transport_valid(candidate)
    with pytest.raises(ValueError, match="more than eight chunks"):
        runner.event_row(
            dataclasses.replace(
                event,
                normalized_text_delta_source="x" * 65_537,
            )
        )


def test_materializer_collects_complete_typed_error_roster_without_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = b"---\neip: 20\ntitle: Valid\n---\n# Abstract\nvalid\n"
    unknown = b"not a registered proposal grammar\n"
    chunk_error = (
        b"---\neip: 22\ntitle: Chunk error\n---\n"
        b"# Abstract\nsynthetic\n"
    )
    redirect = (
        b"This file was moved to "
        b"https://github.com/ethereum/ercs/blob/master/ERCS/erc-23.md\n"
    )
    restored = (
        b"---\neip: 23\ntitle: Restored\n---\n# Abstract\nrestored\n"
    )
    specifications = (
        (None, valid, 20, "CREATE"),
        (None, unknown, 21, "CREATE"),
        (None, chunk_error, 22, "CREATE"),
        (redirect, restored, 23, "UPDATE"),
        (None, b"\xff", 24, "CREATE"),
    )
    groups: list[runner.ProposalGroup] = []
    raw: dict[str, bytes] = {}
    for index, (old, new, proposal, event_type) in enumerate(specifications):
        group, blobs = _group(
            old_raw=old,
            new_raw=new,
            proposal_number=proposal,
            first_parent_index=index,
            event_type=event_type,
        )
        groups.append(group)
        raw.update(blobs)
    original_builder = runner.build_event_semantics_d6

    def build_with_one_typed_chunk_error(
        protocol: str,
        proposal_number: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if proposal_number == 22:
            raise runner.D6SemanticError(
                "ERROR_MODEL_TEXT_CHUNK_TRANSPORT"
            )
        return original_builder(protocol, proposal_number, **kwargs)

    monkeypatch.setattr(
        runner,
        "build_event_semantics_d6",
        build_with_one_typed_chunk_error,
    )
    sink: list[dict[str, Any]] = []
    events = runner._materialize_events_from_raw(
        groups,
        raw,
        runner.AccessLedger.zero(),
        sink,
    )
    assert len(events) == 1
    receipt = sink[0]
    assert receipt["event_count_expected"] == 5
    assert receipt["event_count_materialized"] == 1
    assert receipt["complete_event_outcome_roster"] is True
    assert receipt["semantic_error_count"] == 4
    assert receipt["event_outcome_counts"] == {
        "ERROR_MODEL_TEXT_CHUNK_TRANSPORT": 1,
        "ERROR_STRICT_UTF8": 1,
        "ERROR_UNAUTHORIZED_MIGRATION_RESTORATION": 1,
        "ERROR_UNKNOWN_GRAMMAR": 1,
        "PASS_MODEL_VISIBLE": 1,
    }
    assert receipt["raw_or_normalized_text_published"] is False
    assert _nested_keys(receipt).isdisjoint(
        {
            "exception",
            "failure",
            "normalized_text_delta",
            "normalized_text_delta_chunk",
            "raw_bytes",
            "raw_text",
        }
    )
    serialized = json.dumps(receipt, sort_keys=True)
    assert "registered proposal grammar" not in serialized
    assert "restored" not in serialized


def test_gate_four_accepts_exact_mechanisms_and_binds_every_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, semantics, hydration, ledger = _gate_fixture(monkeypatch)
    gate = runner.gate_event_parser_replay(
        events,
        semantics,
        hydration,
        ledger,
    )
    assert gate.passed
    assert gate.metrics["full_roster_collected_before_decision"] is True
    assert gate.metrics["ethereum"]["migration_and_chunk_mechanisms_exact"]
    assert gate.metrics["ethereum"]["model_text_transport_valid"]
    assert gate.metrics["ethereum"]["complete_typed_outcome_roster"]

    wrong_migration = events[("ethereum", "b")][0]
    wrong_commit = "9" * 40
    wrong_event_id = runner._event_id(
        wrong_migration.protocol,
        wrong_commit,
        wrong_migration.proposal_number,
        wrong_migration.old_blob_oid,
        wrong_migration.new_blob_oid,
    )
    wrong_events = dict(events)
    wrong_events[("ethereum", "b")] = [
        dataclasses.replace(
            wrong_migration,
            commit_oid=wrong_commit,
            event_id=wrong_event_id,
        ),
        *events[("ethereum", "b")][1:],
    ]
    wrong_gate = runner.gate_event_parser_replay(
        wrong_events,
        semantics,
        hydration,
        ledger,
    )
    assert not wrong_gate.passed
    assert not wrong_gate.metrics["ethereum"][
        "migration_and_chunk_mechanisms_exact"
    ]

    tampered_events = dict(events)
    first = tampered_events[("ethereum", "b")][365]
    tampered_events[("ethereum", "b")] = [
        *tampered_events[("ethereum", "b")][:365],
        dataclasses.replace(
            first,
            normalized_text_delta_source=(
                first.normalized_text_delta_source + "tamper"
            ),
        ),
        *tampered_events[("ethereum", "b")][366:],
    ]
    gate = runner.gate_event_parser_replay(
        tampered_events,
        semantics,
        hydration,
        ledger,
    )
    assert not gate.passed
    assert not gate.metrics["ethereum"]["model_text_transport_valid"]
    assert not gate.metrics["ethereum"]["replica_event_and_outcome_replay"]


def test_gate_four_collects_all_replicas_before_typed_error_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = runner._typed_outcome(
        "f" * 64,
        "ERROR_UNKNOWN_GRAMMAR",
        passed=False,
    )
    events, semantics, hydration, ledger = _gate_fixture(
        monkeypatch,
        error_outcomes=[error]
    )
    gate = runner.gate_event_parser_replay(
        events,
        semantics,
        hydration,
        ledger,
    )
    assert not gate.passed
    assert gate.metrics["receipt_roster_exact"] is True
    assert gate.metrics["full_roster_collected_before_decision"] is True
    assert gate.metrics["ethereum"]["semantic_error_count"] == 1
    assert not gate.metrics["ethereum"]["complete_typed_outcome_roster"]

    incomplete = dict(semantics)
    incomplete.pop(("bitcoin", "b"))
    gate = runner.gate_event_parser_replay(
        events,
        incomplete,
        hydration,
        ledger,
    )
    assert not gate.passed
    assert gate.failure == "INCOMPLETE_FOUR_REPLICA_ROSTER"
    assert gate.metrics["full_roster_collected_before_decision"] is False


def test_real_local_clone_uses_d6_hydration_with_fresh_d7_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    origin, sealed_tip, branch_tip = _synthetic_origin(tmp_path)
    spec = _local_spec(origin, sealed_tip)
    monkeypatch.setattr(runner, "_repository_spec", lambda protocol: spec)
    config = runner.Config(source_root=tmp_path / "source")
    ledger = runner.AccessLedger.zero()
    clone_receipt = runner.prepare_source_repository(
        config,
        "ethereum",
        "a",
        ledger,
    )
    assert clone_receipt["sealed_tip"] == sealed_tip
    assert branch_tip != sealed_tip
    repo = runner.clone_path(config, "ethereum", "a")
    records = runner.collect_commit_chain(repo, "ethereum", ledger)
    groups, issues = runner.collect_proposal_groups(repo, records, ledger)
    assert issues == []
    hydration: list[dict[str, Any]] = []
    semantics: list[dict[str, Any]] = []
    events = runner.materialize_events(
        repo,
        groups,
        ledger,
        hydration,
        semantics,
    )
    assert len(hydration) == 1
    assert hydration[0]["fetch_invocations"] == 1
    assert hydration[0]["post_read_fetch_child_processes"] == 0
    assert hydration[0]["post_read_object_store_unchanged"] is True
    assert len(semantics) == 1
    assert semantics[0]["class_counts"] == {"D4_VALID": 1}
    assert semantics[0]["semantic_error_count"] == 0
    assert len(events) == 1
    row = runner.event_row(events[0])
    assert "ABSTRACT|ADD|one" in "".join(
        chunk["normalized_text_delta_chunk"]
        for chunk in row["normalized_text_delta_chunks"]
    ).splitlines()
    assert (tmp_path / "source" / ".psim-d7-traces").is_dir()
    assert not (tmp_path / "source" / ".psim-d6-traces").exists()


def test_hydration_failure_receipt_hashes_error_without_message(
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
    secret = "SENSITIVE_RAW_PROPOSAL_TEXT"

    def fail(repo, object_ids, ledger, progress):
        progress.update(
            {
                "repository_root_name": "ethereum-a.git",
                "fetch_invocations": 1,
                "stage": "post_hydration_inventory",
            }
        )
        raise RuntimeError(secret)

    monkeypatch.setattr(runner, "_hydrate_blob_batch", fail)
    receipts: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match=secret):
        runner.materialize_events(
            tmp_path,
            [group],
            runner.AccessLedger.zero(),
            receipts,
        )
    receipt = receipts[0]
    assert "failure" not in receipt
    assert "error_profile_hash" in receipt
    assert secret not in json.dumps(receipt, sort_keys=True)
    assert receipt["receipt_hash"] == runner.canonical_hash(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_hash"
        }
    )


def test_artifacts_and_controls_preserve_chunk_only_model_boundary() -> None:
    events = runner.synthetic_events()
    cards = runner.build_daily_cards(events)
    controls = runner.build_control_metrics(events, cards)
    assert all(row["passed"] for row in controls.values())
    for control in runner.core.prereg.RELATION_CONTROLS:
        transformed = runner.transform_events(events, control)
        assert len(transformed) == len(events)
        assert all(
            "normalized_text_delta" not in runner.event_row(row)
            for row in transformed
        )
    control_gate = runner.gate_control_sensitivity(controls)
    config = runner.Config()
    raw, manifest = runner.build_pass_artifacts(
        config,
        events,
        cards,
        controls,
        control_gate,
    )
    assert manifest["events"]["rows"] == len(events)
    event_bytes = gzip.decompress(raw[config.events_path])
    card_bytes = gzip.decompress(raw[config.cards_path])
    assert b'"intent_text"' not in event_bytes
    assert b'"intent_text"' not in card_bytes
    assert b'"normalized_text_delta":' not in event_bytes
    assert b'"normalized_text_delta":' not in card_bytes
    assert b'"normalized_text_delta_chunks"' in event_bytes
    assert b'"normalized_text_delta_chunks"' in card_bytes
    assert b'"model_text_transport_receipt"' not in card_bytes


def test_rejection_report_never_serializes_exception_message() -> None:
    secret = "SENSITIVE_RAW_OR_NORMALIZED_TEXT"
    unsafe_receipt = {
        "protocol": "ethereum",
        "receipt_hash": "b" * 64,
        "event_outcomes": [
            runner._typed_outcome(
                "c" * 64,
                "ERROR_UNKNOWN_GRAMMAR",
                passed=False,
            )
        ],
        "migration_restoration_receipts": [
            {
                "causal_episode": {
                    "raw_text": secret,
                    "old_path": "EIPS/eip-1.md",
                }
            }
        ],
    }
    summary = runner._rejection_semantics_summary(unsafe_receipt)
    source_audit = {
        "proposal_path_incidence_opened": True,
        "blob_semantics_receipts": {"ethereum:a": summary},
    }
    assert runner._rejection_semantics_publication_safe(source_audit)
    assert "causal_episode" not in _nested_keys(summary)
    assert secret not in json.dumps(summary, sort_keys=True)
    gate = runner.GateResult(
        name=runner.GATE_NAMES[0],
        passed=False,
        metrics={"gate_evaluation_completed": False},
        failure="typed failure",
    )
    report = runner.build_result_report(
        decision="reject",
        authority={"source_authority_hash": "a" * 64},
        gates=[gate],
        source_audit=source_audit,
        event_count=0,
        card_count=0,
        artifacts=None,
        ledger=runner.AccessLedger.zero(),
        error=RuntimeError(secret),
    )
    serialized = json.dumps(report, sort_keys=True)
    assert secret not in serialized
    assert report["error"] == {"type": "RuntimeError"}
    assert report["outcomes_opened"] is False
    assert report["profitability_result"] is False
    with pytest.raises(RuntimeError, match="rejection report is incomplete"):
        runner.build_result_report(
            decision="reject",
            authority={"source_authority_hash": "a" * 64},
            gates=[gate],
            source_audit={
                "blob_semantics_receipts": {
                    "ethereum:a": unsafe_receipt
                }
            },
            event_count=0,
            card_count=0,
            artifacts=None,
            ledger=runner.AccessLedger.zero(),
        )


def test_source_configuration_is_frozen_and_symlink_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="official source root is frozen"):
        runner._validate_source_configuration(
            runner.Config(source_root=tmp_path / "other")
        )
    stale_root = tmp_path / "stale-source"
    stale_root.mkdir()
    sentinel = stale_root / "preexisting-object"
    sentinel.write_text("must-not-be-reused", encoding="utf-8")
    monkeypatch.setattr(runner, "DEFAULT_SOURCE_ROOT", stale_root)
    stale_config = runner.Config(source_root=stale_root)
    runner._validate_source_configuration(stale_config)
    with pytest.raises(RuntimeError, match="fresh source root already exists"):
        runner._create_fresh_source_root(stale_config)
    assert sentinel.read_text(encoding="utf-8") == "must-not-be-reused"

    fresh_root = tmp_path / "fresh-source"
    monkeypatch.setattr(runner, "DEFAULT_SOURCE_ROOT", fresh_root)
    fresh_config = runner.Config(source_root=fresh_root)
    runner._validate_source_configuration(fresh_config)
    runner._create_fresh_source_root(fresh_config)
    assert fresh_root.is_dir()
    assert list(fresh_root.iterdir()) == []

    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "source-link"
    link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(runner, "DEFAULT_SOURCE_ROOT", link / "nested")
    with pytest.raises(ValueError, match="symlink ancestor"):
        runner._validate_source_configuration(
            runner.Config(source_root=link / "nested")
        )


def test_execution_seal_creation_fails_closed_while_worktree_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_worktree_clean", lambda: False)
    with pytest.raises(RuntimeError, match="clean worktree"):
        runner.build_execution_seal()


def test_official_run_refuses_without_seal_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("official source opened before execution-seal validation")

    monkeypatch.setattr(runner, "terminal_state", lambda config: None)
    monkeypatch.setattr(runner, "_create_fresh_source_root", forbidden)
    monkeypatch.setattr(
        runner,
        "validate_execution_seal",
        lambda: (_ for _ in ()).throw(
            RuntimeError("synthetic missing D7 seal")
        ),
    )
    with pytest.raises(RuntimeError, match="missing D7 seal"):
        runner.run_official(runner.Config())


def test_dual_epoch_verification_combines_archive_and_current_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited = {
        "passed": 602,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    current = {
        "passed": 70,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    monkeypatch.setattr(
        runner,
        "_run_inherited_archive_verification",
        lambda: inherited,
    )
    monkeypatch.setattr(
        runner,
        "_run_pytest_paths",
        lambda paths, *, cwd, epoch, expected_passed: current,
    )
    receipt = runner._run_pytest_verification()
    assert receipt["protocol_version"] == (
        "psim_d7_dual_epoch_pytest_verification_v1"
    )
    assert receipt["inherited_pre_rebase"] == inherited
    assert receipt["current_d7"] == current
    assert receipt["totals"] == {
        "passed": 672,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }


def test_pytest_verification_sanitizes_ambient_selection_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, Any] = {}
    simulated = {"stdout": "1 passed in 0.01s\n"}
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k never_collect_anything")
    monkeypatch.setenv("PYTEST_PLUGINS", "untrusted_plugin")
    monkeypatch.setenv("COVERAGE_PROCESS_START", "/tmp/untrusted")

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout=simulated["stdout"],
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    receipt = runner._run_pytest_paths(
        (Path("tests/test_synthetic.py"),),
        cwd=tmp_path,
        epoch="SYNTHETIC",
        expected_passed=1,
    )
    environment = observed["env"]
    assert environment["PYTEST_ADDOPTS"] == ""
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "PYTEST_PLUGINS" not in environment
    assert "COVERAGE_PROCESS_START" not in environment
    assert receipt["expected_passed"] == 1
    simulated["stdout"] = "2 passed in 0.01s\n"
    with pytest.raises(RuntimeError, match="pytest verification failed"):
        runner._run_pytest_paths(
            (Path("tests/test_synthetic.py"),),
            cwd=tmp_path,
            epoch="SYNTHETIC",
            expected_passed=1,
        )


def test_inherited_archive_verification_rejects_stale_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stale = tmp_path / "stale-archive-worktree"
    stale.mkdir()
    monkeypatch.setattr(
        runner,
        "INHERITED_VERIFICATION_ROOT",
        stale,
    )
    with pytest.raises(RuntimeError, match="unsafe or stale"):
        runner._run_inherited_archive_verification()


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
    assert observed["env"]["PYTEST_ADDOPTS"] == ""
    assert observed["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert payload["expected_passed"] == runner.EXPECTED_SEAL_TEST_PASSED
