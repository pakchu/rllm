from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from training import (
    probe_protocol_specification_intent_maturity_d5_event_semantics as probe,
)

PROBE_SHA256 = (
    "f4496846b979ba1e832b4a7108ae9575f0c0e44101f006062efc4453aa6f8799"
)
RESULT_HASH = (
    "b94321b815f4f32cc8c8b6d9b323b88d3b8f29ab1e7e410f4b8266b92e4c186b"
)


def blob(
    raw: bytes,
    *,
    protocol: str = "ethereum",
    proposal_number: int = 20,
) -> probe.BlobSemantics:
    oid = probe.d4.core.git_object_sha1("blob", raw)
    return probe.decode_blob_d5(
        protocol,
        proposal_number,
        oid,
        raw,
    )


def event(
    old: probe.BlobSemantics | None,
    new: probe.BlobSemantics | None,
    *,
    protocol: str = "ethereum",
    proposal_number: int = 20,
    old_path: str | None = None,
    new_path: str | None = None,
) -> dict:
    prefix = "EIPS/eip" if protocol == "ethereum" else "bip"
    extension = "md" if protocol == "ethereum" else "mediawiki"
    return probe.build_event_semantics_d5(
        protocol,
        proposal_number,
        old_path=(
            old_path
            if old_path is not None
            else (
                None
                if old is None
                else f"{prefix}-{proposal_number}.{extension}"
            )
        ),
        new_path=(
            new_path
            if new_path is not None
            else (
                None
                if new is None
                else f"{prefix}-{proposal_number}.{extension}"
            )
        ),
        old=old,
        new=new,
    )


@pytest.fixture(scope="module")
def payload() -> dict[str, object]:
    return probe.build_probe()


def test_written_probe_is_frozen_canonical_and_replay_equal(
    payload: dict[str, object],
) -> None:
    target = probe.repository_path(probe.DEFAULT_OUTPUT)
    raw = target.read_bytes()
    written = json.loads(raw)
    core = {
        key: value
        for key, value in written.items()
        if key != "result_hash"
    }

    assert probe.DEFAULT_OUTPUT.as_posix() == (
        "results/protocol_specification_intent_maturity_d5_event_semantics_"
        "probe_2026-07-26.json"
    )
    assert hashlib.sha256(raw).hexdigest() == PROBE_SHA256
    assert raw == probe.canonical_json_bytes(written)
    assert written == payload
    assert written["result_hash"] == RESULT_HASH
    assert written["result_hash"] == probe.canonical_hash(core)


def test_probe_binds_exact_d4_terminal_and_complete_census(
    payload: dict[str, object],
) -> None:
    assert payload["d4_terminal_binding"] == {
        "commit": probe.D4_TERMINAL_COMMIT,
        "decision": "reject",
        "first_failure_gate_id": 4,
        "path": probe.D4_TERMINAL_PATH.as_posix(),
        "result_hash": probe.D4_TERMINAL_RESULT_HASH,
        "sha256": probe.D4_TERMINAL_SHA256,
    }
    assert payload["d4_census_binding"] == {
        "commit": probe.CENSUS_COMMIT,
        "decision_document": {
            "path": probe.DECISION_PATH.as_posix(),
            "sha256": probe.DECISION_SHA256,
        },
        "path": probe.CENSUS_PATH.as_posix(),
        "result_hash": probe.CENSUS_RESULT_HASH,
        "script": {
            "path": probe.CENSUS_SCRIPT_PATH.as_posix(),
            "sha256": probe.CENSUS_SCRIPT_SHA256,
        },
        "sha256": probe.CENSUS_SHA256,
        "test": {
            "path": probe.CENSUS_TEST_PATH.as_posix(),
            "sha256": probe.CENSUS_TEST_SHA256,
        },
    }


def test_probe_is_synthetic_only_and_outcome_blind(
    payload: dict[str, object],
) -> None:
    assert payload["synthetic_only"] is True
    assert payload["selection_scope"] == (
        "AUTHORIZE_D5_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
    )
    assert payload["access_boundary"] == {
        "d4_census_artifact_read": True,
        "d4_forensic_root_accessed": False,
        "d4_terminal_artifact_read": True,
        "external_network_accessed_by_probe": False,
        "market_data_accessed": False,
        "model_accessed": False,
        "official_reference_research_preexisted_probe": True,
        "official_historical_proposal_source_accessed": False,
        "outcomes_accessed": False,
    }
    assert payload["candidate"] == {
        "id": "PSIM-D5",
        "name": (
            "Protocol Specification Intent-Maturity relation RLLM, "
            "exact path identity plus normalized text-delta semantics"
        ),
        "source_representation_successor": True,
    }
    assert payload["official_reference_provenance"] == {
        "accessed_by_probe": False,
        "model_visible": False,
        "origin": (
            "PRIOR_OFFICIAL_SOURCE_RESEARCH_RECORDED_IN_D4_CENSUS_DECISION"
        ),
        "selection_evidence_only": True,
    }


def test_valid_event_uses_exact_path_and_normalized_algorithmic_delta() -> (
    None
):
    old = blob(
        b"---\n"
        b"eip: 20\n"
        b"status: Draft\n"
        b"requires: 1\n"
        b"---\n"
        b"# Abstract\n"
        b"Old text.\n"
    )
    new = blob(
        b"---\n"
        b"eip: 20\n"
        b"status: Review\n"
        b"requires: 1, 2\n"
        b"---\n"
        b"# Abstract\n"
        b"New text.\n"
    )

    observed = event(old, new)

    identity = observed["path_identity"]
    assert identity["protocol"] == "ethereum"
    assert identity["proposal_number"] == 20
    assert identity["old_path"] == "EIPS/eip-20.md"
    assert identity["new_path"] == "EIPS/eip-20.md"
    assert len(identity["identity_hash"]) == 64
    assert observed["path_identity_source"] == (
        "PROPOSAL_GROUP_EXACT_SIDE_PATHS"
    )
    assert observed["model_visibility"] == "MODEL_VISIBLE"
    assert observed["dependency_delta_state"] == "ADDED"
    assert observed["dependency_edge_delta_count"] == 1
    assert observed["invalid_metadata_present"] is False
    assert observed["invalid_metadata_states"] == []
    assert observed["audit_line_change_count"] == (
        observed["model_line_change_count"]
    )
    assert observed["normalized_text_delta"].splitlines() == [
        "OTHER|REMOVE|status: Draft",
        "OTHER|REMOVE|requires: 1",
        "OTHER|ADD|status: Review",
        "OTHER|ADD|requires: 1, 2",
        "ABSTRACT|REMOVE|Old text.",
        "ABSTRACT|ADD|New text.",
    ]


@pytest.mark.parametrize(
    ("raw", "expected_state"),
    [
        (
            b"---\neip: 20\nstatus: Draft\nstatus: Draft\n---\n"
            b"# Abstract\nDuplicate.\n",
            "INVALID_DUPLICATE_IDENTICAL",
        ),
        (
            b"---\neip: 20\nstatus: Draft\nstatus: Final\n---\n"
            b"# Abstract\nConflict.\n",
            "INVALID_DUPLICATE_CONFLICTING",
        ),
        (
            b"---\neip: 20\nrequires (*optional): 1\n---\n"
            b"# Abstract\nMalformed.\n",
            "INVALID_MALFORMED_HEADER",
        ),
        (
            b"---\neip: 20\nrequires: 1, 20\n---\n"
            b"# Abstract\nSelf dependency.\n",
            "INVALID_SELF_DEPENDENCY",
        ),
    ],
)
def test_known_invalid_metadata_is_explicit_without_repair_and_text_remains(
    raw: bytes,
    expected_state: str,
) -> None:
    old = blob(
        b"---\neip: 20\ntitle: Synthetic control\n---\n"
        b"# Abstract\nControl.\n"
    )
    invalid = blob(raw)
    observed = event(old, invalid)

    assert invalid.metadata_state == expected_state
    assert invalid.metadata_header == ()
    assert invalid.dependency_edges == ()
    assert invalid.dependency_availability == "UNKNOWN_INVALID_METADATA"
    assert invalid.model_visible is True
    assert observed["model_visibility"] == "MODEL_VISIBLE"
    assert observed["new_metadata_state"] == expected_state
    assert observed["invalid_metadata_present"] is True
    assert observed["invalid_metadata_states"] == [expected_state]
    assert observed["dependency_delta_state"] == "UNKNOWN_INVALID_METADATA"
    assert observed["dependency_edge_delta_count"] is None
    assert observed["normalized_text_delta"]
    assert observed["model_line_change_count"] == (
        observed["audit_line_change_count"]
    )


def test_exact_migration_redirect_is_audited_but_not_model_visible() -> None:
    old = blob(
        b"---\neip: 20\ntitle: Synthetic prior document\n---\n"
        b"# Specification\nSynthetic specification.\n"
    )
    lower = blob(
        b"This file was moved to "
        b"https://github.com/ethereum/ercs/blob/master/ercs/erc-20.md\n"
    )
    upper = blob(
        b"This file was moved to "
        b"https://github.com/ethereum/ercs/blob/master/ERCS/erc-20.md\n"
    )

    migration = event(old, lower)
    case_fix = event(lower, upper)

    assert lower.administrative_class == (
        "ERC_MIGRATION_REDIRECT_LOWER_PATH"
    )
    assert upper.administrative_class == (
        "ERC_MIGRATION_REDIRECT_UPPER_PATH"
    )
    for observed in (migration, case_fix):
        assert observed["model_visibility"] == (
            "ADMINISTRATIVE_QUARANTINE"
        )
        assert observed["normalized_text_delta"] == ""
        assert observed["model_line_change_count"] == 0
        assert observed["audit_line_change_count"] > 0
        assert len(observed["audit_diff_hash"]) == 64
        assert observed["dependency_delta_state"] == (
            "ADMINISTRATIVE_QUARANTINE"
        )
        assert observed["dependency_edge_delta_count"] is None
        assert observed["quarantine_reason"] == (
            "EXACT_2023_ETHEREUM_ERC_REPOSITORY_MIGRATION_STUB"
        )


def test_invalid_to_redirect_keeps_invalid_state_in_quarantine_audit() -> None:
    invalid = blob(
        b"---\neip: 20\nstatus: Draft\nstatus: Final\n---\n"
        b"# Abstract\nSynthetic conflict.\n"
    )
    redirect = blob(
        b"This file was moved to "
        b"https://github.com/ethereum/ercs/blob/master/ercs/erc-20.md\n"
    )

    observed = event(invalid, redirect)

    assert observed["model_visibility"] == "ADMINISTRATIVE_QUARANTINE"
    assert observed["normalized_text_delta"] == ""
    assert observed["invalid_metadata_present"] is True
    assert observed["invalid_metadata_states"] == [
        "INVALID_DUPLICATE_CONFLICTING",
    ]
    assert observed["old_metadata_state"] == (
        "INVALID_DUPLICATE_CONFLICTING"
    )


@pytest.mark.parametrize(
    "raw",
    [
        (
            b"This file was moved to "
            b"https://github.com/ethereum/ercs/blob/master/"
            b"ercs/erc-21.md\n"
        ),
        (
            b"This file was moved to "
            b"https://github.com/ethereum/ercs/blob/master/"
            b"ercs/erc-20.md\nextra\n"
        ),
        b"---\neip: 21\ntitle: Path mismatch\n---\n",
        b"not an EIP preamble\n",
    ],
)
def test_unclassified_grammar_is_preserved_as_state_then_fails_closed(
    raw: bytes,
) -> None:
    control = blob(b"---\neip: 20\ntitle: Control\n---\n")
    unknown = blob(raw)

    assert unknown.metadata_state == probe.UNKNOWN_STATE
    assert unknown.dependency_availability == "UNKNOWN_UNCLASSIFIED"
    assert unknown.model_visible is False
    assert unknown.normalized_lines
    assert len(unknown.classification_detail_hash) == 64
    with pytest.raises(
        ValueError,
        match="unclassified metadata fails closed",
    ):
        event(control, unknown)


def test_reverse_migration_and_cross_path_sides_fail_closed() -> None:
    redirect = blob(
        b"This file was moved to "
        b"https://github.com/ethereum/ercs/blob/master/ercs/erc-20.md\n"
    )
    document = blob(b"---\neip: 20\ntitle: Synthetic return\n---\n")
    other_path = blob(
        b"---\neip: 21\ntitle: Other path\n---\n",
        proposal_number=21,
    )

    with pytest.raises(
        ValueError,
        match="reverse administrative migration is ambiguous",
    ):
        event(redirect, document)
    with pytest.raises(ValueError, match="event path identity changed"):
        probe.build_event_semantics_d5(
            "ethereum",
            20,
            old_path="EIPS/eip-20.md",
            new_path="EIPS/eip-21.md",
            old=document,
            new=other_path,
        )


def test_path_identity_requires_protocol_and_exact_side_path_shape() -> None:
    document = blob(b"---\neip: 20\ntitle: Synthetic\n---\n")

    with pytest.raises(
        ValueError,
        match="event path/blob side shape differs",
    ):
        probe.build_event_semantics_d5(
            "ethereum",
            20,
            old_path=None,
            new_path="EIPS/eip-20.md",
            old=document,
            new=document,
        )
    with pytest.raises(ValueError, match="event path identity changed"):
        probe.build_event_semantics_d5(
            "bitcoin",
            20,
            old_path="bip-20.mediawiki",
            new_path="bip-20.mediawiki",
            old=document,
            new=document,
        )


def test_repeated_and_moved_lines_are_a_deterministic_algorithmic_proxy() -> (
    None
):
    pairs = (
        (
            blob(
                b"---\neip: 20\n---\n# Abstract\nrepeat\nanchor\nrepeat\n"
            ),
            blob(
                b"---\neip: 20\n---\n# Abstract\nrepeat\nrepeat\nanchor\n"
            ),
        ),
        (
            blob(
                b"---\neip: 20\n---\n# Abstract\nfirst\nsecond\nthird\n"
            ),
            blob(
                b"---\neip: 20\n---\n# Abstract\nthird\nfirst\nsecond\n"
            ),
        ),
    )

    for old, new in pairs:
        first = probe._normalized_changed_rows(old, new)
        second = probe._normalized_changed_rows(old, new)
        assert first == second
        assert first[0]


def test_bitcoin_parser_output_remains_exactly_d4() -> None:
    raw = (
        b"<pre>\n"
        b"  BIP: 0020\n"
        b"  Title: Synthetic BIP\n"
        b"  Requires: 1, 2\n"
        b"</pre>\n"
    )
    oid = probe.d4.core.git_object_sha1("blob", raw)

    candidate = probe.decode_blob_d5("bitcoin", 20, oid, raw)
    control = probe.d4.parse_blob_features("bitcoin", 20, oid, raw)

    assert candidate.metadata_state == "VALID"
    assert candidate.metadata_header == tuple(
        sorted(control.header.items())
    )
    assert candidate.dependency_edges == tuple(
        sorted(control.dependency_edges.items())
    )
    with pytest.raises((UnicodeDecodeError, ValueError)):
        blob(
            b"<pre>\n  BIP: zero\n</pre>\n",
            protocol="bitcoin",
        )


def test_blob_oid_protocol_and_empty_event_boundaries_fail_closed() -> None:
    raw = b"---\neip: 20\ntitle: Synthetic\n---\n"
    with pytest.raises(ValueError, match="blob object SHA-1 mismatch"):
        probe.decode_blob_d5("ethereum", 20, "0" * 40, raw)
    with pytest.raises(ValueError, match="protocol must be"):
        probe.decode_blob_d5(
            "other",
            20,
            probe.d4.core.git_object_sha1("blob", raw),
            raw,
        )
    with pytest.raises(ValueError, match="event has no blob sides"):
        probe.build_event_semantics_d5(
            "ethereum",
            20,
            old_path=None,
            new_path=None,
            old=None,
            new=None,
        )


def test_semantics_contract_explicitly_forbids_metadata_repairs(
    payload: dict[str, object],
) -> None:
    contract = payload["semantics_contract"]
    assert isinstance(contract, dict)
    assert contract["path_identity"] == (
        "PROTOCOL_PLUS_EXACT_OLD_NEW_GROUP_PATHS_PLUS_NUMBER_"
        "CANONICAL_HASH_BOUND"
    )
    assert contract["metadata_resolution"] == (
        "NONE_NO_FIRST_LAST_MERGE_DEDUP_RENAME_OR_SELF_EDGE_DROP"
    )
    assert contract["dependency_when_metadata_invalid"] == (
        "UNKNOWN_WITH_NULL_COUNT_NO_REPAIR"
    )
    assert contract["unknown_grammar"] == (
        "FAIL_CLOSED_BEFORE_MODEL_OR_OUTCOMES"
    )
    assert contract["known_invalid_metadata_text_model_visible"] == (
        "TRUE_FOR_NONADMINISTRATIVE_EVENTS"
    )
    assert contract["administrative_text_model_visible"] is False
    assert contract["model_text_field"] == "normalized_text_delta"
    assert contract["normalized_text_delta_is_causal_semantics_claim"] is (
        False
    )


def test_synthetic_battery_is_complete(
    payload: dict[str, object],
) -> None:
    assert payload["synthetic_battery"] == {
        "administrative_quarantine_preserves_invalid_metadata_audit": True,
        "administrative_quarantine_transitions": 4,
        "algorithmic_delta_repeated_or_moved_line_cases": 2,
        "bip_d4_parse_outputs_unchanged": 1,
        "dependency_unknown_without_repair_for_known_invalid": 4,
        "exact_redirect_path_case_variants": 2,
        "known_invalid_metadata_states_model_visible": {
            "INVALID_DUPLICATE_CONFLICTING": True,
            "INVALID_DUPLICATE_IDENTICAL": True,
            "INVALID_MALFORMED_HEADER": True,
            "INVALID_SELF_DEPENDENCY": True,
        },
        "normalized_text_delta_includes_metadata_and_body_lines": True,
        "path_identity_binds_protocol_number_and_exact_side_paths": True,
        "raw_audit_diff_preserved_when_quarantined": True,
        "reverse_migration_fails_closed": True,
        "unknown_grammar_cases_fail_closed": 3,
    }


def test_probe_hash_is_complete(payload: dict[str, object]) -> None:
    core = dict(payload)
    result_hash = core.pop("result_hash")
    assert result_hash == probe.canonical_hash(core)


def test_written_probe_is_canonical_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    target = tmp_path / "probe.json"
    monkeypatch.setattr(probe, "build_probe", lambda: payload)
    first = probe.write_probe(target)
    second = probe.write_probe(target)
    assert first == second == payload
    assert target.read_bytes() == probe.canonical_json_bytes(payload)


def test_existing_probe_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    target = tmp_path / "probe.json"
    target.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(probe, "build_probe", lambda: payload)
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D5 event semantics probe differs",
    ):
        probe.write_probe(target)


def test_probe_imports_no_network_market_or_model_clients() -> None:
    source = probe.repository_path(
        "training/"
        "probe_protocol_specification_intent_maturity_d5_event_semantics.py"
    ).read_text(encoding="utf-8")
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
            "subprocess",
            "torch",
            "transformers",
            "urllib",
            "yfinance",
        }
    )
    assert "/tmp/psim-d4-source" not in source
    assert "build_census(" not in source
    assert (
        "build_protocol_specification_intent_maturity_d4_source_support.run"
        not in source
    )
