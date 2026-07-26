from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from training import (
    preregister_protocol_specification_intent_maturity_d6 as d6,
)
from training import (
    preregister_protocol_specification_intent_maturity_d7 as d7,
)
from training import (
    probe_protocol_specification_intent_maturity_d7_bitcoin_grammar_mechanism
    as mechanism,
)


RESULT_PATH = d7.REPO_ROOT / d7.DEFAULT_OUTPUT
PREREGISTRATION_DOCUMENT_PATH = (
    d7.REPO_ROOT
    / "docs/psim-d7-source-support-preregistration-2026-07-26.md"
)
RESULT_SHA256 = (
    "e9402b984232a9c30a5bc427ee8b828b4e61b7f355746e36ee5fe986be3ae79d"
)


def contract_core(payload: dict[str, object]) -> dict[str, object]:
    core = copy.deepcopy(payload)
    core.pop("manifest_hash")
    core.pop("inheritance_proof")
    return core


def test_d6_preregistration_seal_and_terminal_are_exactly_bound() -> None:
    payload = d7.build_preregistration()
    inheritance = payload["inheritance_proof"]

    d6_registration = d7._read_canonical_json(
        d7.D6_PREREGISTRATION_PATH
    )
    assert d6_registration == d6.build_preregistration()
    assert d7.sha256_file(d7.D6_PREREGISTRATION_PATH) == (
        d7.D6_PREREGISTRATION_SHA256
    )
    assert inheritance["d6_preregistration"] == {
        "path": d7.D6_PREREGISTRATION_PATH.as_posix(),
        "commit": d7.D6_PREREGISTRATION_COMMIT,
        "sha256": d7.D6_PREREGISTRATION_SHA256,
        "manifest_hash": d7.D6_PREREGISTRATION_MANIFEST_HASH,
        "contract_core_hash": d7.D6_PREREGISTRATION_CORE_HASH,
        "producer": {
            "path": d7.D6_PREREGISTRATION_SCRIPT_PATH.as_posix(),
            "sha256": d7.D6_PREREGISTRATION_SCRIPT_SHA256,
        },
        "test": {
            "path": d7.D6_PREREGISTRATION_TEST_PATH.as_posix(),
            "sha256": d7.D6_PREREGISTRATION_TEST_SHA256,
        },
        "document": {
            "path": d7.D6_PREREGISTRATION_DOCUMENT_PATH.as_posix(),
            "sha256": d7.D6_PREREGISTRATION_DOCUMENT_SHA256,
        },
    }

    seal = d7._read_canonical_json(d7.D6_SEAL_PATH)
    assert d7.sha256_file(d7.D6_SEAL_PATH) == d7.D6_SEAL_SHA256
    assert seal["seal_hash"] == d7.D6_SEAL_HASH
    assert seal["shared_commit"] == d7.D6_IMPLEMENTATION_COMMIT
    assert inheritance["d6_execution_seal"]["commit"] == (
        d7.D6_SEAL_COMMIT
    )

    terminal = d7._read_canonical_json(d7.D6_TERMINAL_PATH)
    assert d7.sha256_file(d7.D6_TERMINAL_PATH) == (
        d7.D6_TERMINAL_SHA256
    )
    assert terminal["result_hash"] == d7.D6_TERMINAL_RESULT_HASH
    assert terminal["decision"] == "reject"
    assert terminal["first_failure"] == {
        "gate_id": 4,
        "name": "historical_blob_preamble_dependency_integrity",
    }
    assert terminal["error"] is None
    assert terminal["outcomes_opened"] is False
    assert terminal["profitability_result"] is False
    assert inheritance["d6_terminal_rejection"][
        "terminal_is_direct_child_of_seal_commit"
    ] is True
    assert inheritance["d6_terminal_rejection"][
        "parent_seal_commit"
    ] == d7.D6_SEAL_COMMIT


def test_d6_census_and_both_mechanisms_are_exactly_bound() -> None:
    payload = d7.build_preregistration()
    inheritance = payload["inheritance_proof"]
    probe = mechanism.build_probe()

    assert d7._read_canonical_json(d7.MECHANISM_PROBE_PATH) == probe
    assert inheritance["d6_post_terminal_bitcoin_grammar_census"] == (
        probe["d6_census_binding"]
    )
    assert inheritance["d6_frozen_source_mechanism"] == (
        probe["d6_mechanism_binding"]
    )
    assert inheritance["d7_mechanism_probe"] == {
        **d7.MECHANISM_PROBE_BINDING,
        "mechanism_contract_hash": d7.canonical_hash(
            probe["mechanism_contract"]
        ),
        "scenario_roster_hash": probe["synthetic_battery"][
            "scenario_roster_hash"
        ],
    }
    assert probe["d6_census_binding"]["grammar_class_counts"] == {
        "BIP_LATER_EXACT_PRE_HEADER_AFTER_NONHEADER_PREFIX": 7,
        "BIP_PREFIXED_DECIMAL_DEPENDENCY_TOKEN": 1,
        "D4_VALID": 426,
    }
    assert probe["d6_census_binding"]["unknown_grammar_count"] == 0


def test_candidate_scope_does_not_authorize_official_execution() -> None:
    payload = d7.build_preregistration()

    assert payload["protocol_version"] == (
        "psim_d7_source_preregistration_v1"
    )
    assert payload["candidate"]["id"] == "PSIM-D7"
    assert payload["candidate"]["stage"] == "source_support_only"
    assert payload["candidate"]["selection_commit"] == d7.DECISION_COMMIT
    assert payload["execution_authorization_contract"] == (
        d7.EXECUTION_AUTHORIZATION_CONTRACT
    )
    authorization = payload["execution_authorization_contract"]
    assert authorization[
        "official_source_execution_authorized_by_this_preregistration"
    ] is False
    assert authorization[
        "synthetic_mechanism_probe_authorizes_official_execution"
    ] is False
    assert authorization[
        "d6_forensic_or_source_root_reuse_allowed"
    ] is False
    assert authorization["required_before_official_source_execution"] == [
        "REVIEWED_D7_IMPLEMENTATION_COMMIT",
        "REVIEWED_D7_TEST_COMMIT",
        (
            "CANONICAL_D7_DIRECT_CHILD_EXECUTION_SEAL_BINDING_"
            "PREREGISTRATION_AND_CODE"
        ),
    ]
    assert payload["next_authorized_step"] == (
        "implement, test, review, and seal a synthetic-only PSIM-D7 "
        "source-support evaluator; this preregistration does not authorize "
        "official source execution"
    )


def test_d6_inheritance_delta_is_exact_value_and_path_bound() -> None:
    d6_payload = d6.build_preregistration()
    d7_payload = d7.build_preregistration()
    successor = contract_core(d7_payload)
    delta = d7._diff_values(contract_core(d6_payload), successor)
    inheritance = d7_payload["inheritance_proof"]

    assert len(delta) == 37
    assert tuple(sorted(delta)) == tuple(
        sorted(d7.AUTHORIZED_DELTA_PATHS)
    )
    assert delta == inheritance["authorized_delta"]
    assert inheritance["authorized_delta_paths"] == list(
        d7.AUTHORIZED_DELTA_PATHS
    )
    assert d7.canonical_hash(delta) == d7.AUTHORIZED_DELTA_HASH
    assert inheritance["authorized_delta_hash"] == (
        d7.AUTHORIZED_DELTA_HASH
    )
    assert inheritance["all_other_contract_paths_byte_equal"] is True


def test_d7_grammar_overlay_is_probe_equal_and_identity_agnostic() -> None:
    payload = d7.build_preregistration()
    overlay = payload["event_contract"]["d7_bitcoin_grammar"]
    probe = mechanism.build_probe()
    grammar = overlay["grammar_mechanism"]

    assert overlay == d7.D7_GRAMMAR_CONTRACT
    assert grammar == probe["mechanism_contract"]
    assert overlay["mechanism_version"] == probe["mechanism_version"]
    assert overlay["bitcoin_only"] is True
    assert overlay["ethereum_semantics_changed"] is False
    assert overlay["identity_conditioned_allowlist"] is False
    assert grammar["initial_parser_first"] is True
    assert grammar["header_candidate_selection"] == (
        "EXACTLY_ONE_PARSEABLE_LATER_PRE_HEADER_TOTAL"
    )
    assert grammar["header_path_binding"] == (
        "BIP_FIELD_EQUALS_PATH_PROPOSAL"
    )
    assert grammar["dependency_allowed_prefixed_token"] == (
        "BIP-[0-9]+"
    )
    assert grammar["dependency_prefix_case"] == "EXACT_UPPERCASE"
    assert grammar["unknown_grammar"] == (
        "FAIL_CLOSED_BEFORE_MODEL_OR_OUTCOMES"
    )
    assert d7.canonical_hash(overlay) == (
        d7.D7_GRAMMAR_CONTRACT_HASH
    )
    serialized = json.dumps(overlay, sort_keys=True).lower()
    assert "proposal_allowlist" not in serialized
    assert "oid_allowlist" not in serialized


def test_d6_migration_and_chunk_contracts_remain_frozen() -> None:
    d6_payload = d6.build_preregistration()
    d7_payload = d7.build_preregistration()

    assert d7_payload["event_contract"]["d6_source_mechanisms"] == (
        d6_payload["event_contract"]["d6_source_mechanisms"]
    )
    overlay = d7_payload["event_contract"]["d7_bitcoin_grammar"]
    assert overlay["d6_exact_erc_migration_restoration_frozen"] is True
    assert overlay["d6_lossless_utf8_chunk_transport_frozen"] is True
    assert overlay["d6_max_bytes_per_chunk"] == 8_192
    assert overlay["d6_max_chunks_per_event"] == 8
    assert overlay["d6_ninth_chunk_action"] == (
        "FAIL_CLOSED_NO_TRUNCATION_OR_SUMMARIZATION"
    )

    d6_transport = d6_payload["representation_contract"][
        "model_text_transport_contract"
    ]
    d7_transport = d7_payload["representation_contract"][
        "model_text_transport_contract"
    ]
    assert d7._model_transport_rebased_to_d6(d7_transport) == d6_transport
    assert d7_transport["max_bytes_per_chunk"] == 8_192
    assert d7_transport["max_chunks_per_event"] == 8
    assert d7_transport["max_bytes_per_event"] == 65_536
    assert d7_transport["full_text_reconstruction"] == (
        "BYTE_FOR_BYTE_REQUIRED"
    )
    assert "NO_TRUNCATION_OR_SUMMARIZATION" in (
        d7_transport["ninth_chunk_action"]
    )
    assert d7_transport["model_aggregation_policy"] == (
        "UNDECIDED_NOT_AUTHORIZED_BY_D7_PREREGISTRATION"
    )
    assert d7_payload["memorization_contract"][
        "model_text_chunk_aggregation_policy"
    ] == d7_transport["model_aggregation_policy"]


def test_gate_four_totality_is_unchanged_except_failure_namespace() -> None:
    d6_payload = d6.build_preregistration()
    d7_payload = d7.build_preregistration()
    d6_totality = d6_payload["source_support_contract"][
        "gate_four_totality_contract"
    ]
    d7_totality = d7_payload["source_support_contract"][
        "gate_four_totality_contract"
    ]
    rebased = copy.deepcopy(d7_totality)
    rebased["semantic_error_terminal_action"] = d6.FAILURE_ACTION

    assert rebased == d6_totality
    assert d7_totality["decision_after_complete_roster_only"] is True
    assert d7_totality[
        "event_semantics_exception_may_abort_roster_collection"
    ] is False
    assert d7_totality[
        "canonical_rejection_required_before_return_or_raise"
    ] is True
    assert d7_totality["error_report_raw_or_normalized_text_allowed"] is False
    assert d7_totality["semantic_error_terminal_action"] == (
        d7.FAILURE_ACTION
    )


def test_hydration_is_d6_equal_after_namespace_rebase() -> None:
    payload = d7.build_preregistration()
    contract = payload["source_contract"]["batch_hydration_contract"]

    assert contract == d7.BATCH_HYDRATION_CONTRACT
    assert d7._transport_contract_rebased_to_d6(contract) == (
        d6.BATCH_HYDRATION_CONTRACT
    )
    assert d7.canonical_hash(contract) == (
        d7.BATCH_HYDRATION_CONTRACT_HASH
    )
    assert contract["forbidden_transports"][-1] == (
        "D1, D2, D3, D4, D5, or D6 source-object reuse"
    )
    assert contract["one_fetch_invocation_per_replica"] is True
    assert contract["post_hydration_read"][
        "object_store_ref_and_fetch_head_snapshot_must_be_unchanged"
    ] is True


def test_fresh_d7_root_refs_and_artifacts_forbid_d6_reuse() -> None:
    source = d7.build_preregistration()["source_contract"]

    assert source["source_root"] == "/tmp/psim-d7-source"
    assert source["source_root"] != "/tmp/psim-d6-source"
    assert source["bare_repository_contract"]["sealed_ref"] == (
        "refs/psim-d7/sealed-tip"
    )
    assert source["bare_repository_contract"]["ref_roster"] == [
        "refs/heads/master",
        "refs/psim-d7/sealed-tip",
    ]
    assert all(
        value.startswith(
            (
                "results/protocol_specification_intent_maturity_d7_",
                "data/protocol_specification_intent_maturity_d7_",
            )
        )
        for value in source["artifact_paths"].values()
    )
    assert d7.build_preregistration()["execution_authorization_contract"][
        "d6_forensic_or_source_root_reuse_allowed"
    ] is False


def test_source_authority_intervals_schedules_and_gates_are_unchanged() -> None:
    d6_payload = d6.build_preregistration()
    d7_payload = d7.build_preregistration()
    d6_source = d6_payload["source_contract"]
    d7_source = d7_payload["source_contract"]

    for key in (
        "start",
        "end_exclusive",
        "card_end_exclusive",
        "clone_arguments",
        "traversal",
    ):
        assert d7_source[key] == d6_source[key]
    for d6_repository, d7_repository in zip(
        d6_source["repositories"],
        d7_source["repositories"],
    ):
        d7_rebased = copy.deepcopy(d7_repository)
        d7_rebased["sealed_ref"] = d6_repository["sealed_ref"]
        assert d7_rebased == d6_repository
        assert d7_repository["remote"] == d6_repository["remote"]
        assert d7_repository["sealed_tip"] == d6_repository["sealed_tip"]
    assert d7_payload["availability_contract"] == (
        d6_payload["availability_contract"]
    )
    assert d7_payload["split_contract"] == d6_payload["split_contract"]
    assert d7_payload["source_support_contract"]["gates_in_order"] == (
        d6_payload["source_support_contract"]["gates_in_order"]
    )
    assert d7_payload["source_support_contract"]["relation_controls"] == (
        d6_payload["source_support_contract"]["relation_controls"]
    )
    assert d7_payload["official_sources"] == d6_payload["official_sources"]


def test_preregistration_has_zero_forbidden_access() -> None:
    access = d7.build_preregistration()["inheritance_proof"][
        "preregistration_access"
    ]

    assert access == {
        "git_commands": 0,
        "network_calls": 0,
        "d6_preregistration_artifact_read": True,
        "d6_execution_seal_artifact_read": True,
        "d6_terminal_artifact_read": True,
        "d6_census_artifact_read": True,
        "d6_mechanism_probe_artifact_read": True,
        "d7_mechanism_probe_artifact_read": True,
        "d6_forensic_or_source_root_opened": False,
        "d6_source_runner_invoked": False,
        "d7_official_source_execution_invoked": False,
        "official_historical_proposal_source_opened": False,
        "market_model_outcomes_opened": False,
        "raw_official_text_published": False,
    }


def test_build_never_opens_d6_or_d7_source_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_bytes
    opened: list[str] = []

    def audited_read_bytes(path: Path) -> bytes:
        resolved = str(path)
        opened.append(resolved)
        assert not resolved.startswith("/tmp/psim-d6-source")
        assert not resolved.startswith("/tmp/psim-d7-source")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", audited_read_bytes)
    payload = d7.build_preregistration()

    assert payload["manifest_hash"]
    assert opened


def test_manifest_hash_binds_exact_core_and_replays() -> None:
    payload = d7.build_preregistration()
    core = copy.deepcopy(payload)
    manifest_hash = core.pop("manifest_hash")

    assert manifest_hash == d7.canonical_hash(core)
    assert payload == d7.build_preregistration()


def test_committed_artifact_matches_exact_builder() -> None:
    raw = RESULT_PATH.read_bytes()

    assert d7.sha256_bytes(raw) == RESULT_SHA256
    assert raw == d7.canonical_json_bytes(d7.build_preregistration())
    assert json.loads(raw) == d7.build_preregistration()


def test_write_is_canonical_idempotent_and_rejects_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(d7, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        d7,
        "build_preregistration",
        lambda: {"manifest_hash": "unit"},
    )
    destination = Path("results/unit.json")

    first = d7.write_preregistration(destination)
    first_bytes = first.read_bytes()
    second = d7.write_preregistration(destination)

    assert first == second == tmp_path / destination
    assert first_bytes == d7.canonical_json_bytes(
        {"manifest_hash": "unit"}
    )

    first.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D7 preregistration differs",
    ):
        d7.write_preregistration(destination)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/absolute.json",
        "unit.json",
        "results/nested/unit.json",
        "results/unit.txt",
    ],
)
def test_output_boundary_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(
        RuntimeError,
        match="safe repo-local result",
    ):
        d7._safe_output_path(path)


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    [
        (
            "D6_PREREGISTRATION_SHA256",
            "0" * 64,
            "preregistration authority changed",
        ),
        (
            "D6_SEAL_SHA256",
            "0" * 64,
            "execution seal authority changed",
        ),
        (
            "D6_TERMINAL_SHA256",
            "0" * 64,
            "terminal authority changed",
        ),
        (
            "D6_CENSUS_SHA256",
            "0" * 64,
            "inherited source authority changed",
        ),
        (
            "MECHANISM_PROBE_SHA256",
            "0" * 64,
            "mechanism probe authority changed",
        ),
    ],
)
def test_authority_mutations_fail_closed(
    name: str,
    replacement: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d7, name, replacement)
    with pytest.raises(RuntimeError, match=message):
        d7.build_preregistration()


def test_delta_grammar_transport_and_execution_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d7, "SOURCE_ROOT", "/tmp/psim-d7-mutated")
    with pytest.raises(
        RuntimeError,
        match="authorized source delta hash changed",
    ):
        d7.build_preregistration()

    monkeypatch.undo()
    mutated_grammar = copy.deepcopy(d7.D7_GRAMMAR_CONTRACT)
    mutated_grammar["identity_conditioned_allowlist"] = True
    monkeypatch.setattr(d7, "D7_GRAMMAR_CONTRACT", mutated_grammar)
    with pytest.raises(
        RuntimeError,
        match="authorized source delta hash changed",
    ):
        d7.build_preregistration()

    monkeypatch.undo()
    mutated_hydration = copy.deepcopy(d7.BATCH_HYDRATION_CONTRACT)
    mutated_hydration["timeout_seconds"] += 1
    monkeypatch.setattr(
        d7,
        "BATCH_HYDRATION_CONTRACT",
        mutated_hydration,
    )
    with pytest.raises(
        RuntimeError,
        match="inherited-contract delta changed",
    ):
        d7.build_preregistration()

    monkeypatch.undo()
    mutated_execution = copy.deepcopy(
        d7.EXECUTION_AUTHORIZATION_CONTRACT
    )
    mutated_execution[
        "official_source_execution_authorized_by_this_preregistration"
    ] = True
    monkeypatch.setattr(
        d7,
        "EXECUTION_AUTHORIZATION_CONTRACT",
        mutated_execution,
    )
    with pytest.raises(
        RuntimeError,
        match="inherited-contract delta changed",
    ):
        d7.build_preregistration()


def test_module_has_no_network_git_market_or_model_execution_imports() -> None:
    source = (d7.REPO_ROOT / d7.SCRIPT_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(
                alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported.isdisjoint(
        {
            "aiohttp",
            "binance",
            "ccxt",
            "git",
            "httpx",
            "models",
            "numpy",
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


def test_preregistration_document_binds_machine_contract() -> None:
    text = PREREGISTRATION_DOCUMENT_PATH.read_text(encoding="utf-8")
    payload = d7.build_preregistration()

    assert "PSIM-D7" in text
    assert d7.DECISION_COMMIT in text
    assert d7.D6_TERMINAL_RESULT_HASH in text
    assert d7.D6_CENSUS_RESULT_HASH in text
    assert d7.D6_MECHANISM_RESULT_HASH in text
    assert d7.MECHANISM_PROBE_RESULT_HASH in text
    assert payload["manifest_hash"] in text
    assert d7.AUTHORIZED_DELTA_HASH in text
    assert d7.D7_GRAMMAR_CONTRACT_HASH in text
    assert d7.BATCH_HYDRATION_CONTRACT_HASH in text
    assert d7.EXECUTION_AUTHORIZATION_CONTRACT_HASH in text
    assert RESULT_SHA256 in text
    assert d7.sha256_file(d7.SCRIPT_PATH) in text
    assert d7.sha256_file(Path(__file__).relative_to(d7.REPO_ROOT)) in text
    assert "/tmp/psim-d7-source" in text
    assert "/tmp/psim-d6-source" in text
    assert "does not authorize official source execution" in text
    assert "UNDECIDED_NOT_AUTHORIZED_BY_D7_PREREGISTRATION" in text
