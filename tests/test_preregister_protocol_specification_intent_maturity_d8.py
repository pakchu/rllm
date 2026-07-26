from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from training import (
    preregister_protocol_specification_intent_maturity_d7 as d7,
)
from training import (
    preregister_protocol_specification_intent_maturity_d8 as d8,
)
from training import (
    probe_protocol_specification_intent_maturity_d8_relation_subcard_mechanism
    as mechanism,
)


RESULT_PATH = d8.REPO_ROOT / d8.DEFAULT_OUTPUT
PREREGISTRATION_DOCUMENT_PATH = (
    d8.REPO_ROOT
    / "docs/psim-d8-source-support-preregistration-2026-07-27.md"
)
RESULT_SHA256 = (
    "4dd083cfd54b227c6e5d373564270bcc7fb2f1002bec78a870d9d609176bb605"
)


def contract_core(payload: dict[str, object]) -> dict[str, object]:
    core = copy.deepcopy(payload)
    core.pop("manifest_hash")
    core.pop("inheritance_proof")
    return core


def test_d7_preregistration_seal_and_terminal_are_exactly_bound() -> None:
    payload = d8.build_preregistration()
    inheritance = payload["inheritance_proof"]

    d7_registration = d8._read_canonical_json(
        d8.D7_PREREGISTRATION_PATH
    )
    assert d7_registration == d7.build_preregistration()
    assert d8.sha256_file(d8.D7_PREREGISTRATION_PATH) == (
        d8.D7_PREREGISTRATION_SHA256
    )
    assert inheritance["d7_preregistration"] == {
        "path": d8.D7_PREREGISTRATION_PATH.as_posix(),
        "commit": d8.D7_PREREGISTRATION_COMMIT,
        "sha256": d8.D7_PREREGISTRATION_SHA256,
        "manifest_hash": d8.D7_PREREGISTRATION_MANIFEST_HASH,
        "contract_core_hash": d8.D7_PREREGISTRATION_CORE_HASH,
        "producer": {
            "path": d8.D7_PREREGISTRATION_SCRIPT_PATH.as_posix(),
            "sha256": d8.D7_PREREGISTRATION_SCRIPT_SHA256,
        },
        "test": {
            "path": d8.D7_PREREGISTRATION_TEST_PATH.as_posix(),
            "sha256": d8.D7_PREREGISTRATION_TEST_SHA256,
        },
        "document": {
            "path": d8.D7_PREREGISTRATION_DOCUMENT_PATH.as_posix(),
            "sha256": d8.D7_PREREGISTRATION_DOCUMENT_SHA256,
        },
    }

    seal = d8._read_canonical_json(d8.D7_SEAL_PATH)
    assert d8.sha256_file(d8.D7_SEAL_PATH) == d8.D7_SEAL_SHA256
    assert seal["seal_hash"] == d8.D7_SEAL_HASH
    assert seal["shared_commit"] == d8.D7_IMPLEMENTATION_COMMIT
    assert inheritance["d7_execution_seal"]["commit"] == (
        d8.D7_SEAL_COMMIT
    )

    terminal = d8._read_canonical_json(d8.D7_TERMINAL_PATH)
    assert d8.sha256_file(d8.D7_TERMINAL_PATH) == d8.D7_TERMINAL_SHA256
    assert terminal["result_hash"] == d8.D7_TERMINAL_RESULT_HASH
    assert terminal["decision"] == "reject"
    assert terminal["first_failure"] == {
        "gate_id": 5,
        "name": "split_annual_quarterly_unique_day_support",
    }
    assert terminal["error"] == {"type": "ValueError"}
    assert terminal["outcomes_opened"] is False
    assert terminal["profitability_result"] is False
    assert inheritance["d7_terminal_rejection"][
        "terminal_is_direct_child_of_seal_commit"
    ] is True
    assert inheritance["d7_terminal_rejection"][
        "parent_seal_commit"
    ] == d8.D7_SEAL_COMMIT


def test_d7_forensic_and_d8_mechanism_are_exactly_bound() -> None:
    payload = d8.build_preregistration()
    inheritance = payload["inheritance_proof"]
    probe = mechanism.build_probe()

    assert d8._read_canonical_json(d8.MECHANISM_PROBE_PATH) == probe
    assert inheritance["d7_post_terminal_forensic"] == (
        probe["d7_authority"]["forensic"]
    )
    assert inheritance["d7_observed_source_cardinality"] == {
        "overflow_card_cells": 24,
        "first_overflow_relation_units": 143,
        "maximum_relation_units": 1_221,
    }
    assert inheritance["d8_mechanism_probe"] == {
        **d8.MECHANISM_PROBE_BINDING,
        "mechanism_contract_hash": d8.canonical_hash(
            probe["mechanism_contract"]
        ),
    }
    assert probe["synthetic_battery"]["scenario_count"] == 12
    assert probe["synthetic_battery"]["scenario_roster_hash"] == (
        d8.MECHANISM_PROBE_SCENARIO_ROSTER_HASH
    )


def test_candidate_scope_is_terminal_and_does_not_authorize_execution() -> None:
    payload = d8.build_preregistration()
    candidate = payload["candidate"]
    authorization = payload["execution_authorization_contract"]

    assert payload["protocol_version"] == (
        "psim_d8_source_preregistration_v1"
    )
    assert candidate["id"] == "PSIM-D8"
    assert candidate["stage"] == "source_support_only"
    assert candidate["selection_commit"] == d8.DECISION_COMMIT
    assert candidate["last_source_representation_successor"] is True
    assert candidate["d9_allowed_after_source_failure"] is False
    assert authorization == d8.EXECUTION_AUTHORIZATION_CONTRACT
    assert authorization[
        "official_source_execution_authorized_by_this_preregistration"
    ] is False
    assert authorization[
        "synthetic_mechanism_probe_authorizes_official_execution"
    ] is False
    assert authorization[
        "d7_forensic_or_source_root_reuse_allowed"
    ] is False
    assert authorization["d8_is_last_source_representation_successor"] is True
    assert authorization[
        "d9_source_successor_allowed_after_d8_failure"
    ] is False
    assert authorization["required_before_official_source_execution"] == [
        "REVIEWED_D8_IMPLEMENTATION_COMMIT",
        "REVIEWED_D8_TEST_COMMIT",
        (
            "CANONICAL_D8_DIRECT_CHILD_EXECUTION_SEAL_BINDING_"
            "PREREGISTRATION_AND_CODE"
        ),
    ]
    assert "NO_D9" in d8.FAILURE_ACTION
    assert "/tmp/psim-d9-source" not in json.dumps(payload)


def test_d7_inheritance_delta_is_exact_value_and_path_bound() -> None:
    d7_payload = d7.build_preregistration()
    d8_payload = d8.build_preregistration()
    delta = d8._diff_values(
        contract_core(d7_payload),
        contract_core(d8_payload),
    )

    assert tuple(sorted(delta)) == tuple(
        sorted(d8.AUTHORIZED_DELTA_PATHS)
    )
    assert delta == d8_payload["inheritance_proof"]["authorized_delta"]
    assert d8.canonical_hash(delta) == d8.AUTHORIZED_DELTA_HASH
    assert d8_payload["inheritance_proof"][
        "authorized_delta_hash"
    ] == d8.AUTHORIZED_DELTA_HASH
    assert d8_payload["inheritance_proof"][
        "d7_authority_binding_hash"
    ] == d8.D7_AUTHORITY_BINDING_HASH
    assert d8.canonical_hash(d8._d7_authority_binding()) == (
        d8.D7_AUTHORITY_BINDING_HASH
    )
    assert d8_payload["inheritance_proof"][
        "all_other_contract_paths_byte_equal"
    ] is True


def test_relation_subcard_contract_is_probe_equal_and_lossless() -> None:
    payload = d8.build_preregistration()
    daily = payload["daily_relation_contract"]
    representation = payload["representation_contract"]
    contract = daily["relation_subcard_contract"]

    assert contract == d8.RELATION_SUBCARD_CONTRACT
    assert contract == mechanism.build_probe()["mechanism_contract"]
    assert d8.canonical_hash(contract) == (
        d8.RELATION_SUBCARD_CONTRACT_HASH
    )
    assert contract["logical_daily_card_count"] == (
        "EXACTLY_ONE_PER_SCHEDULE_AND_DECISION_DAY"
    )
    assert contract["logical_daily_relation_roster"] == (
        "EXACT_D7_ORDERED_COMPLETE_RELATION_UNITS"
    )
    assert contract["maximum_model_relation_units_per_subcard"] == 64
    assert contract["control_denominator"] == "UNIQUE_LOGICAL_DECISION_DAYS"
    assert contract["dropping_sampling_summarization_allowed"] is False
    assert contract["cap_raise_allowed"] is False
    assert contract["market_or_outcome_dependent_partition_allowed"] is False
    assert daily["maximum_model_events_per_card"] is None
    assert daily["maximum_model_relation_units_per_subcard"] == 64
    assert representation["logical_daily_card_payload_model_visible"] is False
    assert representation["relation_subcard_manifest_model_visible"] is False


def test_model_input_and_aggregation_remain_unauthorized() -> None:
    payload = d8.build_preregistration()
    daily = payload["daily_relation_contract"]
    representation = payload["representation_contract"]
    memorization = payload["memorization_contract"]
    marker = "UNDECIDED_NOT_AUTHORIZED_BY_D8_SOURCE_PREREGISTRATION"

    assert representation["single_model_single_call_per_card"] is False
    assert representation["model_call_granularity"] == marker
    assert representation["model_text_transport_contract"][
        "model_aggregation_policy"
    ] == marker
    assert daily["model_text_transport_contract"][
        "model_aggregation_policy"
    ] == marker
    assert memorization["model_text_chunk_aggregation_policy"] == marker
    assert representation["later_model_input"] == (
        "VERIFIED_SUBCARD_SLICE_ONLY_UNDER_SEPARATE_PREREGISTRATION"
    )


def test_d7_source_parser_split_schedule_gate_and_controls_are_frozen() -> None:
    d7_payload = d7.build_preregistration()
    d8_payload = d8.build_preregistration()

    for key in (
        "availability_contract",
        "boundary_reset_contract",
        "bucket_contract",
        "event_contract",
        "excluded_feasibility_probe",
        "forbidden_access_contract",
        "official_sources",
        "parser_contract",
        "split_contract",
    ):
        assert d8_payload[key] == d7_payload[key]
    assert d8_payload["source_support_contract"]["gates_in_order"] == (
        d7_payload["source_support_contract"]["gates_in_order"]
    )
    assert d8_payload["source_support_contract"]["relation_controls"] == (
        d7_payload["source_support_contract"]["relation_controls"]
    )
    assert d8_payload["source_support_contract"][
        "control_sensitivity_metric"
    ]["denominator"] == d7_payload["source_support_contract"][
        "control_sensitivity_metric"
    ]["denominator"]


def test_hydration_and_text_transport_are_d7_equal_after_rebase() -> None:
    assert d8._transport_contract_rebased_to_d7(
        d8.BATCH_HYDRATION_CONTRACT
    ) == d7.BATCH_HYDRATION_CONTRACT
    assert d8._model_transport_rebased_to_d7(
        d8.MODEL_TEXT_TRANSPORT_CONTRACT
    ) == d7.MODEL_TEXT_TRANSPORT_CONTRACT


def test_fresh_d8_root_refs_and_artifacts_forbid_d7_reuse() -> None:
    payload = d8.build_preregistration()
    source = payload["source_contract"]

    assert source["source_root"] == "/tmp/psim-d8-source"
    assert source["source_root"] != d7.SOURCE_ROOT
    assert source["artifact_paths"] == d8.ARTIFACT_PATHS
    assert source["bare_repository_contract"]["sealed_ref"] == (
        "refs/psim-d8/sealed-tip"
    )
    assert source["bare_repository_contract"]["source_traversal_ref"] == (
        "refs/psim-d8/sealed-tip"
    )
    assert source["bare_repository_contract"]["ref_roster"] == [
        "refs/heads/master",
        "refs/psim-d8/sealed-tip",
    ]
    assert {
        row["sealed_ref"] for row in source["repositories"]
    } == {"refs/psim-d8/sealed-tip"}
    assert (
        "D1, D2, D3, D4, D5, D6, or D7 source-object reuse"
        in source["batch_hydration_contract"]["forbidden_transports"]
    )


def test_preregistration_has_zero_forbidden_access() -> None:
    access = d8.build_preregistration()["inheritance_proof"][
        "preregistration_access"
    ]

    assert access == {
        "git_commands": 0,
        "network_calls": 0,
        "d7_preregistration_artifact_read": True,
        "d7_execution_seal_artifact_read": True,
        "d7_terminal_artifact_read": True,
        "d7_forensic_artifact_read_via_mechanism_probe": True,
        "d8_mechanism_probe_artifact_read": True,
        "d7_forensic_or_source_root_opened": False,
        "d7_source_runner_invoked": False,
        "d8_source_root_created_or_opened": False,
        "d8_official_source_execution_invoked": False,
        "official_historical_proposal_source_opened": False,
        "market_model_outcomes_opened": False,
    }


def test_build_never_opens_any_psim_source_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_bytes
    opened: list[str] = []

    def audited_read_bytes(path: Path) -> bytes:
        resolved = str(path)
        opened.append(resolved)
        assert not resolved.startswith("/tmp/psim-")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", audited_read_bytes)
    payload = d8.build_preregistration()

    assert payload["manifest_hash"]
    assert opened


def test_build_does_not_invoke_d7_audit_or_source_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("forbidden source/audit execution")

    monkeypatch.setattr(mechanism.d7_audit, "run_audit", forbidden)
    monkeypatch.setattr(mechanism.d7_audit.runner, "run_official", forbidden)

    assert d8.build_preregistration()["manifest_hash"]


def test_manifest_hash_binds_exact_core_and_replays() -> None:
    payload = d8.build_preregistration()
    core = copy.deepcopy(payload)
    manifest_hash = core.pop("manifest_hash")

    assert manifest_hash == d8.canonical_hash(core)
    assert payload == d8.build_preregistration()


def test_committed_artifact_matches_exact_builder() -> None:
    raw = RESULT_PATH.read_bytes()

    assert d8.sha256_bytes(raw) == RESULT_SHA256
    assert raw == d8.canonical_json_bytes(d8.build_preregistration())
    assert json.loads(raw) == d8.build_preregistration()


def test_write_is_canonical_idempotent_and_rejects_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(d8, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        d8,
        "build_preregistration",
        lambda: {"manifest_hash": "unit"},
    )
    destination = Path("results/unit.json")

    first = d8.write_preregistration(destination)
    first_bytes = first.read_bytes()
    second = d8.write_preregistration(destination)

    assert first == second == tmp_path / destination
    assert first_bytes == d8.canonical_json_bytes(
        {"manifest_hash": "unit"}
    )

    first.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D8 preregistration differs",
    ):
        d8.write_preregistration(destination)


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
    with pytest.raises(RuntimeError, match="safe repo-local result"):
        d8._safe_output_path(path)


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    [
        (
            "D7_PREREGISTRATION_SHA256",
            "0" * 64,
            "preregistration authority changed",
        ),
        (
            "D7_PREREGISTRATION_COMMIT",
            "f" * 40,
            "authorized source delta hash changed",
        ),
        (
            "D7_SEAL_SHA256",
            "0" * 64,
            "execution seal authority changed",
        ),
        (
            "D7_SEAL_COMMIT",
            "f" * 40,
            "authorized source delta hash changed",
        ),
        (
            "D7_TERMINAL_SHA256",
            "0" * 64,
            "terminal authority changed",
        ),
        (
            "D7_TERMINAL_COMMIT",
            "f" * 40,
            "authorized source delta hash changed",
        ),
        (
            "MECHANISM_PROBE_SHA256",
            "0" * 64,
            "mechanism probe authority changed",
        ),
        (
            "MECHANISM_PROBE_SCRIPT_SHA256",
            "0" * 64,
            "mechanism probe authority changed",
        ),
        (
            "MECHANISM_PROBE_TEST_SHA256",
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
    monkeypatch.setattr(d8, name, replacement)
    with pytest.raises(RuntimeError, match=message):
        d8.build_preregistration()


def test_delta_subcard_transport_and_execution_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d8, "SOURCE_ROOT", "/tmp/psim-d8-mutated")
    with pytest.raises(
        RuntimeError,
        match="authorized source delta hash changed",
    ):
        d8.build_preregistration()

    monkeypatch.undo()
    mutated_subcard = copy.deepcopy(d8.RELATION_SUBCARD_CONTRACT)
    mutated_subcard["cap_raise_allowed"] = True
    monkeypatch.setattr(d8, "RELATION_SUBCARD_CONTRACT", mutated_subcard)
    with pytest.raises(
        RuntimeError,
        match="mechanism probe authority changed",
    ):
        d8.build_preregistration()

    monkeypatch.undo()
    mutated_hydration = copy.deepcopy(d8.BATCH_HYDRATION_CONTRACT)
    mutated_hydration["timeout_seconds"] += 1
    monkeypatch.setattr(
        d8,
        "BATCH_HYDRATION_CONTRACT",
        mutated_hydration,
    )
    with pytest.raises(
        RuntimeError,
        match="inherited-contract delta changed",
    ):
        d8.build_preregistration()

    monkeypatch.undo()
    mutated_execution = copy.deepcopy(
        d8.EXECUTION_AUTHORIZATION_CONTRACT
    )
    mutated_execution[
        "official_source_execution_authorized_by_this_preregistration"
    ] = True
    monkeypatch.setattr(
        d8,
        "EXECUTION_AUTHORIZATION_CONTRACT",
        mutated_execution,
    )
    with pytest.raises(
        RuntimeError,
        match="inherited-contract delta changed",
    ):
        d8.build_preregistration()


def test_module_has_no_network_git_market_or_model_execution_imports() -> None:
    source = (d8.REPO_ROOT / d8.SCRIPT_PATH).read_text(encoding="utf-8")
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
            "socket",
            "subprocess",
            "torch",
            "transformers",
            "urllib",
            "yfinance",
        }
    )


def test_preregistration_document_binds_machine_contract() -> None:
    text = PREREGISTRATION_DOCUMENT_PATH.read_text(encoding="utf-8")
    payload = d8.build_preregistration()

    assert "PSIM-D8" in text
    assert d8.DECISION_COMMIT in text
    assert d8.D7_TERMINAL_RESULT_HASH in text
    assert mechanism.D7_FORENSIC_RESULT_HASH in text
    assert d8.MECHANISM_PROBE_RESULT_HASH in text
    assert payload["manifest_hash"] in text
    assert d8.AUTHORIZED_DELTA_HASH in text
    assert d8.RELATION_SUBCARD_CONTRACT_HASH in text
    assert d8.BATCH_HYDRATION_CONTRACT_HASH in text
    assert d8.EXECUTION_AUTHORIZATION_CONTRACT_HASH in text
    assert RESULT_SHA256 in text
    assert d8.sha256_file(d8.SCRIPT_PATH) in text
    assert d8.sha256_file(Path(__file__).relative_to(d8.REPO_ROOT)) in text
    assert "/tmp/psim-d8-source" in text
    assert "/tmp/psim-d7-source" in text
    assert "does not authorize official source execution" in text
    assert "UNDECIDED_NOT_AUTHORIZED_BY_D8_SOURCE_PREREGISTRATION" in text
    assert "NO D9" in text
