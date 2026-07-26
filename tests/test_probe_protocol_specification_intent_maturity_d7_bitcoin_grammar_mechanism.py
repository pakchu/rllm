from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from training import (
    probe_protocol_specification_intent_maturity_d7_bitcoin_grammar_mechanism
    as probe,
)


PROBE_SHA256 = (
    "2a549e6acfac2127527272ffe69986177b5e36f68f66623c6921ababac35ee94"
)
RESULT_HASH = (
    "832b1327d19b29f44f4fbd76dac312e001a7da19eb813ce41277d64a45492371"
)
SCENARIO_ROSTER_HASH = (
    "96c44d9f3c1cc2b84ce69fd3787195b095a2a3da3427ef4b59eb319383c9aff0"
)


def probe_bytes() -> bytes:
    return probe.repository_path(probe.DEFAULT_OUTPUT).read_bytes()


def probe_payload() -> dict[str, Any]:
    return json.loads(probe_bytes())


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(nested_keys(nested))
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for nested in value:
            keys.update(nested_keys(nested))
    return keys


def synthetic_bip(
    proposal: int = 900,
    *,
    prefix: str = "",
    dependency: str | None = None,
    second: int | None = None,
) -> bytes:
    return probe._synthetic_bip(
        proposal,
        prefix=prefix,
        dependency=dependency,
        second_header_proposal=second,
    )


def test_probe_is_canonical_and_hash_bound() -> None:
    raw = probe_bytes()
    payload = probe_payload()
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }

    assert hashlib.sha256(raw).hexdigest() == PROBE_SHA256
    assert raw == probe.canonical_json_bytes(payload)
    assert payload["result_hash"] == RESULT_HASH
    assert payload["result_hash"] == probe.canonical_hash(core)
    assert probe.canonical_json_bytes(probe.build_probe()) == raw
    assert payload["protocol_version"] == probe.PROTOCOL_VERSION
    assert payload["mechanism_version"] == probe.MECHANISM_VERSION
    assert payload["policy_id"] == (
        "PSIM-D7-SYNTHETIC-GRAMMAR-MECHANISM-PROBE"
    )
    assert payload["synthetic_only"] is True
    assert payload["selection_scope"] == (
        "AUTHORIZE_D7_PREREGISTRATION_ONLY_NOT_OFFICIAL_SOURCE_EXECUTION"
    )


def test_probe_binds_census_and_keeps_d6_mechanisms_frozen() -> None:
    payload = probe_payload()

    assert payload["d6_census_binding"] == {
        "commit": probe.D6_CENSUS_COMMIT,
        "document": {
            "path": probe.D6_CENSUS_DOCUMENT_PATH.as_posix(),
            "sha256": probe.D6_CENSUS_DOCUMENT_SHA256,
        },
        "grammar_class_counts": (
            probe.d6_census.EXPECTED_GRAMMAR_CLASS_COUNTS
        ),
        "path": probe.D6_CENSUS_PATH.as_posix(),
        "result_hash": probe.D6_CENSUS_RESULT_HASH,
        "script": {
            "path": probe.D6_CENSUS_SCRIPT_PATH.as_posix(),
            "sha256": probe.D6_CENSUS_SCRIPT_SHA256,
        },
        "sha256": probe.D6_CENSUS_SHA256,
        "test": {
            "path": probe.D6_CENSUS_TEST_PATH.as_posix(),
            "sha256": probe.D6_CENSUS_TEST_SHA256,
        },
        "unknown_grammar_count": 0,
    }
    assert payload["d6_mechanism_binding"] == {
        "commit": probe.D6_MECHANISM_COMMIT,
        "document": {
            "path": probe.D6_MECHANISM_DOCUMENT_PATH.as_posix(),
            "sha256": probe.D6_MECHANISM_DOCUMENT_SHA256,
        },
        "mechanism_version": probe.D6_MECHANISM_VERSION,
        "path": probe.D6_MECHANISM_PATH.as_posix(),
        "result_hash": probe.D6_MECHANISM_RESULT_HASH,
        "script": {
            "path": probe.D6_MECHANISM_SCRIPT_PATH.as_posix(),
            "sha256": probe.D6_MECHANISM_SCRIPT_SHA256,
        },
        "sha256": probe.D6_MECHANISM_SHA256,
        "source_mechanism_contract_hash": (
            probe.D6_SOURCE_MECHANISM_CONTRACT_HASH
        ),
        "test": {
            "path": probe.D6_MECHANISM_TEST_PATH.as_posix(),
            "sha256": probe.D6_MECHANISM_TEST_SHA256,
        },
    }


def test_probe_is_source_model_market_and_outcome_blind() -> None:
    payload = probe_payload()

    assert payload["access_boundary"] == {
        "d6_census_artifact_read": True,
        "d6_forensic_root_accessed": False,
        "d6_run_invoked": False,
        "external_network_accessed_by_probe": False,
        "historical_proposal_text_accessed": False,
        "market_data_accessed": False,
        "model_accessed": False,
        "official_historical_proposal_source_accessed": False,
        "outcomes_accessed": False,
        "raw_official_text_published": False,
    }
    source = probe.repository_path(
        "training/"
        "probe_protocol_specification_intent_maturity_d7_bitcoin_grammar_"
        "mechanism.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert "build_census" not in calls
    assert "run_official" not in calls
    assert "collect_commit_chain" not in calls
    assert "collect_proposal_groups" not in calls
    assert "subprocess" not in source
    assert "/tmp/psim-d6-source" not in source


def test_mechanism_contract_is_category_level_and_fail_closed() -> None:
    payload = probe_payload()

    assert payload["candidate"] == {
        "id": "PSIM-D7",
        "name": (
            "UNIQUE_LATER_BIP_HEADER_AND_PREFIXED_DEPENDENCY_GRAMMAR"
        ),
        "source_representation_successor": True,
    }
    assert payload["mechanism_contract"] == {
        "dependency_allowed_bare_token": "[0-9]+",
        "dependency_allowed_prefixed_token": "BIP-[0-9]+",
        "dependency_prefix_case": "EXACT_UPPERCASE",
        "dependency_prefix_effect": (
            "INTEGER_EDGE_VALIDATION_ONLY_SOURCE_TEXT_UNCHANGED"
        ),
        "dependency_token_outer_whitespace": (
            "INHERIT_D6_STRIP_SURROUNDING_SP_HTAB"
        ),
        "dependency_unknown_token": "FAIL_CLOSED",
        "fallback_pre_fence_grammar": (
            "BALANCED_NON_NESTED_EXACT_PAIRS"
        ),
        "fallback_pre_header_blocks_before_selected": "NONE",
        "header_candidate_parser": "UNCHANGED_FROZEN_BIP_PARSER",
        "header_candidate_selection": (
            "EXACTLY_ONE_PARSEABLE_LATER_PRE_HEADER_TOTAL"
        ),
        "header_path_binding": "BIP_FIELD_EQUALS_PATH_PROPOSAL",
        "header_prefix_and_body_retention": "FULL_NORMALIZED_TEXT_RETAINED",
        "identity_conditioned_allowlist": False,
        "initial_parser_first": True,
        "later_header_authorized_initial_error": (
            probe.EXACT_INITIAL_ERROR
        ),
        "multiple_parseable_headers": "FAIL_CLOSED",
        "unknown_grammar": "FAIL_CLOSED_BEFORE_MODEL_OR_OUTCOMES",
    }
    forbidden_identity_keys = {
        "blob_oid",
        "commit_oid",
        "event_id",
        "proposal_number",
        "proposal_roster",
        "raw_text",
        "normalized_text",
    }
    assert nested_keys(payload).isdisjoint(forbidden_identity_keys)


def test_synthetic_battery_is_complete_and_frozen() -> None:
    battery = probe_payload()["synthetic_battery"]

    assert battery["all_passed"] is True
    assert battery["scenario_count"] == 23
    assert len(battery["scenarios"]) == 23
    assert all(row["passed"] is True for row in battery["scenarios"])
    assert battery["scenario_roster_hash"] == SCENARIO_ROSTER_HASH
    assert battery["outcome_counts"] == {
        "ERROR_DEPENDENCY_COUNT": 1,
        "ERROR_DEPENDENCY_DUPLICATE": 1,
        "ERROR_DEPENDENCY_SELF": 1,
        "ERROR_DEPENDENCY_TOKEN_GRAMMAR": 3,
        "ERROR_INITIAL_FAILURE_NOT_AUTHORIZED_FOR_FALLBACK": 1,
        "ERROR_LATER_HEADER_NOT_UNIQUE": 2,
        "ERROR_LATER_HEADER_PATH_OR_PREFIX": 1,
        "ERROR_LATER_HEADER_PREFIX_FENCE": 2,
        "ERROR_PRE_FENCE_GRAMMAR": 4,
        "ERROR_STRICT_SOURCE_NORMALIZATION": 2,
        probe.INITIAL_ANCHOR: 4,
        probe.LATER_PRE_ANCHOR: 1,
    }


def test_frozen_initial_parser_remains_first() -> None:
    selected = probe.select_bip_header_d7(900, synthetic_bip())

    assert selected["audit_receipt"]["anchor"] == probe.INITIAL_ANCHOR
    assert selected["audit_receipt"]["path_proposal_matches"] is True
    assert selected["header"]["bip"] == "900"
    assert selected["normalized_lines"][0] == "<pre>"


def test_unique_later_header_retains_prefix_and_body() -> None:
    prefix = "Synthetic preface\nSecond synthetic line\n"
    raw = synthetic_bip(prefix=prefix)
    selected = probe.select_bip_header_d7(900, raw)

    assert selected["audit_receipt"]["anchor"] == probe.LATER_PRE_ANCHOR
    assert selected["audit_receipt"]["parseable_later_header_count"] == 1
    assert selected["audit_receipt"]["prefix_nonblank_line_count"] == 2
    assert selected["normalized_lines"][:2] == (
        "Synthetic preface",
        "Second synthetic line",
    )
    assert selected["normalized_lines"][-2:] == ("Body", "")
    assert selected["header"]["bip"] == "900"


def test_later_header_requires_unique_parseable_total_and_path_match() -> None:
    with pytest.raises(
        probe.D7MechanismError,
        match="ERROR_LATER_HEADER_NOT_UNIQUE",
    ):
        probe.select_bip_header_d7(
            900,
            synthetic_bip(
                prefix="Synthetic preface\n",
                second=900,
            ),
        )
    with pytest.raises(
        probe.D7MechanismError,
        match="ERROR_LATER_HEADER_NOT_UNIQUE",
    ):
        probe.select_bip_header_d7(
            900,
            synthetic_bip(
                prefix="Synthetic preface\n",
                second=901,
            ),
        )
    with pytest.raises(
        probe.D7MechanismError,
        match="ERROR_LATER_HEADER_PATH_OR_PREFIX",
    ):
        probe.select_bip_header_d7(
            900,
            synthetic_bip(901, prefix="Synthetic preface\n"),
        )


def test_only_exact_historical_initial_error_authorizes_fallback() -> None:
    too_many_blanks = (
        "\n" * 4 + synthetic_bip().decode("utf-8")
    ).encode("utf-8")

    with pytest.raises(
        probe.D7MechanismError,
        match="ERROR_INITIAL_FAILURE_NOT_AUTHORIZED_FOR_FALLBACK",
    ):
        probe.select_bip_header_d7(900, too_many_blanks)


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (
            b"Synthetic preface\n<pre>\nUnclosed block\n"
            + synthetic_bip(),
            "ERROR_PRE_FENCE_GRAMMAR",
        ),
        (
            synthetic_bip(prefix="Synthetic preface\n")
            + b"<pre>\nUnclosed tail\n",
            "ERROR_PRE_FENCE_GRAMMAR",
        ),
        (
            synthetic_bip(prefix="Synthetic preface\n</pre>\n"),
            "ERROR_PRE_FENCE_GRAMMAR",
        ),
        (
            (
                "Synthetic preface\n<pre>\n<pre>\n"
                "Nested synthetic block\n</pre>\n</pre>\n"
            ).encode()
            + synthetic_bip(),
            "ERROR_PRE_FENCE_GRAMMAR",
        ),
        (
            (
                "Synthetic preface\n<pre>\nMalformed field\n</pre>\n"
            ).encode()
            + synthetic_bip(),
            "ERROR_LATER_HEADER_PREFIX_FENCE",
        ),
        (
            (
                "Synthetic preface\n"
                "<pre>\n"
                "  BIP: 900\n"
                "  BIP: 900\n"
                "</pre>\n"
            ).encode()
            + synthetic_bip(),
            "ERROR_LATER_HEADER_PREFIX_FENCE",
        ),
    ],
)
def test_fallback_rejects_malformed_or_ambiguous_fence_prefix(
    raw: bytes,
    code: str,
) -> None:
    with pytest.raises(probe.D7MechanismError, match=code):
        probe.select_bip_header_d7(900, raw)


def test_dependency_prefix_changes_edge_validation_not_source_text() -> None:
    dependency = 899
    token = f"BIP-{dependency}"
    raw = synthetic_bip(dependency=token)
    selected = probe.select_bip_header_d7(900, raw)
    edges = probe.parse_bip_dependencies_d7(
        selected["header"],
        self_id=900,
    )

    assert selected["header"]["requires"] == token
    assert any(token in line for line in selected["normalized_lines"])
    assert edges["requires"] == (dependency,)
    assert probe.decode_bitcoin_semantics_d7(
        900,
        raw,
    )["dependency_edge_count"] == 1


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("bip-899", "ERROR_DEPENDENCY_TOKEN_GRAMMAR"),
        ("BIP -899", "ERROR_DEPENDENCY_TOKEN_GRAMMAR"),
        ("BIP-899-901", "ERROR_DEPENDENCY_TOKEN_GRAMMAR"),
        ("BIP-900", "ERROR_DEPENDENCY_SELF"),
        ("899, BIP-899", "ERROR_DEPENDENCY_DUPLICATE"),
        ("BIP-0", "ERROR_DEPENDENCY_POSITIVE_DECIMAL"),
        ("BIP-x", "ERROR_DEPENDENCY_TOKEN_GRAMMAR"),
        ("BIP-899\n901", "ERROR_DEPENDENCY_MULTILINE"),
    ],
)
def test_dependency_grammar_rejects_noncontract_tokens(
    value: str,
    code: str,
) -> None:
    with pytest.raises(probe.D7MechanismError, match=code):
        probe.parse_dependency_ids_d7(value, self_id=900)


def test_cross_style_dependencies_share_duplicate_and_sorting_rules() -> None:
    assert probe.parse_dependency_ids_d7(
        "901, BIP-899",
        self_id=900,
    ) == (899, 901)
    assert probe.parse_dependency_ids_d7(
        "\tBIP-899 ",
        self_id=900,
    ) == (899,)
    with pytest.raises(
        probe.D7MechanismError,
        match="ERROR_DEPENDENCY_DUPLICATE",
    ):
        probe.parse_dependency_ids_d7(
            "899, BIP-899",
            self_id=900,
        )


def test_strict_source_normalization_is_not_relaxed() -> None:
    for raw in (
        synthetic_bip() + b"\x00",
        synthetic_bip() + b"\xff",
    ):
        with pytest.raises(
            probe.D7MechanismError,
            match="ERROR_STRICT_SOURCE_NORMALIZATION",
        ):
            probe.select_bip_header_d7(900, raw)


@pytest.mark.parametrize(
    "path",
    [
        Path("/tmp/probe.json"),
        Path("results/nested/probe.json"),
        Path("results/probe.txt"),
        Path("probe.json"),
    ],
)
def test_output_guard_rejects_unsafe_paths(path: Path) -> None:
    with pytest.raises(RuntimeError):
        probe._safe_output_path(path)
