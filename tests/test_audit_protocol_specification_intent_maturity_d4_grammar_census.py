from __future__ import annotations

import ast
import hashlib
import json

import pytest

from training import (
    audit_protocol_specification_intent_maturity_d4_grammar_census as census,
)


CENSUS_SHA256 = (
    "eaa5946844bb218b1ae211c84d509c49482111af6ee165bbb54fbad26ff3b77f"
)
RESULT_HASH = (
    "85ff0c04a1fe06b34b7f214f5fd7b9a1191a4ef0dd990a7e7d002f72efe9428d"
)


def census_bytes() -> bytes:
    return census.repository_path(census.DEFAULT_OUTPUT).read_bytes()


def census_payload() -> dict:
    return json.loads(census_bytes())


def classify(
    raw: str,
    proposal_number: int = 20,
) -> tuple[str, dict]:
    encoded = raw.encode("utf-8")
    oid = census.d4.core.git_object_sha1("blob", encoded)
    return census.classify_blob(proposal_number, oid, encoded)


def test_census_is_canonical_hash_bound_to_terminal_d4() -> None:
    raw = census_bytes()
    payload = census_payload()
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }

    assert census.DEFAULT_OUTPUT.as_posix() == (
        "results/protocol_specification_intent_maturity_d4_grammar_census_"
        "2026-07-26.json"
    )
    assert hashlib.sha256(raw).hexdigest() == CENSUS_SHA256
    assert raw == census.canonical_json_bytes(payload)
    assert payload["result_hash"] == RESULT_HASH
    assert payload["result_hash"] == census.canonical_hash(core)
    assert payload["protocol_version"] == census.PROTOCOL_VERSION
    assert payload["policy_id"] == (
        "PSIM-D4-POST-TERMINAL-GRAMMAR-CENSUS"
    )
    assert payload["terminal_binding"] == {
        "commit": census.TERMINAL_COMMIT,
        "path": census.d4.DEFAULT_REJECTION_PATH.as_posix(),
        "result_hash": census.TERMINAL_RESULT_HASH,
        "sha256": census.TERMINAL_SHA256,
    }


def test_census_stayed_source_only_local_and_non_mutating() -> None:
    payload = census_payload()

    assert payload["access_boundary"] == {
        "d4_forensic_root_read": True,
        "d4_run_invoked": False,
        "market_data_accessed": False,
        "model_accessed": False,
        "network_commands": 0,
        "outcomes_accessed": False,
        "source_objects_mutated": False,
    }
    assert payload["candidate_selection"] == {
        "authorized": False,
        "next_step": (
            "OUTCOME_BLIND_D5_DESIGN_REVIEW_BEFORE_PREREGISTRATION"
        ),
        "sequential_first_exception_patching_rejected": True,
    }
    assert payload["forensic_source"] == {
        "chain_hash": census.ETHEREUM_CHAIN_HASH,
        "commit_rows": census.ETHEREUM_CHAIN_ROWS,
        "git_commands": 9_635,
        "group_hash": census.ETHEREUM_GROUP_HASH,
        "groups": census.ETHEREUM_GROUP_ROWS,
        "object_store_after_hash": census.FORENSIC_OBJECT_STORE_HASH,
        "object_store_before_hash": census.FORENSIC_OBJECT_STORE_HASH,
        "object_store_unchanged": True,
        "oid_manifest_sha256": census.OID_MANIFEST_SHA256,
        "requested_blobs": census.REQUESTED_BLOBS,
        "root": str(census.ETHEREUM_ROOT),
    }


def test_census_exhaustively_classifies_all_hydrated_blobs() -> None:
    observed = census_payload()["census"]

    assert observed["class_counts"] == census.EXPECTED_CLASS_COUNTS
    assert sum(observed["class_counts"].values()) == census.REQUESTED_BLOBS
    assert observed["unique_blob_contexts"] == census.REQUESTED_BLOBS
    assert observed["class_proposal_counts"] == {
        "D4_DUPLICATE_IDENTICAL_HEADER": 2,
        "D4_MALFORMED_HEADER_LINE": 4,
        "D4_SELF_DEPENDENCY": 1,
        "D4_VALID": 701,
        "ERC_MIGRATION_REDIRECT_LOWER_PATH": 365,
        "ERC_MIGRATION_REDIRECT_UPPER_PATH": 365,
    }
    assert observed["d4_strict_success_fraction"] == pytest.approx(
        4_440 / 5_206
    )
    assert observed["migration_redirect_blobs"] == 730
    assert observed["migration_redirect_fraction_of_failures"] == (
        pytest.approx(730 / 766)
    )
    assert observed["nonmigration_failure_blobs"] == 36


def test_migration_and_invalid_metadata_states_are_not_silently_repaired() -> (
    None
):
    observed = census_payload()["census"]

    assert observed["class_effective_day_counts"][
        "ERC_MIGRATION_REDIRECT_LOWER_PATH"
    ] == {"2023-10-25": 365}
    assert observed["class_effective_day_counts"][
        "ERC_MIGRATION_REDIRECT_UPPER_PATH"
    ] == {"2023-10-25": 365}
    assert observed["migration_redirect_proposal_roster_hash"] == (
        "c12e353514f8bfdf928e3ba7c0e26b598615c8e20ce25b3dbbe31537fd169ccd"
    )
    assert observed["nonmigration_failure_proposals"] == {
        "D4_DUPLICATE_IDENTICAL_HEADER": [2544, 3102],
        "D4_MALFORMED_HEADER_LINE": [2515, 2615, 2711, 2718],
        "D4_SELF_DEPENDENCY": [3779],
    }
    assert observed["detail_profiles"] == {
        "D4_DUPLICATE_IDENTICAL_HEADER": {
            '{"status":["Draft","Draft"]}': 7,
        },
        "D4_MALFORMED_HEADER_LINE": {
            '["requires (*optional): 155"]': 8,
            '["requires (*optional): 165 721"]': 1,
            '["requires (*optional): 2718"]': 10,
            '["requires (*optional):","replaces (*optional):"]': 1,
        },
        "D4_SELF_DEPENDENCY": {
            "2315, 3540, 3670, 3779, 4200": 9,
        },
    }


def test_redirect_classifier_is_exact_and_path_identity_bound() -> None:
    lower, lower_detail = classify(
        "This file was moved to "
        "https://github.com/ethereum/ercs/blob/master/ercs/erc-20.md\n"
    )
    upper, upper_detail = classify(
        "This file was moved to "
        "https://github.com/ethereum/ercs/blob/master/ERCS/erc-20.md\n"
    )
    mismatch, _ = classify(
        "This file was moved to "
        "https://github.com/ethereum/ercs/blob/master/ercs/erc-21.md\n"
    )
    extra_text, _ = classify(
        "This file was moved to "
        "https://github.com/ethereum/ercs/blob/master/ercs/erc-20.md\n"
        "not an exact administrative redirect\n"
    )

    assert lower == "ERC_MIGRATION_REDIRECT_LOWER_PATH"
    assert upper == "ERC_MIGRATION_REDIRECT_UPPER_PATH"
    assert lower_detail["redirect"]["target_matches_path_proposal"] is True
    assert upper_detail["redirect"]["target_matches_path_proposal"] is True
    assert mismatch == "D4_REDIRECT_TARGET_MISMATCH"
    assert extra_text == "D4_OPENING_FENCE_FAILURE_OTHER"


def test_invalid_metadata_classifier_preserves_distinct_states() -> None:
    duplicate_identical, identical_detail = classify(
        "---\n"
        "eip: 20\n"
        "status: Draft\n"
        "status: Draft\n"
        "---\n"
    )
    duplicate_conflicting, conflicting_detail = classify(
        "---\n"
        "eip: 20\n"
        "status: Draft\n"
        "status: Final\n"
        "---\n"
    )
    malformed, malformed_detail = classify(
        "---\n"
        "eip: 20\n"
        "requires (*optional): 155\n"
        "---\n"
    )
    self_dependency, self_detail = classify(
        "---\n"
        "eip: 20\n"
        "requires: 20\n"
        "---\n"
    )
    valid, valid_detail = classify(
        "---\n"
        "eip: 20\n"
        "title: Synthetic control\n"
        "status: Draft\n"
        "---\n"
        "# Abstract\n"
        "Control text.\n"
    )

    assert duplicate_identical == "D4_DUPLICATE_IDENTICAL_HEADER"
    assert identical_detail["duplicate_fields"] == {
        "status": ["Draft", "Draft"],
    }
    assert duplicate_conflicting == "D4_DUPLICATE_CONFLICTING_HEADER"
    assert conflicting_detail["duplicate_fields"] == {
        "status": ["Draft", "Final"],
    }
    assert malformed == "D4_MALFORMED_HEADER_LINE"
    assert malformed_detail["malformed_lines"] == [
        "requires (*optional): 155",
    ]
    assert self_dependency == "D4_SELF_DEPENDENCY"
    assert self_detail["requires"] == "20"
    assert valid == "D4_VALID"
    assert valid_detail == {}


def test_census_has_no_market_model_or_network_client_imports() -> None:
    source = census.repository_path(
        "training/"
        "audit_protocol_specification_intent_maturity_d4_grammar_census.py"
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
            "torch",
            "transformers",
            "urllib",
            "yfinance",
        }
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["fetch", "origin"],
        ["-C", "/tmp/source.git", "fetch", "origin"],
        ["clone", "https://example.test/repo.git"],
        ["ls-remote", "https://example.test/repo.git"],
        ["remote", "get-url", "origin"],
        ["push", "origin", "HEAD"],
        ["pull"],
        ["submodule", "update", "--remote"],
        ["-c", "protocol.file.allow=always", "cat-file", "--batch"],
        ["-C"],
        [],
    ],
)
def test_git_boundary_rejects_network_and_unrecognized_argv(
    arguments: list[str],
) -> None:
    with pytest.raises(RuntimeError, match="grammar census forbids"):
        census._assert_local_git_arguments(arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        ["cat-file", "--batch"],
        ["verify-pack", "-v", "/tmp/source.git/objects/pack/a.pack"],
        ["-C", "/tmp/source.git", "rev-parse", "HEAD^{commit}"],
        ["-C", "/tmp/source.git", "rev-list", "--first-parent", "HEAD"],
        ["-C", "/tmp/source.git", "diff-tree", "--root", "HEAD"],
        ["-C", "/tmp/source.git", "ls-tree", "-r", "HEAD"],
        ["-C", "/tmp/source.git", "for-each-ref"],
        ["-C", "/tmp/source.git", "symbolic-ref", "HEAD"],
    ],
)
def test_git_boundary_allows_only_expected_local_readers(
    arguments: list[str],
) -> None:
    census._assert_local_git_arguments(arguments)


def test_research_implication_requires_d5_design_before_selection() -> None:
    assert census_payload()["research_implication"] == {
        "administrative_redirects_are_not_specification_intent": True,
        "dependency_header_semantics_are_not_total_over_history": True,
        "preferred_next_mechanism": (
            "PATH_IDENTITY_TEXT_DIFF_WITH_EXACT_ADMINISTRATIVE_"
            "MIGRATION_QUARANTINE_AND_EXPLICIT_INVALID_METADATA_STATE"
        ),
        "strict_d4_header_parser_is_not_a_total_historical_decoder": True,
    }
