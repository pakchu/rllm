from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from training import (
    audit_protocol_specification_intent_maturity_d5_event_semantics_census
    as census,
)


CENSUS_SHA256 = (
    "df2bcbef28b22d6daeb258d5c0f36b918d833b9c5fb0e5c9229a44edce4c2d59"
)
RESULT_HASH = (
    "0ca0a11f6693543dafbcf29052f2e963bf721c6e12f71f6fc9fbb1856e2dfe4a"
)


def census_bytes() -> bytes:
    return census.repository_path(census.DEFAULT_OUTPUT).read_bytes()


def census_payload() -> dict[str, Any]:
    return json.loads(census_bytes())


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


def test_census_is_canonical_and_bound_to_terminal_d5() -> None:
    raw = census_bytes()
    payload = census_payload()
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }

    assert census.DEFAULT_OUTPUT.as_posix() == (
        "results/protocol_specification_intent_maturity_d5_event_semantics_"
        "census_2026-07-26.json"
    )
    assert hashlib.sha256(raw).hexdigest() == CENSUS_SHA256
    assert raw == census.canonical_json_bytes(payload)
    assert payload["result_hash"] == RESULT_HASH
    assert payload["result_hash"] == census.canonical_hash(core)
    assert payload["protocol_version"] == census.PROTOCOL_VERSION
    assert payload["policy_id"] == (
        "PSIM-D5-POST-TERMINAL-EVENT-SEMANTICS-CENSUS"
    )
    assert payload["terminal_binding"] == {
        "commit": census.TERMINAL_COMMIT,
        "path": census.d5.DEFAULT_REJECTION_PATH.as_posix(),
        "result_hash": census.TERMINAL_RESULT_HASH,
        "runner": {
            "commit": census.RUNNER_COMMIT,
            "path": census.d5.RUNNER_PATH.as_posix(),
            "sha256": census.RUNNER_SHA256,
        },
        "sha256": census.TERMINAL_SHA256,
    }
    assert payload["semantics_binding"] == {
        "commit": census.SEMANTICS_COMMIT,
        "path": census.d5.SEMANTICS_PROBE_SCRIPT_PATH.as_posix(),
        "sha256": census.SEMANTICS_SHA256,
    }


def test_census_stayed_source_only_local_and_non_mutating() -> None:
    payload = census_payload()

    assert payload["access_boundary"] == {
        "d5_forensic_root_read": True,
        "d5_run_invoked": False,
        "market_data_accessed": False,
        "model_accessed": False,
        "network_commands": 0,
        "outcomes_accessed": False,
        "raw_text_published": False,
        "source_objects_mutated": False,
    }
    assert payload["candidate_selection"] == {
        "authorized": False,
        "d5_forensic_root_candidate_authorized": False,
        "d5_rerun_authorized": False,
        "next_step": (
            "OUTCOME_BLIND_D6_MECHANISM_DESIGN_AND_SYNTHETIC_PROBE_"
            "BEFORE_PREREGISTRATION"
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
        "source_path_rows_opened": 7_611,
    }


def test_census_exhausts_every_blob_and_event() -> None:
    observed = census_payload()["census"]

    assert (
        observed["blob_class_counts"]
        == census.EXPECTED_BLOB_CLASS_COUNTS
    )
    assert sum(observed["blob_class_counts"].values()) == (
        census.REQUESTED_BLOBS
    )
    assert observed["blob_decode_error_counts"] == {}
    assert observed["blob_decode_representatives"] == {}
    assert observed["decoded_blob_contexts"] == census.REQUESTED_BLOBS
    assert (
        observed["event_outcome_counts"]
        == census.EXPECTED_EVENT_OUTCOME_COUNTS
    )
    assert sum(observed["event_outcome_counts"].values()) == (
        census.ETHEREUM_GROUP_ROWS
    )
    assert observed["groups_evaluated"] == census.ETHEREUM_GROUP_ROWS
    assert observed["event_success_count"] == 4_430
    assert observed["event_error_count"] == 555
    assert observed["failure_type_counts"] == {"ValueError": 555}
    assert observed["event_outcome_roster_hash"] == (
        "0d89d91530566087e50ac55fbad585b5eafc0ddad01382179a03571d4314c3ad"
    )


def test_text_bound_failure_profile_is_complete_and_not_truncated() -> None:
    observed = census_payload()["census"]
    profile = observed["failure_profiles"][census.MODEL_TEXT_BOUND_ERROR]

    assert profile["count"] == 190
    assert profile["event_type_counts"] == {
        "CREATE": 103,
        "UPDATE": 87,
    }
    assert profile["proposal_count"] == 136
    assert len(profile["proposal_roster"]) == 136
    assert profile["proposal_roster_hash"] == (
        "a6765fbf7ae8cf6fe7b1341a03fe3385841cef9d7f28199c8bdcbf1f4e6ee9d6"
    )
    assert profile["event_roster_hash"] == (
        "0f299221248e66ca1eddc9cdd839cab504755537e47464c697c481544d169fd4"
    )
    assert len(profile["effective_day_counts"]) == 160
    assert min(profile["effective_day_counts"]) == "2020-02-12"
    assert max(profile["effective_day_counts"]) == "2023-10-17"
    assert profile["overflow_measurement"] == {
        "allowed_bytes_per_event": 8_192,
        "event_size_roster_hash": (
            "9bff170ce5e0518ea24a7459bad0a8f0c67fcd3a540259d9e9d2c962ada818f4"
        ),
        "max_bytes": 58_416,
        "max_model_line_changes": 943,
        "min_bytes": 8_339,
        "min_model_line_changes": 18,
        "observed_events": 190,
        "sum_bytes": 2_678_077,
        "sum_model_line_changes": 30_580,
    }
    assert observed["max_successful_model_text"]["bytes"] == 8_153
    assert profile["side_profiles"] == [
        {
            "count": 103,
            "event_type": "CREATE",
            "new_administrative_class": "NONE",
            "new_blob_class": "D4_VALID",
            "new_metadata_state": "VALID",
            "old_administrative_class": None,
            "old_blob_class": None,
            "old_metadata_state": None,
        },
        {
            "count": 87,
            "event_type": "UPDATE",
            "new_administrative_class": "NONE",
            "new_blob_class": "D4_VALID",
            "new_metadata_state": "VALID",
            "old_administrative_class": "NONE",
            "old_blob_class": "D4_VALID",
            "old_metadata_state": "VALID",
        },
    ]


def test_administrative_episode_receipts_bind_all_three_steps() -> None:
    episode = census_payload()["census"][
        "administrative_episode_census"
    ]

    assert episode["episode_count"] == 365
    assert episode["covered_administrative_quarantine_events"] == 730
    assert episode["covered_reverse_error_events"] == 365
    assert episode["all_redirect_targets_match_path_proposal"] is True
    assert episode["proposal_roster_hash"] == (
        "c12e353514f8bfdf928e3ba7c0e26b598615c8e20ce25b3dbbe31537fd169ccd"
    )
    assert episode["episode_roster_hash"] == (
        "7065c33783f1ea54af1522da7e442ec05507c38355bb98ed90daf3f87e89b0bd"
    )
    assert len(episode["per_proposal_receipt_hashes"]) == 365
    assert episode["per_proposal_receipt_hashes"][0] == {
        "proposal": 20,
        "receipt_hash": (
            "21bdd346ee5815b44a688c287732efd7a472d7184c5910965e727db81661534b"
        ),
    }
    assert episode["per_proposal_receipt_hashes"][-1] == {
        "proposal": 7_528,
        "receipt_hash": (
            "9c270e8bafb0d65ea92046b5f385c8d43f73c45aec52bcbcc5089f624467ec12"
        ),
    }
    assert episode["sequence_profiles"] == [
        {
            "class_sequence": [
                [
                    "D4_VALID",
                    "ERC_MIGRATION_REDIRECT_LOWER_PATH",
                ],
                [
                    "ERC_MIGRATION_REDIRECT_LOWER_PATH",
                    "ERC_MIGRATION_REDIRECT_UPPER_PATH",
                ],
                [
                    "ERC_MIGRATION_REDIRECT_UPPER_PATH",
                    "D4_VALID",
                ],
            ],
            "commit_sequence": [
                "0f44e2b94df4e504bb7b912f56ebd712db2ad396",
                "47ce70257fae525a427780630bd8d1903cc96e75",
                "25cdf1d059778236e28bf22d752ca48a35af91f6",
            ],
            "count": 365,
            "day_sequence": [
                "2023-10-25",
                "2023-10-25",
                "2023-10-26",
            ],
            "outcome_sequence": [
                "PASS_ADMINISTRATIVE_QUARANTINE",
                "PASS_ADMINISTRATIVE_QUARANTINE",
                "ERROR_REVERSE_ADMINISTRATIVE_MIGRATION",
            ],
        }
    ]
    representative = episode["representative"]
    assert representative["proposal"] == 20
    assert representative["path"] == "EIPS/eip-20.md"
    assert [row["new_blob_class"] for row in representative["steps"]] == [
        "ERC_MIGRATION_REDIRECT_LOWER_PATH",
        "ERC_MIGRATION_REDIRECT_UPPER_PATH",
        "D4_VALID",
    ]
    assert representative["lower_redirect"]["target_proposal"] == 20
    assert representative["upper_redirect"]["target_proposal"] == 20


def test_reverse_administrative_profile_is_exact_migration_restoration() -> (
    None
):
    profile = census_payload()["census"]["failure_profiles"][
        census.REVERSE_ADMIN_ERROR
    ]

    assert profile["count"] == 365
    assert profile["effective_day_counts"] == {"2023-10-26": 365}
    assert profile["event_type_counts"] == {"UPDATE": 365}
    assert profile["proposal_count"] == 365
    assert len(profile["proposal_roster"]) == 365
    assert profile["proposal_roster_hash"] == (
        "c12e353514f8bfdf928e3ba7c0e26b598615c8e20ce25b3dbbe31537fd169ccd"
    )
    assert profile["event_roster_hash"] == (
        "2bc4fec07ac144245a48d44e0d24f5425705eeb81d41b4268bfe4367eee54a1a"
    )
    assert profile["side_profiles"] == [
        {
            "count": 365,
            "event_type": "UPDATE",
            "new_administrative_class": "NONE",
            "new_blob_class": "D4_VALID",
            "new_metadata_state": "VALID",
            "old_administrative_class": (
                "ERC_MIGRATION_REDIRECT_UPPER_PATH"
            ),
            "old_blob_class": "ERC_MIGRATION_REDIRECT_UPPER_PATH",
            "old_metadata_state": "ADMINISTRATIVE_REDIRECT",
        }
    ]


def test_census_publishes_metadata_and_hashes_but_no_raw_text() -> None:
    keys = nested_keys(census_payload())

    assert keys.isdisjoint(
        {
            "intent_text",
            "normalized_text_delta",
            "raw_blob",
            "raw_bytes",
            "raw_lines",
            "raw_text",
        }
    )
    for profile in census_payload()["census"]["failure_profiles"].values():
        representative = profile["representative"]
        assert representative["event_id"]
        assert representative["effective_day"]
        assert representative["event_type"] in {
            "CREATE",
            "UPDATE",
            "DELETE",
        }
        assert representative["new_blob_sha256"] is not None


def test_census_has_no_market_model_or_network_client_imports() -> None:
    source = census.repository_path(
        "training/"
        "audit_protocol_specification_intent_maturity_d5_event_semantics_"
        "census.py"
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
    "path",
    [
        "/tmp/psim-d5-source/census.json",
        "/tmp/psim-d5-source/ethereum-a.git/objects/census.json",
        "/tmp/elsewhere.json",
        str(
            census.d5.REPO_ROOT.resolve()
            / "results"
            / "absolute-census.json"
        ),
        "docs/census.json",
        "results/nested/census.json",
        "results/../results/census.json",
        "results/census.txt",
    ],
)
def test_output_boundary_rejects_forensic_and_non_result_paths(
    path: str,
) -> None:
    with pytest.raises(RuntimeError, match="safe repo-local result"):
        census._safe_output_path(path)


def test_output_boundary_accepts_only_flat_json_result() -> None:
    assert census._safe_output_path(census.DEFAULT_OUTPUT) == (
        census.d5.REPO_ROOT.resolve() / census.DEFAULT_OUTPUT
    )


def test_output_boundary_rejects_symlinked_results_root(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "results").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(census.d5, "REPO_ROOT", root)

    with pytest.raises(RuntimeError, match="safe repo-local result"):
        census._safe_output_path("results/census.json")


def test_output_boundary_rejects_symlinked_result_file(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "repo"
    results = root / "results"
    results.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (results / "census.json").symlink_to(outside)
    monkeypatch.setattr(census.d5, "REPO_ROOT", root)

    with pytest.raises(RuntimeError, match="safe repo-local result"):
        census._safe_output_path("results/census.json")


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
    with pytest.raises(RuntimeError, match="event census forbids"):
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


def test_research_implication_requires_outcome_blind_d6_probe() -> None:
    assert census_payload()["research_implication"] == {
        "administrative_lifecycle_contract_is_not_total": True,
        "fixed_whole_event_model_text_bound_is_not_total": True,
        "required_administrative_successor_property": (
            "EXACT_CAUSAL_MIGRATION_EPISODE_QUARANTINE_INCLUDING_"
            "RESTORATION_WITHOUT_GENERAL_EXCEPTION"
        ),
        "required_text_successor_property": (
            "LOSSLESS_AUDIT_DIFF_WITH_DETERMINISTIC_CAUSAL_BOUNDED_"
            "MODEL_CHUNKS_NO_TRUNCATION_OR_SUMMARIZATION"
        ),
        "source_decoder_is_total_for_all_hydrated_blobs": True,
    }
