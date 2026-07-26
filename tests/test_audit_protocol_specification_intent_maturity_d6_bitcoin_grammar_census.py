from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from training import (
    audit_protocol_specification_intent_maturity_d6_bitcoin_grammar_census
    as census,
)


CENSUS_SHA256 = (
    "8bfe4a6c44a4c5381bb98caf2ffea57b42f2b3d77caec9e656895336b72d0217"
)
RESULT_HASH = (
    "7ef74a017f8c0c1eb416608dcf59c2ce74af6587f5a71203b53e846d31c039ed"
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


def raw_bip(
    proposal: int,
    *,
    prefix: str = "",
    requires: str | None = None,
    duplicate_header: bool = False,
) -> bytes:
    dependency = (
        "" if requires is None else f"  Requires: {requires}\n"
    )
    header = (
        "<pre>\n"
        f"  BIP: {proposal}\n"
        "  Title: Synthetic grammar fixture\n"
        f"{dependency}"
        "</pre>\n"
    )
    duplicate = header if duplicate_header else ""
    return f"{prefix}{header}{duplicate}Body\n".encode("utf-8")


def classify(
    raw: bytes,
    *,
    path_proposal: int = 900,
) -> tuple[str, dict[str, Any], str | None]:
    oid = census.d6.core.git_object_sha1("blob", raw)
    return census.classify_bitcoin_blob(
        path_proposal,
        oid,
        raw,
    )


def test_census_is_canonical_and_bound_to_terminal_d6() -> None:
    raw = census_bytes()
    payload = census_payload()
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }

    assert hashlib.sha256(raw).hexdigest() == CENSUS_SHA256
    assert raw == census.d6.canonical_json_bytes(payload)
    assert payload["result_hash"] == RESULT_HASH
    assert payload["result_hash"] == census.d6.canonical_hash(core)
    assert payload["protocol_version"] == census.PROTOCOL_VERSION
    assert payload["policy_id"] == census.POLICY_ID
    assert payload["terminal_binding"] == {
        "commit": census.TERMINAL_COMMIT,
        "path": census.d6.DEFAULT_REJECTION_PATH.as_posix(),
        "result_hash": census.TERMINAL_RESULT_HASH,
        "runner": {
            "commit": census.RUNNER_COMMIT,
            "path": census.d6.RUNNER_PATH.as_posix(),
            "sha256": census.RUNNER_SHA256,
        },
        "seal": {
            "commit": census.SEAL_COMMIT,
            "path": census.d6.EXECUTION_SEAL_PATH.as_posix(),
            "sha256": census.SEAL_SHA256,
        },
        "sha256": census.TERMINAL_SHA256,
    }


def test_census_stayed_source_only_and_quarantined_drifted_replica() -> None:
    payload = census_payload()

    assert payload["access_boundary"] == {
        "d6_forensic_root_read": True,
        "d6_run_invoked": False,
        "market_data_accessed": False,
        "model_accessed": False,
        "network_commands": 0,
        "outcomes_accessed": False,
        "post_terminal_replica_a_drift_detected": True,
        "quarantined_object_store_read_by_final_census": False,
        "raw_or_normalized_text_published": False,
        "source_objects_mutated": False,
    }
    assert payload["candidate_selection"] == {
        "d6_forensic_root_candidate_authorized": False,
        "d6_rerun_authorized": False,
        "d7_candidate_authorized": False,
        "identity_conditioned_exceptions_authorized": False,
        "next_step": (
            "OUTCOME_BLIND_D7_GRAMMAR_MECHANISM_SYNTHETIC_PROBE_"
            "BEFORE_PREREGISTRATION"
        ),
    }
    assert payload["forensic_source"] == {
        "census_replica_root_name": "bitcoin-b.git",
        "chain_hash": census.BITCOIN_CHAIN_HASH,
        "commit_rows": census.BITCOIN_CHAIN_ROWS,
        "git_commands": 954,
        "group_hash": census.BITCOIN_GROUP_HASH,
        "groups": census.BITCOIN_GROUP_ROWS,
        "object_store_after_hash": census.FORENSIC_OBJECT_STORE_HASH,
        "object_store_before_hash": census.FORENSIC_OBJECT_STORE_HASH,
        "object_store_unchanged": True,
        "oid_manifest_sha256": census.OID_MANIFEST_SHA256,
        "post_terminal_quarantine": {
            "cause_attribution": "UNKNOWN_NOT_INFERRED",
            "census_use_authorized": False,
            "expected_terminal_snapshot_hash": (
                census.FORENSIC_OBJECT_STORE_HASH
            ),
            "observed_object_count": (
                census.QUARANTINED_OBJECT_COUNT
            ),
            "observed_snapshot_hash": (
                census.QUARANTINED_OBJECT_STORE_HASH
            ),
            "root_name": "bitcoin-a.git",
        },
        "requested_blobs": census.REQUESTED_BLOBS,
        "source_path_rows_opened": 545,
        "terminal_replica_receipts_equal": True,
    }


def test_census_exhausts_every_blob_and_failed_event() -> None:
    observed = census_payload()["census"]

    assert (
        observed["grammar_class_counts"]
        == census.EXPECTED_GRAMMAR_CLASS_COUNTS
    )
    assert sum(observed["grammar_class_counts"].values()) == (
        census.REQUESTED_BLOBS
    )
    assert observed["baseline_error_blob_count"] == 8
    assert (
        observed["baseline_error_counts"]
        == census.EXPECTED_BASELINE_ERROR_COUNTS
    )
    assert observed["baseline_error_roster_hash"] == (
        "c9a2c9f57b39926a37df2324970b8b4c8b543ef027f426055e1b72cbb091eeed"
    )
    assert observed["class_roster_hashes"] == {
        census.LATER_PRE_CLASS: (
            "7735f98836200a609d812a0cc27fcb4c14120267e56877654abea53ba70e8ddd"
        ),
        census.PREFIXED_DEPENDENCY_CLASS: (
            "bb4de1c9c2c5d7fd2b773c0477007d35220b01418318a3900366bf3c9e7af05e"
        ),
        census.VALID_CLASS: (
            "d0a4c83eb10f5ecfbaf9b3db79a92905f8eee33bee6ec3e1904a054827975a88"
        ),
    }
    assert observed["error_event_count"] == 7
    assert observed["error_event_profile_counts"] == [
        {
            "count": 1,
            "event_type": "CREATE",
            "new_grammar_class": census.PREFIXED_DEPENDENCY_CLASS,
            "old_grammar_class": None,
        },
        {
            "count": 6,
            "event_type": "UPDATE",
            "new_grammar_class": census.LATER_PRE_CLASS,
            "old_grammar_class": census.LATER_PRE_CLASS,
        },
    ]
    assert observed["error_event_roster_hash"] == (
        "bf14e2a091cabc9ceb6ccd5f8c7b14c263929e803c0eff4bbb2ac191ed03ca29"
    )
    assert observed["error_proposal_count"] == 2
    assert observed["error_proposal_roster_hash"] == (
        "529bb6a22fbd3dacef6a64e1696a07a6c74c96ef07f963addc2c47bff1ac61f4"
    )
    assert observed["terminal_error_event_set_matches"] is True
    assert observed["terminal_semantic_error_roster_hash"] == (
        census.TERMINAL_SEMANTIC_ERROR_ROSTER_HASH
    )
    assert observed["unique_blob_contexts"] == census.REQUESTED_BLOBS
    assert observed["unknown_grammar_count"] == 0


def test_census_measurements_define_categories_without_identity_lists() -> None:
    observed = census_payload()["census"]

    assert observed["grammar_measurements"] == {
        census.LATER_PRE_CLASS: {
            "all_path_numbers_match": True,
            "measurement_roster_hash": (
                "87ceaaa6632069427ab50a4a48a3631f73b3bc224c3ffb51be00a7c91b909172"
            ),
            "numeric_ranges": {
                "candidate_dependency_edge_count": {
                    "max": 0,
                    "min": 0,
                    "sum": 0,
                },
                "candidate_header_field_count": {
                    "max": 10,
                    "min": 10,
                    "sum": 70,
                },
                "exact_later_pre_candidate_count": {
                    "max": 1,
                    "min": 1,
                    "sum": 7,
                },
                "opening_line_index": {
                    "max": 8,
                    "min": 7,
                    "sum": 54,
                },
                "prefix_bytes": {
                    "max": 548,
                    "min": 467,
                    "sum": 3_674,
                },
                "prefix_nonblank_lines": {
                    "max": 7,
                    "min": 6,
                    "sum": 47,
                },
            },
        },
        census.PREFIXED_DEPENDENCY_CLASS: {
            "all_path_numbers_match": True,
            "measurement_roster_hash": (
                "8fbec10d3d14a601fa3773d3d774e7c002e5c6bff7f152ec3a2ce79d4824bee8"
            ),
            "numeric_ranges": {
                "bare_decimal_token_count": {
                    "max": 0,
                    "min": 0,
                    "sum": 0,
                },
                "dependency_field_count": {
                    "max": 1,
                    "min": 1,
                    "sum": 1,
                },
                "normalized_dependency_count": {
                    "max": 1,
                    "min": 1,
                    "sum": 1,
                },
                "prefixed_decimal_token_count": {
                    "max": 1,
                    "min": 1,
                    "sum": 1,
                },
            },
        },
    }
    forbidden_identity_keys = {
        "blob_oid",
        "commit_oid",
        "event_id",
        "proposal_number",
        "raw_text",
        "normalized_text",
    }
    assert nested_keys(observed).isdisjoint(forbidden_identity_keys)
    encoded = json.dumps(observed, sort_keys=True)
    assert re.search(r"BIP-[0-9]+", encoded) is None


def test_normal_bip_remains_valid() -> None:
    classification, detail, error = classify(raw_bip(900))

    assert classification == census.VALID_CLASS
    assert detail == {}
    assert error is None


def test_exact_later_pre_header_is_category_based_and_path_bound() -> None:
    raw = raw_bip(
        900,
        prefix="Arbitrary historical preface\nSecond line\n",
    )
    classification, detail, error = classify(raw)

    assert classification == census.LATER_PRE_CLASS
    assert detail["exact_later_pre_candidate_count"] == 1
    assert detail["candidate_path_number_matches"] is True
    assert detail["prefix_nonblank_lines"] == 2
    assert error == "ValueError: PSIM malformed header line"

    mismatch, _, _ = classify(raw, path_proposal=901)
    assert mismatch == census.UNKNOWN_CLASS


def test_ambiguous_later_pre_headers_fail_closed() -> None:
    raw = raw_bip(
        900,
        prefix="Arbitrary historical preface\n",
        duplicate_header=True,
    )

    assert classify(raw)[0] == census.UNKNOWN_CLASS


def test_exact_uppercase_prefixed_dependency_is_category_based() -> None:
    synthetic_dependency = 899
    classification, detail, error = classify(
        raw_bip(900, requires=f"BIP-{synthetic_dependency}"),
    )

    assert classification == census.PREFIXED_DEPENDENCY_CLASS
    assert detail == {
        "bare_decimal_token_count": 0,
        "candidate_path_number_matches": True,
        "dependency_field_count": 1,
        "dependency_fields": ["requires"],
        "normalized_dependency_count": 1,
        "prefixed_decimal_token_count": 1,
    }
    assert error == (
        "ValueError: PSIM proposal number is not ASCII decimal"
    )
    assert classify(
        raw_bip(900, requires=f"bip-{synthetic_dependency}")
    )[0] == (
        census.UNKNOWN_CLASS
    )
    assert classify(
        raw_bip(900, requires=f"BIP -{synthetic_dependency}")
    )[0] == (
        census.UNKNOWN_CLASS
    )


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["fetch", "origin"],
        ["-C"],
        ["-c", "protocol.file.allow=always", "cat-file", "--batch"],
        ["status"],
        [
            "-C",
            str(census.QUARANTINED_BITCOIN_ROOT),
            "cat-file",
            "--batch",
        ],
        ["-C", "/tmp", "cat-file", "--batch"],
        [
            "-C",
            str(census.FORENSIC_BITCOIN_ROOT),
            "symbolic-ref",
            "HEAD",
            "refs/heads/mutated",
        ],
        [
            "verify-pack",
            "-v",
            str(
                census.QUARANTINED_BITCOIN_ROOT
                / "objects"
                / "pack"
                / "mutated.pack"
            ),
        ],
    ],
)
def test_forensic_git_guard_rejects_nonlocal_or_ambiguous_commands(
    arguments: list[str],
) -> None:
    with pytest.raises(RuntimeError):
        census._assert_local_git_arguments(arguments)


def test_forensic_git_guard_accepts_only_frozen_local_readers() -> None:
    root = str(census.FORENSIC_BITCOIN_ROOT)
    oid = "0" * 40
    safe_argv = [
        ["-C", root, "cat-file", "--batch"],
        [
            "-C",
            root,
            "cat-file",
            "--batch-all-objects",
            "--batch-check=%(objectname) %(objecttype)",
        ],
        [
            "-C",
            root,
            "diff-tree",
            "--root",
            "--no-commit-id",
            "-r",
            "--raw",
            "-z",
            "--no-renames",
            oid,
        ],
        [
            "-C",
            root,
            "diff-tree",
            "--no-commit-id",
            "-r",
            "--raw",
            "-z",
            "--no-renames",
            oid,
            oid,
        ],
        [
            "-C",
            root,
            "for-each-ref",
            "--format=%(refname) %(objectname)",
        ],
        ["-C", root, "ls-tree", "-r", "-z", "--name-only", oid],
        [
            "-C",
            root,
            "rev-list",
            "--first-parent",
            "--reverse",
            census.d6.SEALED_REF,
        ],
        ["-C", root, "rev-parse", "HEAD^{commit}"],
        [
            "-C",
            root,
            "rev-parse",
            f"{census.d6.SEALED_REF}^{{commit}}",
        ],
        ["-C", root, "symbolic-ref", "HEAD"],
        [
            "verify-pack",
            "-v",
            str(
                census.FORENSIC_BITCOIN_ROOT
                / "objects"
                / "pack"
                / "synthetic.pack"
            ),
        ],
    ]
    for arguments in safe_argv:
        census._assert_local_git_arguments(arguments)


@pytest.mark.parametrize(
    "path",
    [
        Path("/tmp/census.json"),
        Path("results/nested/census.json"),
        Path("results/census.txt"),
        Path("census.json"),
    ],
)
def test_output_guard_rejects_nonflat_or_nonjson_paths(path: Path) -> None:
    with pytest.raises(RuntimeError):
        census._safe_output_path(path)
