from __future__ import annotations

import hashlib
import json
import subprocess

from training import preregister_cboe_edge_flip_sequence_policy_d2 as d2


ARTIFACT_SHA256 = (
    "9f5afdb4647de01e7f5c5130fba4b68cf0c10824f90f33802b57da33d314de2a"
)
MANIFEST_HASH = (
    "fa1da6ebd2ecc674d64aa95ca0860434db071a62ffbe3442218c73699d312e00"
)
SCIENTIFIC_MECHANISM_HASH = (
    "63d23fbbf5013a331147d50b918109d79c09c23fa7aac99140aa3fe02641e6a4"
)
D1_SCIENTIFIC_CONTRACT_HASH = (
    "a32cbe62065ea3a756d495cb42a6f335746696d66e731841a56dac1106c96be2"
)
PRODUCER_COMMIT = "a7ac3239122adf06ded0f374c582cbca9df253da"
PRODUCER_SHA256 = (
    "ab0021a327c351e030098c388ae6a8d35016edcfc50cba42177c3206b4c6a9e4"
)


def artifact() -> dict:
    return json.loads(
        (d2.REPOSITORY_ROOT / d2.DEFAULT_OUTPUT).read_text()
    )


def test_preregistration_artifact_is_exact_and_self_consistent() -> None:
    payload = artifact()
    assert d2.sha256_file(d2.DEFAULT_OUTPUT) == ARTIFACT_SHA256
    assert payload["manifest_hash"] == MANIFEST_HASH
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == d2.canonical_hash(core)
    assert payload["scientific_mechanism_hash"] == SCIENTIFIC_MECHANISM_HASH
    assert payload["scientific_mechanism"][
        "d1_scientific_contract_hash"
    ] == D1_SCIENTIFIC_CONTRACT_HASH
    d2.validate_manifest(payload)


def test_artifact_binds_exact_committed_producer_and_absolute_git() -> None:
    payload = artifact()
    assert payload["authority"]["producer"] == {
        "path": d2.PRODUCER_SCRIPT,
        "commit": PRODUCER_COMMIT,
        "sha256": PRODUCER_SHA256,
    }
    sealed = subprocess.run(
        [
            d2.GIT_EXECUTABLE,
            "show",
            f"{PRODUCER_COMMIT}:{d2.PRODUCER_SCRIPT}",
        ],
        cwd=d2.REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(sealed).hexdigest() == PRODUCER_SHA256
    assert d2.sha256_file(d2.PRODUCER_SCRIPT) == PRODUCER_SHA256
    assert payload["authority"]["runtime"] == {
        "path": "/usr/bin/git",
        "path_component": "/usr/bin",
        "sha256": d2.GIT_EXECUTABLE_SHA256,
        "version": "git version 2.43.0",
    }


def test_artifact_inherits_full_d1_scientific_contract() -> None:
    payload = artifact()
    contract = payload["scientific_mechanism"]["d1_scientific_contract"]
    assert set(contract) == set(d2.D1_SCIENTIFIC_KEYS)
    assert contract == d2.d1_scientific_contract()
    assert contract["sequence_language"]["action_space"] == [
        "TARGET_LONG",
        "TARGET_FLAT",
        "TARGET_SHORT",
    ]
    assert contract["clock"]["scheduled_exit"] == (
        "entry plus 288 five-minute bars"
    )
    assert contract["support_gates"]["minimum_total_intervals"] == 920


def test_artifact_opened_no_source_value_or_outcome() -> None:
    payload = artifact()
    assert payload["source_rows_parsed"] == 0
    assert payload["source_values_opened"] is False
    assert payload["outcomes_opened"] is False
    assert all(value == 0 for value in payload["forbidden_access"].values())
    assert payload["terminal_actions"] == d2.TERMINAL_ACTIONS
    assert d2.write_once(d2.DEFAULT_OUTPUT, payload) == ARTIFACT_SHA256
