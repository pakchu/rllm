from __future__ import annotations

import hashlib
import json

from training import (
    build_protocol_specification_intent_maturity_source_support as psim,
)


REJECTION_SHA256 = (
    "9b0b2354c6edbcfe627527bf4370a4eb0c1e6c1bcb76843f843d9028b16e6494"
)
RESULT_HASH = (
    "5815f7473410c7d75aabea8b6a97cfb7f963b1c6d29f8efa22f0a0a64d33655d"
)
SEAL_HASH = (
    "c26397920fa1137845f5dea56eab72cb1a8d4ead401e7ee3e249c5c1e39aa506"
)


def rejection_bytes() -> bytes:
    return (psim.REPO_ROOT / psim.DEFAULT_REJECTION_PATH).read_bytes()


def rejection() -> dict:
    return json.loads(rejection_bytes())


def test_rejection_is_canonical_and_hash_bound() -> None:
    raw = rejection_bytes()
    payload = rejection()
    core = {
        key: value for key, value in payload.items() if key != "result_hash"
    }

    assert hashlib.sha256(raw).hexdigest() == REJECTION_SHA256
    assert raw == psim.canonical_json_bytes(payload)
    assert payload["result_hash"] == RESULT_HASH == psim.canonical_hash(core)
    assert payload["protocol_version"] == psim.RESULT_PROTOCOL
    assert payload["policy_id"] == psim.POLICY_ID


def test_rejection_is_gate_one_terminal_without_source_incidence() -> None:
    payload = rejection()

    assert payload["decision"] == "reject"
    assert payload["terminal_action"] == psim.FAILURE_ACTION
    assert payload["first_failure"] == {
        "gate_id": 1,
        "name": psim.GATE_NAMES[0],
    }
    assert len(payload["gates"]) == 1
    assert payload["gates"][0] == {
        "name": psim.GATE_NAMES[0],
        "passed": False,
        "metrics": {
            "gate_evaluation_completed": False,
            "error_type": "RuntimeError",
        },
        "failure": (
            "sealed_git_identity_and_object_integrity raised RuntimeError"
        ),
    }
    assert payload["error"] == {"type": "RuntimeError"}
    assert payload["source_incidence_opened"] is False
    assert payload["profitability_result"] is False
    assert payload["outcomes_opened"] is False
    assert payload["counts"] == {"events": 0, "daily_cards": 0}
    assert payload["artifacts"] is None


def test_rejection_access_ledger_and_source_audit_are_pre_incidence() -> None:
    payload = rejection()
    access = payload["access_ledger"]
    audit = payload["source_audit"]

    assert {key: value for key, value in access.items() if value} == {
        "git_commands": 8,
        "network_commands": 3,
    }
    assert not any(access[name] for name in psim.FORBIDDEN_ACCESS_FIELDS)
    assert audit["source_classes_opened"] == ["git_remote_identity"]
    assert audit["remote_identity_opened"] is True
    assert audit["commit_metadata_opened"] is False
    assert audit["proposal_path_incidence_opened"] is False
    assert audit["proposal_blobs_opened"] is False
    assert audit["disk_used_gib_at_start"] == 291
    assert audit["disk_below_limit_at_start"] is True
    assert audit["repair_or_provider_swap_used"] is False
    assert audit["source_run_attempt"] == 1


def test_rejection_binds_committed_execution_seal() -> None:
    execution = rejection()["authority"]["execution_seal"]

    assert execution["path"] == psim.EXECUTION_SEAL_PATH.as_posix()
    assert execution["seal_hash"] == SEAL_HASH
    assert execution["shared_commit"] == (
        "80b656994f17548a7a599a548e23e9f1cd01302d"
    )


def test_only_terminal_rejection_artifact_exists() -> None:
    assert psim.terminal_state() == rejection()
    assert (psim.REPO_ROOT / psim.DEFAULT_REJECTION_PATH).is_file()
    assert not any(
        (psim.REPO_ROOT / path).exists()
        for path in (
            psim.DEFAULT_RESULT_PATH,
            psim.DEFAULT_EVENTS_PATH,
            psim.DEFAULT_CARDS_PATH,
            psim.DEFAULT_CONTROLS_PATH,
            psim.RUN_LOCK_PATH,
        )
    )
