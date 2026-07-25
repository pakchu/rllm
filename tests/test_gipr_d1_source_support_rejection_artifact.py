from __future__ import annotations

import hashlib
import json
import os

from training import (
    build_governance_intent_payload_relation_source_support as gipr,
)


REJECTION_SHA256 = (
    "6ec88e943114b7b067abc8135f663bdf8e367ae71886974035ba89caddd0d5db"
)
RESULT_HASH = (
    "30c1b746de020805ff92c93216154ae48f3edb7a686f6840c3382d516345348b"
)


def rejection_bytes() -> bytes:
    return (gipr.REPO_ROOT / gipr.REJECTION_REPORT).read_bytes()


def rejection() -> dict:
    return json.loads(rejection_bytes())


def test_rejection_is_canonical_gate_two_terminal_evidence() -> None:
    raw = rejection_bytes()
    payload = rejection()
    assert hashlib.sha256(raw).hexdigest() == REJECTION_SHA256
    assert raw == gipr.canonical_json_bytes(payload)
    core = {
        key: value for key, value in payload.items() if key != "result_hash"
    }
    assert payload["result_hash"] == RESULT_HASH == gipr.canonical_hash(core)
    assert payload["decision"] == "reject"
    assert payload["terminal_action"] == gipr.FAILURE_ACTION
    assert payload["first_failure"] == {
        "gate_id": 2,
        "name": "dual_transport_canonical_replay",
    }
    assert [gate["passed"] for gate in payload["gates"]] == [True, False]
    assert payload["gates"][1]["metrics"] == {
        "gate_evaluation_completed": False,
        "error_type": "HTTPError",
    }
    assert payload["error"] == {"type": "HTTPError"}


def test_rejection_opened_source_only_and_no_outcomes() -> None:
    payload = rejection()
    assert payload["source_incidence_opened"] is True
    assert payload["source_audit"]["event_logs_opened"] is True
    assert payload["source_audit"]["canonical_log_rows"] == 0
    assert payload["counts"] == {
        "events": 0,
        "proposals": 0,
        "daily_cards": 0,
    }
    assert payload["profitability_result"] is False
    assert payload["outcomes_opened"] is False
    assert payload["artifacts"] is None
    assert payload["forbidden_access"] == gipr.AccessLedger.zero().snapshot()


def test_rejection_has_no_pass_artifact_siblings() -> None:
    for path in (
        gipr.EVENT_OUTPUT,
        gipr.PROPOSAL_OUTPUT,
        gipr.CARD_OUTPUT,
        gipr.CONTROL_OUTPUT,
        gipr.PASS_REPORT,
    ):
        assert not os.path.lexists(gipr.REPO_ROOT / path)


def test_terminal_state_is_idempotent_without_rpc_configuration() -> None:
    payload = gipr.terminal_state()
    assert payload is not None
    assert payload["result_hash"] == RESULT_HASH
    repeated = gipr.run_official(gipr.Config())
    assert repeated == payload
