from __future__ import annotations

import hashlib
import json

from training import (
    build_collateral_liquidity_ordering_relation_source_support as s,
)


ARTIFACT_SHA256 = (
    "2fd8bf4546ddac9a566bb6a8a7ca34c077d4335b2920a6b4e91b7490e41dc58f"
)
RESULT_HASH = (
    "eb158cd4634be0bd4452dd9e0fd4f8cffeba714bebbf35a123fec36b6cb8e8fc"
)


def rejection_bytes() -> bytes:
    return (s.REPOSITORY_ROOT / s.REJECTION_REPORT).read_bytes()


def rejection() -> dict:
    return json.loads(rejection_bytes())


def test_rejection_is_canonical_and_hash_bound() -> None:
    raw = rejection_bytes()
    payload = rejection()
    assert hashlib.sha256(raw).hexdigest() == ARTIFACT_SHA256
    assert raw == s.json_bytes(payload)
    core = {
        key: value for key, value in payload.items() if key != "result_hash"
    }
    assert payload["result_hash"] == RESULT_HASH == s.canonical_hash(core)


def test_rejection_stops_at_gate_one_without_outcomes() -> None:
    payload = rejection()
    assert payload["decision"] == "reject"
    assert payload["terminal_action"] == s.FAILURE_ACTION
    assert payload["first_failure"] == {
        "gate_id": 1,
        "name": "source_schema_chronology_reconciliation",
    }
    assert payload["error"] == {
        "type": "RuntimeError",
        "message": "CLOR-D1 decimal uses signed zero",
    }
    assert payload["gates"] == [
        s._gate_record(
            1,
            {"source_build_completed": False},
            {
                "decoded_rows": {
                    "treasury": 445,
                    "soma_operations": 1_259,
                    "soma_details": 182_616,
                    "ofr": 77_369,
                    "predecessors": 0,
                },
                "error_type": "RuntimeError",
                "error_message": "CLOR-D1 decimal uses signed zero",
            },
        )
    ]
    assert payload["source_values_opened"] is True
    assert payload["outcomes_opened"] is False
    assert payload["profitability_result"] is False
    assert payload["forbidden_access"] == s.forbidden_access()
    assert payload["artifacts"] is None


def test_rejection_is_the_only_terminal_artifact() -> None:
    assert not (s.REPOSITORY_ROOT / s.SOURCE_OUTPUT).exists()
    assert not (s.REPOSITORY_ROOT / s.CONTROL_OUTPUT).exists()
    assert not (s.REPOSITORY_ROOT / s.PASS_REPORT).exists()
    assert s.terminal_state() == rejection()
