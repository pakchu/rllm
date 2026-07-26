from __future__ import annotations

import hashlib
import json

from training import (
    build_protocol_specification_intent_maturity_d7_source_support as runner,
)


REJECTION_SHA256 = (
    "36702b4737f1bb37e901241a96e04f30e77132bb6a18ade1fab277a83f15557e"
)
RESULT_HASH = (
    "45846070617398860a03f5a401047c95a37c7ba3526c37fbcea5a11687e8658b"
)


def rejection_bytes() -> bytes:
    return (runner.REPO_ROOT / runner.DEFAULT_REJECTION_PATH).read_bytes()


def rejection() -> dict:
    return json.loads(rejection_bytes())


def test_terminal_rejection_is_canonical_and_hash_bound() -> None:
    raw = rejection_bytes()
    payload = rejection()
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }

    assert hashlib.sha256(raw).hexdigest() == REJECTION_SHA256
    assert raw == runner.canonical_json_bytes(payload)
    assert payload["result_hash"] == RESULT_HASH
    assert payload["result_hash"] == runner.canonical_hash(core)
    assert payload["decision"] == "reject"
    assert payload["terminal_action"] == runner.FAILURE_ACTION
    assert payload["profitability_result"] is False
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is True
    assert payload["artifacts"] is None


def test_first_four_gates_passed_and_gate_five_failed() -> None:
    payload = rejection()
    assert payload["first_failure"] == {
        "gate_id": 5,
        "name": "split_annual_quarterly_unique_day_support",
    }
    assert [row["name"] for row in payload["gates"]] == list(
        runner.GATE_NAMES[:5]
    )
    assert [row["passed"] for row in payload["gates"]] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert payload["gates"][4] == {
        "name": "split_annual_quarterly_unique_day_support",
        "passed": False,
        "metrics": {
            "gate_evaluation_completed": False,
            "error_type": "ValueError",
        },
        "failure": (
            "split_annual_quarterly_unique_day_support raised ValueError"
        ),
    }
    assert payload["error"] == {"type": "ValueError"}


def test_gate_four_materialized_the_complete_source_roster() -> None:
    payload = rejection()
    gate = payload["gates"][3]

    assert gate["metrics"]["attempted_blob_total"] == 11_280
    assert gate["metrics"]["requested_blob_total"] == 11_280
    assert gate["metrics"]["full_roster_collected_before_decision"] is True
    assert gate["metrics"]["receipt_roster_exact"] is True
    assert gate["metrics"]["bitcoin"]["events"] == 371
    assert gate["metrics"]["ethereum"]["events"] == 4_985
    assert gate["metrics"]["bitcoin"]["semantic_error_count"] == 0
    assert gate["metrics"]["ethereum"]["semantic_error_count"] == 0
    assert gate["metrics"]["bitcoin"]["registered_class_roster_exact"] is True
    assert gate["metrics"]["ethereum"]["registered_class_roster_exact"] is True

    receipts = payload["source_audit"]["blob_semantics_receipts"]
    assert receipts["bitcoin:a"]["class_counts"] == {
        "D4_VALID": 426,
        "D7_BIP_LATER_HEADER": 7,
        "D7_BIP_PREFIXED_DEPENDENCY": 1,
    }
    assert receipts["bitcoin:a"]["event_count_materialized"] == 371
    assert receipts["ethereum:a"]["event_count_materialized"] == 4_985
    assert receipts["bitcoin:a"]["semantic_error_count"] == 0
    assert receipts["ethereum:a"]["semantic_error_count"] == 0


def test_rejection_preceded_cards_models_outcomes_and_economics() -> None:
    payload = rejection()
    access = payload["access_ledger"]

    assert payload["counts"] == {"events": 5_356, "daily_cards": 0}
    assert access["git_commands"] == 21_270
    assert access["network_commands"] == 16
    assert access["source_path_rows_opened"] == 16_312
    assert access["proposal_blobs_opened"] == 11_280
    assert access["proposal_text_rows_opened"] == 11_280
    assert access["daily_cards_built"] == 0
    assert not any(access[name] for name in runner.FORBIDDEN_ACCESS_FIELDS)

    audit = payload["source_audit"]
    assert audit["source_run_attempt"] == 1
    assert audit["events_sha256"] == (
        "c74cf78c3607ecfd3026d3ee4b9cddd5eef13cc2c7c4e10576e5cb84999ed5ae"
    )
    assert audit["cards_sha256"] is None
    assert audit["controls_sha256"] is None
    assert audit["repair_or_provider_swap_used"] is False


def test_terminal_authority_and_absent_pass_artifacts() -> None:
    payload = rejection()
    seal = payload["authority"]["execution_seal"]

    assert seal["shared_commit"] == (
        "0e8f22f2680a9edb2cf8497343444c16e4946df0"
    )
    assert seal["seal_hash"] == (
        "8088c0902479612bb7cc64f0c729c7375640fcb095bdd9c3d0fe62dcd35fa308"
    )
    assert seal["sha256"] == (
        "ea94ec6566b5925fb0be16bc30aae0e47f7215d42a202943e4d5213f144573d6"
    )
    assert payload["authority"]["source_authority_hash"] == (
        "98ebc81f94bb14b8dd4f8ae8b10ee9e2a514683f2aa418830fa968cd0e1e8745"
    )
    assert runner.terminal_state() == payload
    assert not any(
        (runner.REPO_ROOT / path).exists()
        for path in (
            runner.DEFAULT_RESULT_PATH,
            runner.DEFAULT_EVENTS_PATH,
            runner.DEFAULT_CARDS_PATH,
            runner.DEFAULT_CONTROLS_PATH,
            runner.RUN_LOCK_PATH,
        )
    )
