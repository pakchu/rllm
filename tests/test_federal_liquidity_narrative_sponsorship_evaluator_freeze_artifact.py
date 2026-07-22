from __future__ import annotations

from training import evaluate_federal_liquidity_narrative_sponsorship_relay as evaluator


FREEZE_FILE_SHA256 = (
    "5109f4b91cc3b17ea3cea8ffc79864f393c36341efe30234c36ed77d1b8cbfdf"
)
FREEZE_MANIFEST_HASH = (
    "09dade9c6e5198465a8480d8559c31f703d5517d9f2b0a58a1c6a87e8c427f50"
)
EVALUATOR_SOURCE_SHA256 = (
    "fad04d652d368922bdc5a8e847453d5e22959697e973eb61093b1c30eeff15b2"
)


def test_frozen_evaluator_artifact_is_hash_valid_outcome_free_and_immutable() -> None:
    payload = evaluator.verify_evaluator_freeze()
    registration = evaluator._load_json(evaluator.PREREGISTRATION)
    assert evaluator._sha256(evaluator.EVALUATOR_FREEZE) == FREEZE_FILE_SHA256
    assert payload["manifest_hash"] == FREEZE_MANIFEST_HASH
    assert payload["evaluator_source_sha256"] == EVALUATOR_SOURCE_SHA256
    assert payload["economic_gates"] == registration["economic_gates"]
    assert payload["outcomes_opened"] is False
    assert payload["opened_windows"] == []
    assert payload["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_parsed_during_freeze"] == 0
    assert payload["simulation_run_during_freeze"] is False
    assert payload["mutable_parameters"] == []
    assert all(payload["invariant_checks"].values())


def test_frozen_primary_and_delay_schedules_keep_every_event_in_both_splits() -> None:
    payload = evaluator.verify_evaluator_freeze()
    primary = payload["schedules"]["primary"]
    delayed = payload["schedules"][evaluator.DELAY_CLOCK_NAME]
    assert (primary["events"], primary["stage1_events"], primary["stage2_events"]) == (
        89,
        67,
        22,
    )
    assert (delayed["events"], delayed["stage1_events"], delayed["stage2_events"]) == (
        89,
        67,
        22,
    )
    assert primary["schedule_hash"] == (
        "244324ab8f2e772f8c29f34593de6c83ba29ada543266b53ca64238b0afd1246"
    )
    assert delayed["schedule_hash"] == (
        "f66d0590c63d2df27f6c3567e630a14018459f3c5dd6bd747919fdd6453a0693"
    )
