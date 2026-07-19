from __future__ import annotations

from training import evaluate_bitmex_trollbox_attention_saturation as evaluator


FREEZE_SHA256 = (
    "36dde44985f26896fcc6ef861dc3a45c81479915038e5ea537cc2582f0b3b45a"
)
EVALUATOR_SHA256 = (
    "d32055317913bd80b00d0115bb0d5f26fa70b9f7d456d3718852e535a70ff193"
)


def test_evaluator_freeze_artifact_is_hash_pinned_and_unopened() -> None:
    assert evaluator._sha256(evaluator.EVALUATOR_FREEZE) == FREEZE_SHA256
    assert evaluator._sha256(evaluator.EVALUATOR_SOURCE) == EVALUATOR_SHA256

    payload = evaluator.verify_evaluator_freeze()

    assert payload["manifest_hash"] == (
        "3cde808815dc91283c65b23bb2462bf7d4cff087d6dff55c07f774dcd0707dc0"
    )
    assert payload["evaluator_source_sha256"] == EVALUATOR_SHA256
    assert payload["opened_windows"] == []
    assert payload["sealed_windows"] == ["train", "test"]
    assert payload["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_parsed_during_freeze"] == 0
    assert payload["price_conditioned_schedules_built_during_freeze"] is False
    assert payload["execution_data_bytes_hashed_during_freeze"] is False
    assert payload["simulation_run_during_freeze"] is False
    assert payload["strategy_outcomes_calculated"] is False
    assert payload["mutable_parameters"] == []
    assert payload["event_counts_before_market"] == {
        "train": {"clear_semantic_events": 1_728},
        "test": {"clear_semantic_events": 990},
    }
    assert len(payload["source_contracts"]["stage_market_months"]["train"]) == 24
    assert len(payload["source_contracts"]["stage_market_months"]["test"]) == 36
