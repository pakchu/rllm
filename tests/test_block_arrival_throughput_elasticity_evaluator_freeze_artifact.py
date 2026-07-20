from __future__ import annotations

import json

from training import evaluate_block_arrival_throughput_elasticity_pre2024 as evaluate
from training import freeze_block_arrival_throughput_elasticity_evaluator as freeze


FREEZE_SHA256 = "9473f5ba1b821e1692b608dfa54c47fa8d666bb188e58f72d03dd08355e03a3c"
FREEZE_MANIFEST_HASH = (
    "23a5febaebf1c0db4d24d726ff5aad3012ddde834b7465247b76c297a24403ee"
)
CONTROL_COUNTS = {
    "primary": 971,
    "direction_flip": 971,
    "weight_only": 1014,
    "tx_only": 1002,
    "denominator_free": 666,
    "stale_24h": 970,
    "random_clock": 971,
    "one_bar_delayed_entry": 971,
}


def test_frozen_evaluator_artifact_has_zero_outcome_access() -> None:
    assert evaluate.sha256_file(evaluate.EVALUATION_FREEZE) == FREEZE_SHA256
    payload = json.loads(evaluate.EVALUATION_FREEZE.read_text())
    freeze.validate_manifest(payload)
    assert payload["manifest_hash"] == FREEZE_MANIFEST_HASH
    assert payload["outcomes_opened"] is False
    assert payload["opened_windows"] == []
    assert payload["market_value_rows_parsed_during_freeze"] == 0
    assert payload["funding_value_rows_parsed_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False
    assert payload["evaluation_source_commit"] == (
        "d837c58e78f93e6ef492610d45ffb263648f59c3"
    )
    assert payload["evaluation_source_sha256"] == (
        "da0fccbea5b9b1eba280e449444edd1a35b02cc17df35524ca846773774cbe0f"
    )
    assert payload["freeze_source_sha256"] == (
        "9ccb2ff7bc5323e313c21a374c219d6c440321489b24cb0641383e6964f16a48"
    )


def test_frozen_controls_and_timestamp_only_boundaries_replay() -> None:
    payload = evaluate.verify_evaluation_freeze()
    assert payload["control_clock_counts"] == CONTROL_COUNTS
    assert payload["control_semantics"] == evaluate.CONTROL_SEMANTICS
    assert payload["outcome_boundaries"]["market"]["value_rows_parsed"] == 0
    assert payload["outcome_boundaries"]["funding"]["value_rows_parsed"] == 0
    assert payload["outcome_boundaries"]["market"]["window_value_row_counts"] == {
        "train": 210_240,
        "selection": 105_120,
    }
    assert payload["outcome_boundaries"]["funding"]["window_value_row_counts"] == {
        "train": 2_190,
        "selection": 1_095,
    }
