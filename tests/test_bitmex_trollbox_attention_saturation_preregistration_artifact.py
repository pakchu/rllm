from __future__ import annotations

from pathlib import Path

from training import evaluate_bitmex_trollbox_attention_saturation as evaluator


PREREGISTRATION_SHA256 = (
    "1db642127a01b5910267a9b986186e9fb1e7d31dccb81170df476461af669b21"
)
DOCUMENT_SHA256 = (
    "ff4a3c15128f6d3da89786b17168338751050fd52e8accf31f3fa6c7a8b5cd37"
)


def test_preregistration_artifact_is_hash_pinned_and_outcome_blind() -> None:
    assert evaluator._sha256(evaluator.PREREGISTRATION) == PREREGISTRATION_SHA256
    assert evaluator._sha256(evaluator.PREREGISTRATION_DOC) == DOCUMENT_SHA256

    payload = evaluator.verify_preregistration()

    assert payload["manifest_hash"] == (
        "0e9a7eb9cf61f23502fbe2779b4bd6e04c5f3718a4cdbcd2e3ac3fb1e698c42e"
    )
    assert payload["market_or_funding_rows_parsed"] == 0
    assert payload["strategy_outcomes_calculated"] is False
    assert payload["mutable_parameters"] == []
    assert payload["sequential_opening"] == {
        "order": ["train", "test"],
        "test_rows_must_not_be_parsed_until_train_passes": True,
        "stop_on_first_failure": True,
        "post_failure_parameter_repair": False,
    }
    assert Path(payload["semantic_input"]["clock"]) == evaluator.SEMANTIC_CLOCK
    assert payload["price_displacement"]["reference_shift_bars"] == 13
    assert payload["price_displacement"]["parameter_search"] == []
