from __future__ import annotations

from training import evaluate_cboe_volatility_term_rotation as evaluator


def test_committed_cvtr_evaluator_freeze_is_sealed_and_replays() -> None:
    report = evaluator.verify_evaluator_freeze()
    assert report["policy_id"] == "CVTR-1"
    assert report["manifest_hash"] == (
        "b27ff7b86817be1a2fb24497b194630fae25239d7636b5c406f4d1e1ceaa69f3"
    )
    assert report["evaluator_source_sha256"] == (
        "1bb47f6d704c2f977e44e378bf57acf4d4f6ab6455346e7b720149132f2f1f0e"
    )
    assert report["opened_windows"] == []
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["simulation_run_during_freeze"] is False
