from __future__ import annotations

from training import evaluate_cboe_tail_hedge_disagreement as evaluator


def test_committed_cthd_evaluator_freeze_is_sealed_and_replays() -> None:
    report = evaluator.verify_evaluator_freeze()
    assert report["policy_id"] == "CTHD-1"
    assert report["manifest_hash"] == (
        "7d8e08053bfebe85dfb973818f810427fde80e1025c10eb6c6e464b126866018"
    )
    assert report["evaluator_source_sha256"] == (
        "7bdb67fc82b46cfbcca8bdd076b196cf84a9bca9662dd12223b8508939ec6fd5"
    )
    assert report["opened_windows"] == []
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["simulation_run_during_freeze"] is False
