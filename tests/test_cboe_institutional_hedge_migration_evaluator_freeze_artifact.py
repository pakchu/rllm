from __future__ import annotations

from training import evaluate_cboe_institutional_hedge_migration as evaluator


def test_committed_cihm_evaluator_freeze_is_sealed_and_replays() -> None:
    report = evaluator.verify_evaluator_freeze()
    assert report["policy_id"] == "CIHM-1"
    assert report["manifest_hash"] == (
        "adeebd3c552789dec754e1fdd0f6e697c9fe1d0f0e83265e360422f3b7197112"
    )
    assert report["evaluator_source_sha256"] == (
        "b02b68acf1f2a57e9a55a57e76380e3984c68d49f1b872de7e3608058235e9e5"
    )
    assert report["opened_windows"] == []
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["simulation_run_during_freeze"] is False
