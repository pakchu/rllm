import hashlib
import json

from training import (
    evaluate_high_volatility_relative_daily_volume_continuation_relay_gross9_novelty as novelty,
)


def test_hvrdv_gross9_novelty_pass_is_reproducible_and_outcome_blind():
    result = json.loads(novelty.OUTPUT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == novelty.canonical_hash(core)
    assert result["policy_id"] == "HVRDV-8"
    assert result["every_gross9_sleeve_passed"] is True
    assert result["gross9_novelty_status"] == "passed"
    assert result["advance_to_economic_outcomes"] is True
    assert result["evidence_boundary"]["outcomes_opened"] is False
    assert result["evidence_boundary"]["btc_execution_rows_opened"] == 0
    assert result["evidence_boundary"]["funding_rows_opened"] == 0
    assert all(item["passed"] for item in result["gross9_sleeves"].values())
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == (
        "fe7f3f2ecfa920e8ed31665c84ebd0c2e73cd8c51c0dc1d3f49ed15f9d34978c"
    )
