import json
from pathlib import Path


RESULT = Path(
    "results/confirmation_ladder_witness_migration_sponsorship_relay_train_economics_2026-08-13.json"
)


def test_train_is_terminal_and_later_stages_are_closed():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "CLWMSR-6"
    assert report["stage"] == "train"
    assert report["passed"] is False
    assert report["advance_to_next_stage"] is False
    assert report["advance_to_post_stage_volatility_audit"] is False
    assert report["later_stage_outcomes_opened"] is False
    assert report["decision"] == "terminal_reject_no_repair"


def test_frozen_train_failures_are_preserved():
    report = json.loads(RESULT.read_text())
    base = report["primary"]["base"]
    stress = report["primary"]["stress"]
    assert base["trades"] == 24
    assert base["absolute_return_pct"] < 0
    assert base["mean_gross_underlying_bp"] < 20
    assert stress["absolute_return_pct"] < 0
    assert report["primary"]["cluster_signflip"]["pvalue"] > 0.1
    assert not all(report["checks"].values())
