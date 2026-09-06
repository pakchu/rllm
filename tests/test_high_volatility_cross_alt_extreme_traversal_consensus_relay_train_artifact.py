import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_cross_alt_extreme_traversal_consensus_relay_train_economics_2026-08-12.json"
)


def test_hvcatcr_train_artifact_is_terminal():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "HVCATCR-8"
    assert report["stage"] == "train"
    assert report["passed"] is False
    assert report["decision"] == "terminal_reject_no_repair"
    assert report["later_stage_outcomes_opened"] is False
    assert report["primary"]["base"]["absolute_return_pct"] < 0
    assert report["primary"]["base"]["mean_gross_underlying_bp"] < 0
    assert report["checks"]["absolute_return_positive"] is False
    assert report["checks"]["each_calendar_half_positive"] is False
