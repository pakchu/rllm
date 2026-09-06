import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_turning_point_deficiency_continuation_relay_train_economics_2026-08-12.json"
)


def test_hvtpdcr_train_rejection_is_terminal_and_later_stages_are_sealed():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "HVTPDCR-8"
    assert report["stage"] == "train"
    assert report["passed"] is False
    assert report["decision"] == "terminal_reject_no_repair"
    assert report["later_stage_outcomes_opened"] is False
    assert report["advance_to_next_stage"] is False
    assert report["advance_to_post_stage_volatility_audit"] is False
    assert report["checks"]["cagr_to_strict_mdd_min_3"] is False
    assert report["checks"]["stress_cagr_to_strict_mdd_min_2_5"] is False
    assert report["primary"]["base"]["absolute_return_pct"] > 0
    assert report["primary"]["base"]["cagr_to_strict_mdd"] < 3
    assert report["primary"]["stress"]["cagr_to_strict_mdd"] < 2.5
    assert report["physical_rows_opened"]["primary_clock"] == 26
