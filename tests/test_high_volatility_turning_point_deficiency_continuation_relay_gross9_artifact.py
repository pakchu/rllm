import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_turning_point_deficiency_continuation_relay_gross9_novelty_2026-08-12.json"
)


def test_hvtpdcr_gross9_artifact_authorizes_economics_without_outcomes():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "HVTPDCR-8"
    assert report["source_support_passed"] is True
    assert report["every_gross9_sleeve_passed"] is True
    assert report["gross9_novelty_status"] == "passed"
    assert report["advance_to_economic_outcomes"] is True
    boundary = report["evidence_boundary"]
    assert boundary["hvtpdcr_clock_rows_opened"] == 138
    assert boundary["btc_price_or_return_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert boundary["portfolio_return_or_pnl_metrics_computed"] is False
    for sleeve in report["gross9_sleeves"].values():
        assert sleeve["passed"] is True
        assert all(sleeve["checks"].values())
