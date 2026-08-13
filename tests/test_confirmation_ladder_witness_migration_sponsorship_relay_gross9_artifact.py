import json
from pathlib import Path


RESULT = Path(
    "results/confirmation_ladder_witness_migration_sponsorship_relay_gross9_novelty_2026-08-13.json"
)


def test_gross9_pass_and_closed_economics():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "CLWMSR-6"
    assert report["source_support_passed"] is True
    assert report["every_gross9_sleeve_passed"] is True
    assert report["gross9_novelty_status"] == "passed"
    assert report["advance_to_economic_outcomes"] is True
    assert report["evidence_boundary"]["outcomes_opened"] is False
    assert all(sleeve["passed"] for sleeve in report["gross9_sleeves"].values())
