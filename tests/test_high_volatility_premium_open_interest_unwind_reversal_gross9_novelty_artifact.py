import json
from pathlib import Path


def test_novelty_artifact_passes_all_sleeves():
    value = json.loads(Path("results/high_volatility_premium_open_interest_unwind_reversal_gross9_novelty_2026-08-11.json").read_text())
    assert value["policy_id"] == "HVPOIUR-8"
    assert value["every_gross9_sleeve_passed"] is True
    assert value["advance_to_economic_outcomes"] is True
    assert value["evidence_boundary"]["candidate_clock_rows_opened"] == 277
    assert value["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    for sleeve in value["gross9_sleeves"].values():
        assert sleeve["passed"] is True
        assert all(sleeve["checks"].values())
