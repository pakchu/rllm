import json
from pathlib import Path


def test_novelty_passes_before_economics():
 v=json.loads(Path("results/options_risk_peak_leverage_handoff_continuation_gross9_novelty_2026-08-11.json").read_text())
 assert v["policy_id"]=="ORPLHC-6" and v["every_gross9_sleeve_passed"] is True
 assert v["advance_to_economic_outcomes"] is True
 assert v["evidence_boundary"]["candidate_clock_rows_opened"]==141
 assert v["evidence_boundary"]["economic_outcome_rows_opened"]==0
 assert all(x["passed"] for x in v["gross9_sleeves"].values())
