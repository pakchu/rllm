import json
from pathlib import Path


def test_novelty_failure_is_terminal_before_economics():
    value=json.loads(Path("results/high_volatility_basis_leverage_ignition_continuation_gross9_novelty_2026-08-11.json").read_text())
    assert value["policy_id"]=="HVBLIC-6"
    assert value["every_gross9_sleeve_passed"] is False
    assert value["advance_to_economic_outcomes"] is False
    assert value["evidence_boundary"]["economic_outcome_rows_opened"]==0
    failed=value["gross9_sleeves"]["markov_transition_long"]
    assert failed["metrics"]["one_to_one_6h_max_matched_share"]==0.47058823529411764
    assert failed["checks"]["one_to_one_6h_max_matched_share"] is False
