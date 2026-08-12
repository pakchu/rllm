import hashlib
import json

from training import preregister_high_volatility_daily_flow_impact_capacity_reversal as p


def test_manifest_is_deterministic_and_outcome_blind():
    first = p.build()
    second = p.build()
    assert first == second
    p.validate(first)
    core = {key: value for key, value in first.items() if key != "manifest_hash"}
    assert first["manifest_hash"] == p.canonical_hash(core)
    assert first["outcomes_opened"] is False
    assert first["source_incidence_opened"] is False
    assert first["gross9_rows_opened"] is False


def test_policy_and_sequential_gates_are_frozen():
    value = p.build()
    assert value["policy_id"] == "HVDFICR-12"
    assert value["policy"]["impact_rank_min"] == 0.75
    assert value["policy"]["variation_rank_min"] == 0.65
    assert value["policy"]["hold_hours"] == 12
    assert value["clock"]["side"] == "negative aggregate-flow sign"
    assert value["novelty_gates"]["must_pass_before_economics"] is True
    assert value["economic_gates"]["stop_on_first_failure"] is True
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
    assert value["diagnostic_controls"]["cannot_be_promoted"] is True


def test_canonical_json_preserves_utf8_contract():
    value = p.build()
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    assert hashlib.sha256(encoded).hexdigest()
