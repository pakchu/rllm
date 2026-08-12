import json

from training import preregister_high_volatility_cross_alt_illiquidity_impulse_consensus_relay as p


def test_registration_is_canonical_blind_and_frozen():
    value = p.build()
    p.validate(value)
    core = {k: v for k, v in value.items() if k != "manifest_hash"}
    assert p.canonical_hash(core) == value["manifest_hash"]
    assert value["policy_id"] == "HVCIIC-8"
    assert not value["outcomes_opened"]
    assert not value["source_incidence_opened"]
    assert not value["gross9_rows_opened"]
    assert value["policy"]["minimum_consensus_breadth"] == 4
    assert value["policy"]["impulse_rank_min"] == 0.8
    assert value["policy"]["variation_rank_min"] == 0.65
    assert value["stopping_rule"].startswith("terminal first failure")
    json.dumps(value, allow_nan=False)
