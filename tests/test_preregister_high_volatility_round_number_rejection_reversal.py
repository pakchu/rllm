import json
from training import preregister_high_volatility_round_number_rejection_reversal as prereg

def test_manifest_is_canonical_singleton():
    payload = prereg.build(); prereg.validate(payload)
    assert payload["policy_id"] == "HVRNRR-6" and payload["singleton"] is True
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["grid"] is False

def test_round_lattice_clock_and_gates_are_frozen():
    payload = prereg.build(); policy = payload["policy"]
    assert policy["round_quantum_usd"] == 1000.0
    assert (policy["history_hours"], policy["minimum_history_hours"]) == (2160, 1440)
    assert (policy["penetration_rank_min"], policy["variation_rank_min"]) == (0.75, 0.65)
    assert payload["clock"]["entry"] == "exact BTCUSDT D+5m open"
    assert payload["clock"]["hold"] == "6 elapsed hours"
    assert payload["novelty_gates"]["must_pass_before_economics"] is True

def test_outcomes_and_later_sources_are_sealed(tmp_path):
    payload = prereg.build()
    assert payload["outcomes_opened"] is False and payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["source_plan"]["funding_rate_values"] == "sealed during source support"
    path = tmp_path / "x.json"; path.write_text(json.dumps(payload, allow_nan=False))
    assert json.loads(path.read_text()) == payload
