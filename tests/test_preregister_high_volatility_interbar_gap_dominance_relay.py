import json
from training import preregister_high_volatility_interbar_gap_dominance_relay as prereg

def test_manifest_is_canonical_singleton():
    payload = prereg.build(); prereg.validate(payload)
    assert payload["policy_id"] == "HVIGDR-6"
    assert payload["singleton"] is True
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["grid"] is False

def test_gap_policy_clock_and_gates_are_frozen():
    payload = prereg.build(); policy = payload["policy"]
    assert policy["block_minutes"] == 360 and policy["gap_count"] == 359
    assert policy["history_hours"] == 2160 and policy["minimum_history_hours"] == 1440
    assert (policy["gap_dominance_rank_min"], policy["block_return_rank_min"], policy["variation_rank_min"]) == (0.80, 0.60, 0.65)
    assert payload["clock"]["entry"] == "exact BTCUSDT D+5m open"
    assert payload["clock"]["hold"] == "6 elapsed hours"
    assert payload["novelty_gates"]["must_pass_before_economics"] is True

def test_outcomes_and_later_sources_are_sealed(tmp_path):
    payload = prereg.build()
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["source_plan"]["funding_rate_values"] == "sealed during source support"
    path = tmp_path / "x.json"; path.write_text(json.dumps(payload, allow_nan=False))
    assert json.loads(path.read_text()) == payload
