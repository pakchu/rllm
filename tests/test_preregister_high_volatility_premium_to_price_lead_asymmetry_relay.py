import json

from training import preregister_high_volatility_premium_to_price_lead_asymmetry_relay as prereg


def test_manifest_is_canonical_and_singleton():
    payload = prereg.build()
    prereg.validate(payload)
    assert payload["policy_id"] == "HVPPLA-8"
    assert payload["singleton"] is True
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["grid"] is False


def test_frozen_cross_lag_clock_and_gates():
    payload = prereg.build()
    assert payload["policy"]["block_minutes"] == 480
    assert payload["policy"]["lead_pairs"] == 478
    assert payload["policy"]["lead_advantage_rank_min"] == 0.75
    assert payload["policy"]["premium_displacement_rank_min"] == 0.60
    assert payload["policy"]["variation_rank_min"] == 0.65
    assert payload["clock"]["entry"] == "exact BTCUSDT D+5m open"
    assert payload["clock"]["hold"] == "8 elapsed hours"
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert payload["novelty_gates"]["must_pass_before_economics"] is True


def test_outcomes_and_funding_values_are_sealed(tmp_path):
    payload = prereg.build()
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["features"]["actual_funding_value"] == "not read or used"
    assert payload["source_plan"]["funding_rate_values"] == "sealed during source support"
    output = tmp_path / "prereg.json"
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    assert json.loads(output.read_text()) == payload
