import json

from training import preregister_high_volatility_daily_online_expert_relay as prereg


def test_manifest_and_online_contract_are_frozen():
    payload = prereg.build(); prereg.validate(payload)
    assert payload["policy_id"] == "HVDOER-12"
    assert payload["features"]["experts_in_tie_order"] == [
        "momentum_6h", "reversal_6h", "momentum_24h", "reversal_24h"
    ]
    assert payload["policy"]["expert_memory_days"] == 60
    assert payload["policy"]["minimum_mature_labels"] == 30


def test_causal_boundary_and_strict_gates_are_frozen():
    payload = prereg.build()
    assert payload["current_candidate_outcomes_opened"] is False
    assert payload["candidate_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["causal_label_authorization"]["current_candidate_postentry_return_forbidden"] is True
    encoded = json.dumps(payload)
    assert "full-calendar CAGR" in encoded
    assert "RV20 q90 only after all economics pass" in encoded
