import json
from training import preregister_high_volatility_dollar_factor_response_memory_relay as prereg
def test_manifest_and_memory_are_frozen():
 p=prereg.build();prereg.validate(p);assert p["policy_id"]=="HVDFRM-12";assert p["policy"]["response_memory"]==32;assert p["policy"]["minimum_mature_responses"]==20
def test_causal_and_economic_boundaries_are_frozen():
 p=prereg.build();assert p["current_candidate_outcomes_opened"] is False;assert p["source_incidence_opened"] is False;assert p["gross9_rows_opened"] is False;e=json.dumps(p);assert "full-calendar CAGR" in e;assert "RV20 q90 only after all economics pass" in e
