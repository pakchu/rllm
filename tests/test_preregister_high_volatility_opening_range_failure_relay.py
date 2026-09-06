import json
from training import preregister_high_volatility_opening_range_failure_relay as prereg
def test_manifest_and_geometry_are_frozen():
 p=prereg.build();prereg.validate(p);assert p["policy_id"]=="HVORFR-6";assert p["policy"]["block_bars"]==72;assert p["policy"]["phase_bars"]==24;assert p["clock"]["hold"]=="6 elapsed hours"
def test_boundaries_and_gates_are_frozen():
 p=prereg.build();assert p["outcomes_opened"] is False;assert p["source_incidence_opened"] is False;assert p["gross9_rows_opened"] is False;e=json.dumps(p);assert "full-calendar CAGR" in e;assert "RV20 q90 only after all economics pass" in e
