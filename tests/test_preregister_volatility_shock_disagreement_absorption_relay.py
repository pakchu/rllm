import pytest
from training import preregister_volatility_shock_disagreement_absorption_relay as p
def test_vsdar_outcome_blind_causal_singleton():
 x=p.build();p.validate(x);assert x["policy_id"]=="VSDAR-24" and x["singleton"] is True;assert x["outcomes_opened"] is False and x["source_incidence_opened"] is False;assert x["research_boundary"]["promoted_prior_control"] is False;assert x["clock"]["entry"].startswith("first later exact common Cboe")
def test_vsdar_tamper_rejected():
 x=p.build();x["features"]["magnitude_gate"]="rank>=0.5"
 with pytest.raises(RuntimeError,match="canonical hash mismatch"):p.validate(x)
