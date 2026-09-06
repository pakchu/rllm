import pytest
from training import preregister_volatility_conditioned_overnight_momentum_relay as p
def test_vomr_outcome_blind_causal_singleton():
 x=p.build();p.validate(x);assert x["policy_id"]=="VOMR-6" and x["singleton"] is True;assert x["outcomes_opened"] is False and x["source_incidence_opened"] is False;assert x["clock"]["entry"].endswith("10:05 America/New_York");assert x["research_boundary"]["promoted_cvsdr_control"] is False
def test_vomr_tamper_rejected():
 x=p.build();x["features"]["gates"]["relative_convexity_min"]=.5
 with pytest.raises(RuntimeError,match="canonical hash mismatch"):p.validate(x)
