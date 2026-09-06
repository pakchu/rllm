import pytest
from training import preregister_cboe_broad_slope_acceleration_dual_confirmation_relay as p
def test_cvdc_momentum_outcome_blind_causal_singleton():
 x=p.build();p.validate(x);assert x["policy_id"]=="CVBDMR-6" and x["singleton"] is True;assert x["outcomes_opened"] is False and x["source_incidence_opened"] is False;assert x["clock"]["entry"].endswith("10:05 America/New_York");assert x["research_boundary"]["promoted_cvfdmr_control"] is False
def test_cvdc_momentum_tamper_rejected():
 x=p.build();x["clock"]["hold"]="12 hours"
 with pytest.raises(RuntimeError,match="canonical hash mismatch"):p.validate(x)
