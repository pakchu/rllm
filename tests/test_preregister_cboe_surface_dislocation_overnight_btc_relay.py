import pytest
from training import preregister_cboe_surface_dislocation_overnight_btc_relay as p
def test_cvsdr_outcome_blind_causal_singleton():
 x=p.build();p.validate(x);assert x["policy_id"]=="CVSDR-6" and x["singleton"] is True;assert x["outcomes_opened"] is False and x["source_incidence_opened"] is False;assert x["clock"]["entry"].endswith("10:05 America/New_York");assert x["research_boundary"]["promoted_cvtbr_control"] is False
def test_cvsdr_tamper_rejected():
 x=p.build();x["features"]["outer_dislocation"]["short_min"]=.5
 with pytest.raises(RuntimeError,match="canonical hash mismatch"):p.validate(x)
