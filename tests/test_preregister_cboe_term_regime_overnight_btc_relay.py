import pytest
from training import preregister_cboe_term_regime_overnight_btc_relay as p
def test_cvtbr_outcome_blind_causal_singleton():
 x=p.build();p.validate(x);assert x["policy_id"]=="CVTBR-6" and x["singleton"] is True;assert x["outcomes_opened"] is False and x["source_incidence_opened"] is False;assert x["clock"]["entry"].endswith("10:05 America/New_York");assert x["research_boundary"]["promoted_cvobr_control"] is False
def test_cvtbr_tamper_rejected():
 x=p.build();x["features"]["outer_state"]["stress_min"]=.5
 with pytest.raises(RuntimeError,match="canonical hash mismatch"):p.validate(x)
