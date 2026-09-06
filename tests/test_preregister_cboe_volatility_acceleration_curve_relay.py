import pytest
from training import preregister_cboe_volatility_acceleration_curve_relay as prereg
def test_cvacr_is_causal_outcome_blind_singleton():
 p=prereg.build();prereg.validate(p);assert p["policy_id"]=="CVACR-24" and p["singleton"] is True;assert p["outcomes_opened"] is False and p["source_incidence_opened"] is False;assert p["clock"]["entry"].startswith("first later exact common Cboe");assert p["research_boundary"]["repair_of_cvcir"] is False and p["research_boundary"]["promoted_prior_control"] is False
def test_cvacr_tamper_rejected():
 p=prereg.build();p["features"]["absolute_level_or_rank_gate"]=.5
 with pytest.raises(RuntimeError,match="canonical hash mismatch"):prereg.validate(p)
