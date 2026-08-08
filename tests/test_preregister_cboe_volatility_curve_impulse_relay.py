import pytest
from training import preregister_cboe_volatility_curve_impulse_relay as prereg
def test_cvcir_is_causal_outcome_blind_singleton():
 p=prereg.build();prereg.validate(p);assert p["policy_id"]=="CVCIR-24" and p["singleton"] is True;assert p["outcomes_opened"] is False and p["source_incidence_opened"] is False;assert p["clock"]["entry"].startswith("first later exact common Cboe");assert p["research_boundary"]["repair_of_cvsrc"] is False
def test_cvcir_tamper_rejected():
 p=prereg.build();p["features"]["vix_rank_min"]=.5
 with pytest.raises(RuntimeError,match="canonical hash mismatch"):prereg.validate(p)
