from __future__ import annotations
import copy,pytest
from training import preregister_high_volatility_cross_structure_action_vote as hvcav
from training import preregister_high_volatility_intraday_premium_activity_polarity_router as p
def test_fixed_singleton_intraday_router():
 v=p.build();p.validate(v);assert v["policy_id"]=="HVIPRAPR-8";assert v["candidate_family"]==["HVIPRAPR-8"];assert v["policy"]["block_minutes"]==480;assert v["policy"]["lower_relative_activity_rank_max"]==.25;assert v["policy"]["upper_relative_activity_rank_min"]==.75;assert v["policy"]["hold_hours"]==8
def test_contract_gates_exact():
 v=p.build();c=hvcav.build()
 for k in ("stages","source_support_gates","gross9_novelty_gates","economic_gates"):assert v[k]==c[k]
def test_causal_and_research_boundary():
 v=p.build();assert v["features"]["causal_ranks"].startswith("strict-prior");assert "lower tail" in v["features"]["side"];assert v["source_incidence_opened"] is False and v["outcomes_opened"] is False;b=v["research_boundary"];assert b["prior_daily_HVPRAPR_gross9_failure_known"] is True;assert b["same_mechanism_as_daily_candidates"] is False;assert b["repair_of_prior_candidate"] is False
def test_manifest_drift_rejected():
 v=copy.deepcopy(p.build());v["policy"]["block_minutes"]=240
 with pytest.raises(RuntimeError,match="preregistration drift"):p.validate(v)
