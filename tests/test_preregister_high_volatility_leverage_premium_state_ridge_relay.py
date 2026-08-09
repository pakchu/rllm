from training import preregister_high_volatility_leverage_premium_state_ridge_relay as s
def test_hvlpsr_is_oos_blind_singleton():
 p=s.build();assert p["policy_id"]=="HVLPSR-8" and p["singleton"] is True;assert p["oos_outcomes_opened"] is False and p["oos_source_incidence_opened"] is False;assert p["research_boundary"]["prior_oi_economic_outcomes_opened"] is False and p["research_boundary"]["repair_of_prior_candidate"] is False
def test_hvlpsr_frozen_contract():
 p=s.build();assert p["feature_contract"]["ordered_features"]==list(s.FEATURES) and len(s.FEATURES)==15;assert p["training_contract"]["hyperparameter_grid"] is False;assert p["policy"]["hold_hours"]==8 and p["diagnostic_controls"]["names"][-1]=="forced_long"
def test_hvlpsr_hash():
 p=s.build();assert p["manifest_hash"]==s.canonical_hash({k:v for k,v in p.items() if k!="manifest_hash"})
