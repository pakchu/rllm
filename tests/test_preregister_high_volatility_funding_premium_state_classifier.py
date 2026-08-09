from training import preregister_high_volatility_funding_premium_state_classifier as s
def test_hvfpsc_blind_singleton():
 p=s.build();assert p["policy_id"]=="HVFPSC-8" and p["singleton"] is True;assert p["oos_outcomes_opened"] is False and p["oos_source_incidence_opened"] is False and p["research_boundary"]["repair_of_prior_candidate"] is False
def test_hvfpsc_contract():
 p=s.build();assert p["feature_contract"]["ordered_features"]==list(s.FEATURES) and len(s.FEATURES)==13;assert p["policy"]["score_quantile"]==.6 and p["training_contract"]["hyperparameter_grid"] is False;assert "D+5m" in p["training_contract"]["decisions"]
def test_hvfpsc_hash():
 p=s.build();assert p["manifest_hash"]==s.canonical_hash({k:v for k,v in p.items() if k!="manifest_hash"})
