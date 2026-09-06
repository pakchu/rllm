from training import preregister_high_volatility_ordinal_pattern_classifier_relay as s
def test_hvocpr_is_blind_singleton():
 p=s.build();assert p["policy_id"]=="HVOCPR-8" and p["singleton"] is True;assert p["oos_outcomes_opened"] is False and p["oos_source_incidence_opened"] is False and p["research_boundary"]["repair_of_prior_candidate"] is False
def test_hvocpr_contract():
 p=s.build();assert p["feature_contract"]["ordered_features"]==list(s.FEATURES) and len(s.FEATURES)==15;assert p["policy"]["score_quantile"]==.5 and p["training_contract"]["hyperparameter_grid"] is False;assert p["oos_clock"]["hold"]=="8 elapsed hours"
def test_hvocpr_hash():
 p=s.build();assert p["manifest_hash"]==s.canonical_hash({k:v for k,v in p.items() if k!="manifest_hash"})
