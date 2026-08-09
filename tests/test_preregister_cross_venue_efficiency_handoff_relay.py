from training import preregister_cross_venue_efficiency_handoff_relay as s
def test_cveh_blind_singleton():
 p=s.build();assert p["policy_id"]=="CVEH-6" and p["singleton"] is True;assert p["oos_outcomes_opened"] is False and p["candidate_source_incidence_opened"] is False and p["research_boundary"]["repair_of_prior_candidate"] is False
def test_cveh_frozen_geometry():
 p=s.build();assert p["policy"]["spot_efficiency_rank_min"]==.8 and p["policy"]["perpetual_efficiency_rank_max"]==.5 and p["policy"]["handoff_rank_min"]==.75;assert p["clock"]["hold"]=="6 elapsed hours" and p["feature_contract"]["onset"].startswith("current")
def test_cveh_hash():
 p=s.build();assert p["manifest_hash"]==s.canonical_hash({k:v for k,v in p.items() if k!="manifest_hash"})
