from training import preregister_high_volatility_turnover_lead_response_relay as p
def test_hvtlrr_boundary_and_policy():
 r=p.build();assert r["policy_id"]=="HVTLRR-8" and r["outcomes_opened"] is False and r["source_incidence_opened"] is False and r["gross9_rows_opened"] is False and r["singleton"] is True
 assert r["research_boundary"]["candidate_count"]==1 and r["research_boundary"]["grid"] is False and r["research_boundary"]["repair_of_prior_candidate"] is False
 assert r["policy"]["response_strength_rank_min"]==.75 and r["policy"]["variation_rank_min"]==.65 and r["policy"]["hold_hours"]==8
def test_hvtlrr_hash():
 r=p.build();assert r["manifest_hash"]==p.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
