from training import preregister_high_volatility_btc_factor_residual_relay as prereg

def test_preregistration_hash_and_boundary():
 r=prereg.build();prereg.validate(r);assert r["policy_id"]=="HVBFRR-12";assert r["source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert len(r["policy"]["symbols"])==7;assert r["policy"]["minimum_fit_days"]==180
def test_singleton_and_controls_frozen():
 r=prereg.build();assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["diagnostic_controls"]["cannot_be_promoted"] is True
