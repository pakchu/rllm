from training import preregister_commodity_currency_relative_stress_relay as prereg

def test_preregistration_hash_and_boundary():
 r=prereg.build();prereg.validate(r);assert r["policy_id"]=="CCRSR-12";assert r["source_incidence_opened"] is False;assert r["gross9_rows_opened"] is False;assert r["policy"]["absolute_stress_rank_min"]==.70;assert r["clock"]["entry"]=="exact BTCUSDT 21:05 UTC open"
def test_singleton_and_controls_frozen():
 r=prereg.build();assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["diagnostic_controls"]["cannot_be_promoted"] is True
