from pathlib import Path
from training import preregister_cross_asset_volatility_breadth_relay as prereg
def test_cavbr_is_singleton_outcome_blind_and_independent():
 r=prereg.build();prereg.validate(r);assert r["policy_id"]=="CAVBR-12";assert r["singleton"] is True;assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["repair_of_prior_candidate"] is False
def test_cavbr_sources_breadth_clock_and_gates_are_frozen():
 r=prereg.build();assert r["policy"]["breadth_min"]==3;assert "VIX,VXN,GVZ,OVX" in r["features"]["common_source_dates"];assert "rank>=0.65" in r["features"]["btc_variation_rank"];assert r["clock"]["hold"]=="12 elapsed hours";assert r["economic_gates"]["stop_on_first_failure"] is True
 for x in prereg.SOURCES.values():assert Path(x["path"]).is_file()
