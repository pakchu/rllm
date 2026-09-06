from training import preregister_high_volatility_close_vwap_reversion_relay as prereg
def test_hvcvr_is_outcome_blind_independent_singleton():
 r=prereg.build();assert r["policy_id"]=="HVCVR-2";assert r["outcomes_opened"] is False;assert r["source_incidence_opened"] is False;assert r["singleton"] is True;assert r["research_boundary"]["candidate_count"]==1;assert r["research_boundary"]["grid"] is False;assert r["research_boundary"]["fhvar_direction_fade_control_promoted"] is False
def test_hvcvr_freezes_traded_center_reversion_and_gates():
 r=prereg.build();p=r["policy"];assert p["history_observations"]==180;assert p["variation_rank_min"]==.65;assert p["absolute_normalized_displacement_min"]==.2;assert p["hold_hours"]==2;assert r["source_plan"]["btc_1m"]["columns"][-2:]==["volume","quote_asset_volume"];assert r["economic_gates"]["cagr_to_strict_mdd_min"]==3.
def test_hvcvr_hash_binds_core():
 r=prereg.build();assert r["manifest_hash"]==prereg.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
