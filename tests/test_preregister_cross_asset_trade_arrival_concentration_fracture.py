from training import preregister_cross_asset_trade_arrival_concentration_fracture as prereg
def test_manifest_boundary():
 p=prereg.build();assert p["manifest_hash"]==prereg.canonical_hash({k:v for k,v in p.items() if k!="manifest_hash"});assert not p["outcomes_opened"] and not p["source_incidence_opened"] and not p["gross9_rows_opened"]
def test_formula_clock_gates():
 p=prereg.build();f=p["features"];assert len(f["symbols"])==7 and "2160" in f["tail_ranks"] and "fracture_rank>=0.90" in f["eligible_state"];assert p["clock"]["hold"]=="8 elapsed hours";assert p["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8}
def test_rv_later_controls():
 p=prereg.build();assert p["rv20_stress_slice"]["entry_filter"] is False;assert p["post_stage_volatility_audit"]["minimum_q90_trades"]==8;assert set(p["diagnostic_controls"]["definitions"])=={"btc_arrival_hhi_only","cross_asset_fracture_without_return_tail","btc_return_tail_only","one_hour_stale_fracture","direction_flip"}
