from training import preregister_weekend_monday_liquidity_handoff_relay as prereg
def test_manifest_boundary():
 p=prereg.build();assert p["manifest_hash"]==prereg.canonical_hash({k:v for k,v in p.items() if k!="manifest_hash"});assert not p["outcomes_opened"] and not p["source_incidence_opened"] and not p["gross9_rows_opened"]
def test_windows_clock_gates():
 p=prereg.build();assert "2,880" in p["features"]["weekend_window"];assert "480" in p["features"]["monday_window"];assert p["clock"]["hold"]=="16 elapsed hours";assert p["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8}
def test_rv_later_and_controls_frozen():
 p=prereg.build();assert p["rv20_stress_slice"]["entry_filter"] is False;assert p["post_stage_volatility_audit"]["minimum_q90_trades"]==8;assert set(p["diagnostic_controls"]["definitions"])=={"no_liquidity_reentry","weekend_continuation","monday_fade","direction_flip"}
