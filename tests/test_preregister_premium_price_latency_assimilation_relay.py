from training import preregister_premium_price_latency_assimilation_relay as prereg
def test_manifest_boundary():
 p=prereg.build();assert p["manifest_hash"]==prereg.canonical_hash({k:v for k,v in p.items() if k!="manifest_hash"});assert not p["outcomes_opened"] and not p["source_incidence_opened"] and not p["gross9_rows_opened"]
def test_formula_clock_and_gates():
 p=prereg.build();f=p["features"];assert "p_E*p_L>0" in f["eligible_state"] and "0.5*prior_median" in f["eligible_state"];assert p["clock"]["hold"]=="8 elapsed hours";assert p["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8}
def test_rv_later_and_controls_frozen():
 p=prereg.build();assert p["rv20_stress_slice"]["entry_filter"] is False;assert p["post_stage_volatility_audit"]["minimum_q90_trades"]==8;assert set(p["diagnostic_controls"]["definitions"])=={"premium_persistence_without_underresponse","late_btc_return_only","contemporaneous_agreement","one_hour_stale_premium","direction_flip"}
