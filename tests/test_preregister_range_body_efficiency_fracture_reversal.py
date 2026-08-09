from training import preregister_range_body_efficiency_fracture_reversal as prereg
def test_manifest_boundary():
 p=prereg.build();assert p["manifest_hash"]==prereg.canonical_hash({k:v for k,v in p.items() if k!="manifest_hash"});assert not p["outcomes_opened"] and not p["source_incidence_opened"] and not p["gross9_rows_opened"]
def test_feature_clock_gates():
 p=prereg.build();assert "96" in p["features"]["window"] and "24" in p["features"]["impulse"];assert p["clock"]["hold"]=="8 elapsed hours";assert p["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8}
def test_rv_later_and_controls_frozen():
 p=prereg.build();assert p["rv20_stress_slice"]["entry_filter"] is False;assert p["post_stage_volatility_audit"]["minimum_q90_trades"]==8;assert set(p["diagnostic_controls"]["definitions"])=={"raw_range_variance_tail","inverse_body_variance_tail","one_decision_stale_fracture","direction_flip"}
