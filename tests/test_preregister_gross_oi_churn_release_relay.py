from training import preregister_gross_oi_churn_release_relay as prereg
def test_manifest_boundary():
 p=prereg.build();assert p["manifest_hash"]==prereg.canonical_hash({k:v for k,v in p.items() if k!="manifest_hash"});assert not p["outcomes_opened"] and not p["source_incidence_opened"] and not p["gross9_rows_opened"]
def test_formula_clock_gates():
 p=prereg.build();f=p["features"];assert "73 exact" in f["oi_window"] and "1-N/G" in f["cancellation"] and "720" in f["prior_ranks"];assert p["clock"]["hold"]=="12 elapsed hours";assert p["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8}
def test_rv_later_controls():
 p=prereg.build();assert p["rv20_stress_slice"]["entry_filter"] is False;assert p["post_stage_volatility_audit"]["minimum_q90_trades"]==8;assert set(p["diagnostic_controls"]["definitions"])=={"net_oi_tail_only","gross_churn_without_cancellation","late_price_escape_only","one_decision_stale_churn","direction_flip"}
