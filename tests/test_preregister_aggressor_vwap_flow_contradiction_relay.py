from training import preregister_aggressor_vwap_flow_contradiction_relay as prereg

def test_manifest_and_boundaries():
 p=prereg.build();core={k:v for k,v in p.items() if k!="manifest_hash"}
 assert p["manifest_hash"]==prereg.canonical_hash(core)
 assert p["outcomes_opened"] is False and p["source_incidence_opened"] is False and p["gross9_rows_opened"] is False

def test_vwap_clock_and_gates():
 p=prereg.build();assert p["features"]["buy_vwap"]=="buy_quote/buy_base";assert p["features"]["sell_vwap"]=="sell_quote/sell_base"
 assert p["clock"]["hold"]=="12 elapsed hours";assert p["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8}
 assert set(p["diagnostic_controls"]["definitions"])=={"no_flow_contradiction","flow_side","window_return_side","direction_flip"}

def test_rv20_is_later_only():
 p=prereg.build();assert p["rv20_stress_slice"]["entry_filter"] is False;assert p["post_stage_volatility_audit"]["minimum_q90_trades"]==8
