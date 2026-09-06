from training import preregister_cftc_enforcement_volatility_momentum_relay as prereg
def test_hash_blind_boundary():
 p=prereg.build();core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==prereg.canonical_hash(core);assert p["outcomes_opened"] is False;assert p["research_boundary"]["cftc_action_rows_or_incidence_opened"] is False;assert p["gross9_rows_opened"] is False
def test_fixed_contract():
 p=prereg.build();assert p["clock"]["hold"]=="24 elapsed hours";assert p["features"]["variation_rank"].endswith(">=0.65");assert p["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8}
