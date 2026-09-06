from training import preregister_bitcoin_hashrate_acceleration_retarget_relay as prereg
def test_hash_blind_boundary():
 p=prereg.build();core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==prereg.canonical_hash(core);assert p["outcomes_opened"] is False;assert p["research_boundary"]["retarget_hash_header_nbits_or_incidence_opened"] is False;assert p["gross9_rows_opened"] is False
def test_fixed_contract():
 p=prereg.build();assert p["clock"]["hold"]=="48 elapsed hours";assert p["features"]["volatility_rank"].endswith(">=0.35");assert p["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8};assert len(p["authority"]["transports"])==2
