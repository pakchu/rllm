import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_spot_perpetual_sign_transition_leadership_relay as p
P=Path("results/high_volatility_spot_perpetual_sign_transition_leadership_relay_support_2026-08-13.json");EXPECTED="fd284ea375c3ee742c0a240a41bc3eeb2ffa1b8489a7fa2e8c7784537f3ec153"
def test_terminal_blind_source_rejection_is_immutable():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;x=json.loads(P.read_text());h=x.pop("manifest_hash");assert p.canonical_hash(x)==h;assert not x["support_passed"] and x["decision"]=="terminal_source_support_reject";assert not x["advance_to_gross9_novelty"] and not x["advance_to_economic_outcomes"];assert not x["postentry_return_pnl_execution_price_opened"] and not x["funding_values_opened"] and not x["gross9_rows_opened"];assert {k:v["events"] for k,v in x["support"].items()}=={"train":0,"test":6,"eval":1,"final":5}
