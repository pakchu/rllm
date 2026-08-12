import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_cross_venue_cash_shock_sponsorship_relay as p
P=Path("results/high_volatility_cross_venue_cash_shock_sponsorship_relay_support_2026-08-13.json");EXPECTED="76c918e40752859e4eb1e1ea18aa034bc05ecf89dd58c2e4417c3f3899bdd1e0"
def test_source_rejection_is_immutable_blind_and_terminal():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;x=json.loads(P.read_text());h=x.pop("manifest_hash");assert p.canonical_hash(x)==h;assert x["policy_id"]=="HVCVCSS-8" and not x["support_passed"] and x["decision"]=="terminal_source_support_reject";assert not x["advance_to_gross9_novelty"] and not x["advance_to_economic_outcomes"];assert not x["postentry_return_pnl_execution_price_opened"] and not x["funding_values_opened"] and not x["gross9_rows_opened"];assert {k:v["events"] for k,v in x["support"].items()}=={"train":0,"test":0,"eval":0,"final":6}
