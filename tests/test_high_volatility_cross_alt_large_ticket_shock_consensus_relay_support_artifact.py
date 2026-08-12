import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_cross_alt_large_ticket_shock_consensus_relay as p
P=Path("results/high_volatility_cross_alt_large_ticket_shock_consensus_relay_support_2026-08-13.json");EXPECTED="9a6149f81149620bcea36df383b6909d2a60d3e9eca9adf5a76990edaf83d4de"
def test_source_rejection_is_immutable_blind_and_terminal():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;x=json.loads(P.read_text());h=x.pop("manifest_hash");assert p.canonical_hash(x)==h;assert x["policy_id"]=="HVCALTS-8" and not x["support_passed"] and x["decision"]=="terminal_source_support_reject";assert not x["advance_to_gross9_novelty"] and not x["advance_to_economic_outcomes"];assert not x["postentry_return_pnl_execution_price_opened"] and not x["funding_values_opened"] and not x["gross9_rows_opened"];assert {k:v["events"] for k,v in x["support"].items()}=={"train":12,"test":48,"eval":65,"final":36};assert not x["support_checks"]["train_month_concentration"]
