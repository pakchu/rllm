import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_cross_alt_execution_swarm_exhaustion_reversal as p
P=Path("results/high_volatility_cross_alt_execution_swarm_exhaustion_reversal_support_2026-08-13.json");EXPECTED="75fbb10626b22eb30dd756878c7470a64ab2bac7673b7559f873e483cd18796e"
def test_source_pass_is_immutable_blind_and_authorizes_only_novelty():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;x=json.loads(P.read_text());h=x.pop("manifest_hash");assert p.canonical_hash(x)==h;assert x["policy_id"]=="HVCAESR-8" and x["support_passed"] and x["decision"]=="pass_to_novelty";assert x["advance_to_gross9_novelty"] and not x["advance_to_economic_outcomes"];assert not x["postentry_return_pnl_execution_price_opened"] and not x["funding_values_opened"] and not x["gross9_rows_opened"];assert {k:v["events"] for k,v in x["support"].items()}=={"train":107,"test":214,"eval":210,"final":98};assert all(x["support_checks"].values())
