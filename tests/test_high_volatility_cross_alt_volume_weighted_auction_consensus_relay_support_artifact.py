import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_cross_alt_volume_weighted_auction_consensus_relay as p
P=Path("results/high_volatility_cross_alt_volume_weighted_auction_consensus_relay_support_2026-08-13.json");EXPECTED="81a1631a9c99c8f5c969bc1030349d1535de22a223b5fcf30cc82b2544cfa542"
def test_source_pass_is_immutable_blind_and_authorizes_only_novelty():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;x=json.loads(P.read_text());h=x.pop("manifest_hash");assert p.canonical_hash(x)==h;assert x["policy_id"]=="HVCAVAC-8" and x["support_passed"] and x["decision"]=="pass_to_novelty";assert x["advance_to_gross9_novelty"] and not x["advance_to_economic_outcomes"];assert not x["postentry_return_pnl_execution_price_opened"] and not x["funding_values_opened"] and not x["gross9_rows_opened"];assert {k:v["events"] for k,v in x["support"].items()}=={"train":46,"test":102,"eval":103,"final":46};assert all(x["support_checks"].values())
