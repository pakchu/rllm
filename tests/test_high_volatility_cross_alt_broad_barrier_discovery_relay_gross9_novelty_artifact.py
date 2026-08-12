import hashlib,json
from training import evaluate_high_volatility_cross_alt_broad_barrier_discovery_relay_gross9_novelty as n
EXPECTED="5b5b58568170ba2c3a946ba07a574f3a04f5a6a07d6df09d4b5e50562b0a6192"
def test_novelty_pass_is_immutable_and_authorizes_economics():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()==EXPECTED;x=json.loads(n.OUTPUT.read_text());h=x.pop("manifest_hash");assert n.canonical_hash(x)==h;assert x["source_support_passed"] and x["every_gross9_sleeve_passed"] and x["gross9_novelty_status"]=="passed" and x["advance_to_economic_outcomes"];e=x["evidence_boundary"];assert e["btc_execution_rows_opened"]==e["btc_price_or_return_rows_opened"]==e["funding_rows_opened"]==e["economic_outcome_rows_opened"]==0 and not e["outcomes_opened"];assert max(v["metrics"]["one_to_one_6h_max_matched_share"] for v in x["gross9_sleeves"].values())<=.35
