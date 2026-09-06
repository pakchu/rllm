import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_cross_alt_volume_weighted_auction_consensus_relay_gross9_novelty as n
P=n.OUTPUT;EXPECTED="1668da5b4fdf67705d83b8edd350063c8da6c462b38a2c710a90dc5d66325333"
def test_novelty_failure_is_immutable_and_terminal_before_outcomes():
 assert hashlib.sha256(P.read_bytes()).hexdigest()==EXPECTED;x=json.loads(P.read_text());h=x.pop("manifest_hash");assert n.canonical_hash(x)==h;assert x["source_support_passed"] and not x["every_gross9_sleeve_passed"] and x["gross9_novelty_status"]=="failed" and not x["advance_to_economic_outcomes"];assert x["gross9_sleeves"]["fresh_kimchi_fx"]["metrics"]["one_to_one_6h_max_matched_share"]>.35;e=x["evidence_boundary"];assert e["btc_execution_rows_opened"]==e["btc_price_or_return_rows_opened"]==e["funding_rows_opened"]==e["economic_outcome_rows_opened"]==0 and not e["outcomes_opened"]
