import hashlib,json
from training import evaluate_high_volatility_cross_alt_volume_shock_consensus_relay_gross9_novelty as n
EXPECTED="92fa8d8691c90f00b6d265ae482f7aa2bcb39e7c54089a933128291695acfc9f"
def test_novelty_failure_is_immutable_and_terminal_before_outcomes():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()==EXPECTED;x=json.loads(n.OUTPUT.read_text());h=x.pop("manifest_hash");assert n.canonical_hash(x)==h;assert x["source_support_passed"] and not x["every_gross9_sleeve_passed"] and x["gross9_novelty_status"]=="failed" and not x["advance_to_economic_outcomes"];assert x["gross9_sleeves"]["frozen_annual_rank7"]["metrics"]["one_to_one_6h_max_matched_share"]>.35;e=x["evidence_boundary"];assert e["btc_execution_rows_opened"]==e["btc_price_or_return_rows_opened"]==e["funding_rows_opened"]==e["economic_outcome_rows_opened"]==0 and not e["outcomes_opened"]
