import hashlib,json
from training import evaluate_high_volatility_cross_alt_sign_configuration_entropy_relay_gross9_novelty as n
EXPECTED="4bef5585f8fbe341f04c648bf13aa0eee528ef845153017cda2fda0075a35793"
def test_novelty_failure_is_immutable_and_terminal_before_outcomes():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()==EXPECTED;x=json.loads(n.OUTPUT.read_text());h=x.pop("manifest_hash");assert n.canonical_hash(x)==h;assert x["source_support_passed"] and not x["every_gross9_sleeve_passed"] and x["gross9_novelty_status"]=="failed" and not x["advance_to_economic_outcomes"];assert x["gross9_sleeves"]["cand_rex_veto_7"]["metrics"]["one_to_one_6h_max_matched_share"]>.35;e=x["evidence_boundary"];assert e["btc_execution_rows_opened"]==e["btc_price_or_return_rows_opened"]==e["funding_rows_opened"]==e["economic_outcome_rows_opened"]==0 and not e["outcomes_opened"]
