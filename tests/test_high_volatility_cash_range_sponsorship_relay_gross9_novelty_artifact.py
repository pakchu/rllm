import hashlib,json
from training import evaluate_high_volatility_cash_range_sponsorship_relay_gross9_novelty as n
EXPECTED='146cb22c0cd8921717ff23129b7e9a1a4a676e44bc00dd919eac80235b0b881f'
def test_novelty_failure_is_immutable_and_seals_economics():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()==EXPECTED;x=json.loads(n.OUTPUT.read_text());h=x.pop('manifest_hash');assert n.canonical_hash(x)==h;assert x['source_support_passed'] and not x['every_gross9_sleeve_passed'] and x['gross9_novelty_status']=='failed' and not x['advance_to_economic_outcomes'];e=x['evidence_boundary'];assert not e['outcomes_opened'] and e['btc_execution_rows_opened']==e['funding_rows_opened']==e['economic_outcome_rows_opened']==0;assert x['gross9_sleeves']['markov_transition_long']['metrics']['one_to_one_6h_max_matched_share']>.35
