import hashlib,json
from pathlib import Path
from training import evaluate_high_volatility_cash_volume_temporal_precedence_source_integrity as s
RESULT=Path('results/high_volatility_cash_volume_temporal_precedence_relay_source_rejection_2026-08-13.json');EXPECTED='442f45e9d56f96ef8e842cb58c0ff143cc8428a2ab65a3998d3db25bedf2b220'
def test_terminal_source_contract_rejection_is_immutable_and_outcomes_sealed():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED
 x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert s.chash(x)==h
 assert x['policy_id']=='HVCVTP-8' and not x['source_integrity_passed'] and x['decision']=='terminal_source_contract_reject' and not x['advance_to_source_support']
 assert not x['candidate_incidence_opened'] and not x['postentry_outcomes_opened'] and not x['gross9_rows_opened']
 assert x['observed']['nonnull_rows']==39501 and x['observed']['first_nonnull'].startswith('2026-07-04')
 assert not any(x['checks'].values())
