import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_basis_regime_transition_relay as p
RESULT=Path('results/high_volatility_basis_regime_transition_relay_support_2026-08-13.json');EXPECTED='bbe564385a969d621f424dce01ea328d0df2ceb6fecde4fa76ec5bf60c3e59ae'
def test_source_rejection_is_immutable_blind_and_terminal():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED;r=json.loads(RESULT.read_text());h=r.pop('manifest_hash');assert p.canonical_hash(r)==h;assert r['policy_id']=='HVBSRT-8' and r['support_passed'] is False and r['decision']=='terminal_source_support_reject';assert r['support']['train']['max_month_share']>.45 and r['support']['test']['events']<12;assert not r['advance_to_gross9_novelty'] and not r['advance_to_economic_outcomes'] and not r['postentry_return_pnl_execution_price_opened'] and not r['funding_values_opened'] and not r['gross9_rows_opened']
