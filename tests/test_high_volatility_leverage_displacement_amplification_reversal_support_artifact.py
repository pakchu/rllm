import hashlib,json
from pathlib import Path
from training import preregister_high_volatility_leverage_displacement_amplification_reversal as p
RESULT=Path('results/high_volatility_leverage_displacement_amplification_reversal_support_2026-08-13.json');EXPECTED='5eb93047bd41e1115800f3870fe6e3b88d0fc436f3c631847cd4c1a4acbcfbd1'
def test_source_pass_is_immutable_blind_and_authorizes_only_novelty():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()==EXPECTED;x=json.loads(RESULT.read_text());h=x.pop('manifest_hash');assert p.canonical_hash(x)==h;assert x['policy_id']=='HVLDAR-8' and x['support_passed'] and x['decision']=='pass_to_novelty';assert not x['postentry_return_pnl_execution_price_opened'] and not x['gross9_rows_opened'];assert {k:v['events'] for k,v in x['support'].items()}=={'train':61,'test':113,'eval':135,'final':69};assert all(x['support_checks'].values())
