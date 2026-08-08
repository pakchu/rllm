import hashlib,json
from training import evaluate_fear_greed_price_leadlag_relay_economics as economics

def test_fgplr_train_rejection_is_terminal_and_sequential():
 p=economics.OUTPUTS['train'];assert hashlib.sha256(p.read_bytes()).hexdigest()=='89a9ae5b72e21569cb3b9464b643d3f5cfacba960cc00831972d4651edde1768';r=json.loads(p.read_text());assert r['policy_id']=='FGPLR-24';assert r['stage']=='train';assert r['passed'] is False;assert r['decision']=='terminal_reject_no_repair';assert r['later_stage_outcomes_opened'] is False;assert not economics.OUTPUTS['test'].exists();assert not economics.OUTPUTS['eval'].exists();assert not economics.OUTPUTS['final'].exists()

def test_fgplr_positive_raw_edge_fails_three_strict_gates():
 r=json.loads(economics.OUTPUTS['train'].read_text());assert r['manifest_hash']==economics.canonical_hash({k:v for k,v in r.items() if k!='manifest_hash'});b=r['primary']['base'];assert b['absolute_return_pct']==6.073476647048626;assert b['cagr_to_strict_mdd']==1.5831342882545794;assert b['mean_gross_underlying_bp']==89.41268716398575;assert r['primary']['cluster_signflip']['pvalue']==.21967780322196778;assert r['primary']['stress']['cagr_to_strict_mdd']==1.3440886851792744;assert all(v['absolute_return_pct']>0 for v in r['primary']['calendar_halves'].values());assert sum(r['checks'].values())==5
