import hashlib,json
from training import evaluate_scandinavian_haven_risk_rotation_relay_economics as economics

def test_shrr_train_rejection_is_terminal_and_sequential():
 p=economics.OUTPUTS['train'];assert hashlib.sha256(p.read_bytes()).hexdigest()=='db084fd97402d4c85daf84bb228295c410fc978c933685264518be8753273669';r=json.loads(p.read_text());assert r['policy_id']=='SHRR-12';assert r['stage']=='train';assert r['passed'] is False;assert r['decision']=='terminal_reject_no_repair';assert r['later_stage_outcomes_opened'] is False;assert not economics.OUTPUTS['test'].exists();assert not economics.OUTPUTS['eval'].exists();assert not economics.OUTPUTS['final'].exists()

def test_shrr_positive_return_still_fails_three_frozen_gates():
 r=json.loads(economics.OUTPUTS['train'].read_text());assert r['manifest_hash']==economics.canonical_hash({k:v for k,v in r.items() if k!='manifest_hash'});b=r['primary']['base'];assert b['absolute_return_pct']==3.442557415023595;assert b['cagr_to_strict_mdd']==2.539424900375946;assert b['mean_gross_underlying_bp']==63.89793943948281;assert r['primary']['cluster_signflip']['pvalue']==.09222907770922291;assert r['primary']['stress']['cagr_to_strict_mdd']==2.0929367412229176;assert r['primary']['calendar_halves']['first']['absolute_return_pct']==-.003750309579453326;assert sum(r['checks'].values())==5
