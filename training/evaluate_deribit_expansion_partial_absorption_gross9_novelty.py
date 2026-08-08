"""Evaluate frozen DEPAR-6 structural novelty against every Gross9 sleeve."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
from training import export_gross9_structural_clocks as gross9
from training import evaluate_options_led_volatility_expansion_premium_relay_gross9_novelty as metric
POLICY='DEPAR-6';PROTOCOL='depar_6_gross9_novelty_v1';PREREG=Path('results/deribit_expansion_partial_absorption_relay_preregistration_2026-08-08.json');PREREG_SHA='d5ae12db6ae389de60612ab41cef3847196175ef5142d3c971eb60be64e64c46';SUPPORT=Path('results/deribit_expansion_partial_absorption_relay_support_2026-08-08.json');SUPPORT_SHA='d3473bbbe2ef39a7304f63da252e5709d612e293009bb14a4ca598cbb1ac9fff';CLOCK=Path('data/deribit_expansion_partial_absorption_relay_clocks_2023_2026.csv.gz');CLOCK_SHA='9087ed48c339edb9193172e5609ede22dca2b56ef938aff0806148c9cd4d34dd';OUTPUT=Path('results/deribit_expansion_partial_absorption_relay_gross9_novelty_2026-08-08.json');LIMITS=metric.LIMITS
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return gross9.canonical_hash(x)
def load(p:Path)->dict:
 d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 if d.get('manifest_hash')!=chash(core):raise RuntimeError(f'manifest drift: {p}')
 return d
def run(output:Path=OUTPUT)->dict:
 if sha(PREREG)!=PREREG_SHA or sha(SUPPORT)!=SUPPORT_SHA or sha(CLOCK)!=CLOCK_SHA:raise RuntimeError('DEPAR predecessor hash drift')
 reg,support=load(PREREG),load(SUPPORT);expected={'exact_entry_jaccard_max':.1,'candidate_near_6h_share_max':.45,'occupied_5m_jaccard_max':.3,'absolute_signed_exposure_pearson_max':.35,'must_pass_before_economics':True}
 if reg.get('novelty_gates')!=expected:raise RuntimeError('DEPAR novelty limits drift')
 if support.get('policy_id')!=POLICY or support.get('support_passed') is not True or support.get('advance_to_gross9_novelty') is not True or support.get('advance_to_economic_outcomes') is not False or support.get('clock',{}).get('sha256')!=CLOCK_SHA:raise RuntimeError('DEPAR predecessor state drift')
 manifest=load(gross9.DEFAULT_MANIFEST);authority=manifest.get('authority',{})
 if manifest.get('protocol_version')!=gross9.PROTOCOL_VERSION or manifest.get('all_authoritative_counts_verified') is not True or authority.get('sha256')!=gross9.ANCHOR_SHA256 or authority.get('weights')!=gross9.EXPECTED_WEIGHTS or set(manifest.get('clocks',{}))!=set(gross9.EXPECTED_WEIGHTS):raise RuntimeError('Gross9 authority drift')
 candidate=metric.load_clock(CLOCK,label='DEPAR primary');results={}
 for sleeve in gross9.EXPECTED_WEIGHTS:
  rec=manifest['clocks'][sleeve];path=Path(rec['path'])
  if sha(path)!=rec['sha256']:raise RuntimeError(f'Gross9 clock hash drift: {sleeve}')
  comparator=metric.load_clock(path,label=f'Gross9 {sleeve}')
  if len(comparator)!=rec['rows'] or len(comparator)!=sum(gross9.EXPECTED_COUNTS[sleeve].values()):raise RuntimeError(f'Gross9 count drift: {sleeve}')
  results[sleeve]=metric.evaluate_pair(candidate,comparator)
 passed=all(x['passed'] for x in results.values());advance=support['support_passed'] and passed;core={'protocol_version':PROTOCOL,'policy_id':POLICY,'preregistration':{'path':str(PREREG),'sha256':PREREG_SHA,'manifest_hash':reg['manifest_hash']},'source_support':{'path':str(SUPPORT),'sha256':SUPPORT_SHA,'manifest_hash':support['manifest_hash'],'predecessor_mutated':False},'gross9_structural_clocks':{'path':str(gross9.DEFAULT_MANIFEST),'sha256':sha(gross9.DEFAULT_MANIFEST),'manifest_hash':manifest['manifest_hash'],'authority_sha256':gross9.ANCHOR_SHA256,'complete_roster':list(gross9.EXPECTED_WEIGHTS)},'evidence_boundary':{'depar_clock_rows_opened':len(candidate),'gross9_structural_clock_rows_opened':sum(x['rows'] for x in manifest['clocks'].values()),'btc_execution_rows_opened':0,'btc_price_or_return_rows_opened':0,'funding_rows_opened':0,'economic_outcome_rows_opened':0,'portfolio_return_or_pnl_metrics_computed':False,'outcomes_opened':False},'limits':LIMITS,'gross9_sleeves':results,'source_support_passed':support['support_passed'],'every_gross9_sleeve_passed':passed,'gross9_novelty_status':'passed' if passed else 'failed','advance_to_economic_outcomes':advance,'failure_action':None if advance else 'reject DEPAR-6 unchanged before economic outcomes'};r={**core,'manifest_hash':chash(core)};output.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n');return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();r=run(a.output);print(json.dumps({'status':r['gross9_novelty_status'],'advance':r['advance_to_economic_outcomes']}))
