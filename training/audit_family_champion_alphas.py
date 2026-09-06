"""Freeze and report two distinct pre-2024 champions per economic family."""
from __future__ import annotations
import argparse,hashlib,json
import numpy as np,pandas as pd
from training import search_meaningful_alpha_combinations as base
from training import search_regime_diverse_alpha_combinations as regime
from training import search_macro_flow_alpha_combinations as macro
OUT=base.ROOT/'research/family_champion_alphas'
DESIGN={'version':1,'source_universe':'all non-portfolio base candidates from macro-flow study, including annual ML and formula rules','selection':'top two distinct position paths per family by six-half-year 2021--2023 robust rank','reports':['2024','2025','2026H1','combined'],'report_gate':{'positive_each_period':True,'combined_10bp_return_positive':True,'combined_mdd_max':15.,'combined_entries_min':30},'costs':[0.,.0006,.001],'correlation':'2021--2023 hourly strategy return correlation','purpose':'find independent family-level alpha hidden below portfolio top-five ranking','no_report_rerank':True,'live_enabled':False}
def register():
 d={'design':DESIGN,'code_sha256':base.sha(__file__),'base_sha256':base.sha(base.__file__),'macro_sha256':base.sha(macro.__file__),'market_sha256':base.sha(base.MARKET),'funding_sha256':base.sha(base.FUNDING)};p=OUT/'design.json'
 if p.exists() and json.loads(p.read_text())!=d:raise RuntimeError('Family audit design drift')
 base.write_json(p,d);return d

def run():
 reg=json.loads((OUT/'design.json').read_text());
 if reg!=register():raise RuntimeError('Registration changed')
 m,f=base.load_sources();x=base.features(m,f);x,data,receipt=base.execution_blocks(m,f,x)
 raw=pd.read_csv(base.MARKET,usecols=['date','dxy','usdkrw','kimchi_premium','dxy_available','usdkrw_available','kimchi_available']);x=pd.concat([x,macro.macro_features(raw,x.index)],axis=1)
 signals,specs,notes=macro.candidates(x,data);names=list(signals);p=np.column_stack(list(signals.values()));score,stats,halves=regime.rank(data,p);selection=base.window_mask(data,'2021-01-01','2024-01-01')
 chosen=[];counts={};seen=set()
 for i in np.argsort(-score,kind='stable'):
  family=specs[names[i]]['family'];h=hashlib.sha256(p[selection,i].tobytes()).hexdigest()
  if counts.get(family,0)>=2 or h in seen:continue
  seen.add(h);counts[family]=counts.get(family,0)+1;chosen.append(int(i))
 freeze={'selection':'2021--2023 only','all_families_retained':True,'report_reranking':False,'champions':[{'name':names[i],'spec':specs[names[i]],'rank_score':float(score[i]),'selection':base.stats_row(stats,i),'half_sharpes':halves[:,i].tolist(),'position_hash':hashlib.sha256(p[selection,i].tobytes()).hexdigest()} for i in chosen]};base.write_json(OUT/'selection_freeze.json',freeze)
 cp=p[:,chosen];cn=[names[i] for i in chosen];windows={'report2024':('2024-01-01','2025-01-01'),'report2025':('2025-01-01','2026-01-01'),'report2026':('2026-01-01','2026-06-01'),'combined':('2024-01-01','2026-06-01')};reports={}
 for window,(a,b) in windows.items():
  use=base.window_mask(data,a,b);reports[window]={}
  for cost in DESIGN['costs']:
   r=base.simulate(base.subset(data,use),cp[use],cost=cost,fine=True);reports[window][str(cost)]={name:base.stats_row(r,j) for j,name in enumerate(cn)}
 gate={}
 for name in cn:
  periods=[reports[w]['0.0006'][name] for w in ['report2024','report2025','report2026']];combined=reports['combined']['0.0006'][name];stress=reports['combined']['0.001'][name]
  checks={'positive_each_period':all(r['return_pct']>0 for r in periods),'combined_10bp_return_positive':stress['return_pct']>0,'combined_mdd':combined['mdd_pct']<=15,'combined_entries':combined['entry_episodes']>=30};gate[name]={'checks':checks,'passed':all(checks.values())}
 select_returns=base.simulate(base.subset(data,selection),cp[selection],cost=.0006)['returns'];corr=np.corrcoef(select_returns.T)
 result={'registration':reg,'source_receipt':receipt,'ml_fit_audit':notes,'freeze':freeze,'reports':reports,'report_gate':gate,'passing':[n for n in cn if gate[n]['passed']],'selection_return_correlation':{'names':cn,'matrix':corr.tolist()},'live_enabled':False};base.write_json(OUT/'report.json',result)
 print('families',len(counts),'champions',len(cn),'passing',result['passing'],flush=True)
 for n in result['passing']:
  print(n, specs[n],flush=True)
  for w in windows:print(w,reports[w]['0.0006'][n],flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--freeze',action='store_true');q=a.parse_args();register() if q.freeze else run()
