"""Gate the fixed bidirectional candidate with train-fit path-memory states."""
from __future__ import annotations
import argparse,itertools,json
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market,_split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config,W,extra,mk,sim
from training.search_path_memory_bidirectional_alpha import features
def q(f,m,c,z):
 x=f.loc[m,c].to_numpy(float);x=x[np.isfinite(x)];return float(np.quantile(x,z))
def run(cfg,candidate_json):
 cand=json.load(open(candidate_json))['alpha_pool_qualifiers'][0];m=_load_market(cfg);base=build_market_feature_frame(m,window_size=cfg.window_size);f=features(m,extra(m,pd.concat([base,build_interest_features(m,base)],axis=1)));dates=pd.to_datetime(m.date);tr=_split_mask(dates,*W['train']);bl=mk(f,[(x['feature'],x['op'],x['threshold']) for x in cand['long_conditions']]);bs=mk(f,[(x['feature'],x['op'],x['threshold']) for x in cand['short_conditions']]);rows=[]
 gates=[]
 for c in ('pm_sign_entropy','pm_sign_autocorr','pm_variance_ratio_12','pm_eff_72','pm_semivol_skew'):
  for z in (.1,.2,.3,.7,.8,.9):gates.append((c,'le' if z<.5 else 'ge',q(f,tr,c,z),z))
 for c,o,t,z in gates:
  g=mk(f,[(c,o,t)])
  for apply in ('long','short','both'):
   la=bl&(g if apply in ('long','both') else True);sa=bs&(g if apply in ('short','both') else True)
   s=sim(m,dates,la,sa,cfg,cand['hold_bars'],cand['stride_bars'],cand['tp'],cand['sl'],'test2024')
   if s['longs']>=6 and s['shorts']>=6:rows.append({'gate':{'feature':c,'op':o,'threshold':t,'train_quantile':z,'apply_to':apply},'test2024':s})
 rows.sort(key=lambda r:(r['test2024']['ratio'],r['test2024']['return_pct']),reverse=True);sel=rows[:50]
 for r in sel:
  g=mk(f,[(r['gate']['feature'],r['gate']['op'],r['gate']['threshold'])]);ap=r['gate']['apply_to'];la=bl&(g if ap in ('long','both') else True);sa=bs&(g if ap in ('short','both') else True)
  for w in ('train','eval2025','ytd2026'):r[w]=sim(m,dates,la,sa,cfg,cand['hold_bars'],cand['stride_bars'],cand['tp'],cand['sl'],w)
  e=r['eval2025'];en=e['longs']>=4 and e['shorts']>=4;r['passes_alpha_pool']=en and r['test2024']['ratio']>=2.5 and e['ratio']>=2.5;r['passes_live_grade']=en and r['test2024']['ratio']>=3 and e['ratio']>=3;r['passes_2026_target']=r['ytd2026']['trades']>=8 and r['ytd2026']['ratio']>=5
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'base_candidate':cand['name'],'protocol':'fixed base entries/exits; train-fit path gate; test-only gate selection; sealed eval/2026; 6bp/side','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate-json',required=True);p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');a=p.parse_args();cj=a.candidate_json;delattr(a,'candidate_json');o=run(Config(**vars(a)),cj);print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
