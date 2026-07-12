"""Gate fixed jump alpha entries with train-fit causal volume-clock states."""
from __future__ import annotations
import argparse,itertools,json
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market,_split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config,W,sim,mk
from training.search_jump_variation_bidirectional_alpha import features as jump_features
from training.search_volume_clock_bidirectional_alpha import features as vc_features,q
def run(cfg,candidate_json):
 cand=json.load(open(candidate_json))['alpha_pool_qualifiers'][0];m=_load_market(cfg);base=build_market_feature_frame(m,window_size=cfg.window_size);f=vc_features(m,jump_features(m,pd.concat([base,build_interest_features(m,base)],axis=1)));dates=pd.to_datetime(m.date);tr=_split_mask(dates,*W['train']);bl=mk(f,[(x['feature'],x['op'],x['threshold']) for x in cand['long_conditions']]);bs=mk(f,[(x['feature'],x['op'],x['threshold']) for x in cand['short_conditions']]);rows=[]
 feats=[f'vc_{x}_{tag}' for tag in ('0p25','0p5','1p0') for x in ('ret','duration','imbalance','speed','flow_speed')]
 for c,zv,mode in itertools.product(feats,(.1,.2,.3,.7,.8,.9),('long','short','both','same_direction')):
  if mode=='same_direction':lt=q(f,tr,c,max(zv,1-zv));st=q(f,tr,c,min(zv,1-zv));la=bl&mk(f,[(c,'ge',lt)]);sa=bs&mk(f,[(c,'le',st)]);meta={'feature':c,'mode':mode,'long_threshold':lt,'short_threshold':st}
  else:
   op='le' if zv<.5 else 'ge';thr=q(f,tr,c,zv);g=mk(f,[(c,op,thr)]);la=bl&(g if mode in ('long','both') else True);sa=bs&(g if mode in ('short','both') else True);meta={'feature':c,'mode':mode,'op':op,'threshold':thr,'train_quantile':zv}
  s=sim(m,dates,la,sa,cfg,cand['hold_bars'],cand['stride_bars'],cand['tp'],cand['sl'],'test2024')
  if s['longs']>=6 and s['shorts']>=6:rows.append({'gate':meta,'test2024':s})
 rows.sort(key=lambda r:(r['test2024']['ratio'],r['test2024']['return_pct']),reverse=True);sel=rows[:100]
 for r in sel:
  g=r['gate'];c=g['feature']
  if g['mode']=='same_direction':la=bl&mk(f,[(c,'ge',g['long_threshold'])]);sa=bs&mk(f,[(c,'le',g['short_threshold'])])
  else:
   x=mk(f,[(c,g['op'],g['threshold'])]);la=bl&(x if g['mode'] in ('long','both') else True);sa=bs&(x if g['mode'] in ('short','both') else True)
  for w in ('train','eval2025','ytd2026'):r[w]=sim(m,dates,la,sa,cfg,cand['hold_bars'],cand['stride_bars'],cand['tp'],cand['sl'],w)
  e=r['eval2025'];en=e['longs']>=4 and e['shorts']>=4;r['passes_alpha_pool']=en and r['test2024']['ratio']>=2.5 and e['ratio']>=2.5;r['passes_live_grade']=en and r['test2024']['ratio']>=3 and e['ratio']>=3;r['passes_2026_target']=r['ytd2026']['trades']>=8 and r['ytd2026']['ratio']>=5
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'base_candidate':cand['name'],'protocol':'fixed jump alpha; train-fit causal volume-clock gate; test-only selection; sealed eval/2026; 6bp/side; strict MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate-json',required=True);p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');a=p.parse_args();cj=a.candidate_json;delattr(a,'candidate_json');o=run(Config(**vars(a)),cj);print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
