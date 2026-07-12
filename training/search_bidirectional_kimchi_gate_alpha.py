"""Apply train-fit Kimchi lead/lag gates to a fixed bidirectional BTC alpha."""
from __future__ import annotations
import argparse,json,itertools
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market,_split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config,W,extra,mk,sim
from training.search_kimchi_leadlag_bidirectional_alpha import features,q
def run(cfg,candidate_json):
 cand=json.load(open(candidate_json))['alpha_pool_qualifiers'][0];m=_load_market(cfg);base=build_market_feature_frame(m,window_size=cfg.window_size);f=features(m,extra(m,pd.concat([base,build_interest_features(m,base)],axis=1)));dates=pd.to_datetime(m.date);tr=_split_mask(dates,*W['train']);bl=mk(f,[(x['feature'],x['op'],x['threshold']) for x in cand['long_conditions']]);bs=mk(f,[(x['feature'],x['op'],x['threshold']) for x in cand['short_conditions']]);rows=[]
 feats=[f'kl_{x}_{n}' for x in ('kimchi_delta','kimchi_btc_gap','local_impulse') for n in (48,144,288)]+['kl_accel_48_144']
 for c,zv,mode in itertools.product(feats,(.1,.2,.3,.7,.8,.9),('same_direction','long','short','both')):
  if mode=='same_direction':lg=mk(f,[(c,'ge',q(f,tr,c,zv if zv>.5 else 1-zv))]);sg=mk(f,[(c,'le',q(f,tr,c,1-zv if zv>.5 else zv))]);la=bl&lg;sa=bs&sg;meta={'feature':c,'mode':mode,'long_op':'ge','long_threshold':q(f,tr,c,zv if zv>.5 else 1-zv),'short_op':'le','short_threshold':q(f,tr,c,1-zv if zv>.5 else zv)}
  else:
   op='le' if zv<.5 else 'ge';thr=q(f,tr,c,zv);g=mk(f,[(c,op,thr)]);la=bl&(g if mode in ('long','both') else True);sa=bs&(g if mode in ('short','both') else True);meta={'feature':c,'mode':mode,'op':op,'threshold':thr,'train_quantile':zv}
  s=sim(m,dates,la,sa,cfg,cand['hold_bars'],cand['stride_bars'],cand['tp'],cand['sl'],'test2024')
  if s['longs']>=6 and s['shorts']>=6:rows.append({'gate':meta,'test2024':s})
 rows.sort(key=lambda r:(r['test2024']['ratio'],r['test2024']['return_pct']),reverse=True);sel=rows[:80]
 for r in sel:
  g=r['gate'];c=g['feature']
  if g['mode']=='same_direction':la=bl&mk(f,[(c,g['long_op'],g['long_threshold'])]);sa=bs&mk(f,[(c,g['short_op'],g['short_threshold'])])
  else:
   x=mk(f,[(c,g['op'],g['threshold'])]);la=bl&(x if g['mode'] in ('long','both') else True);sa=bs&(x if g['mode'] in ('short','both') else True)
  for w in ('train','eval2025','ytd2026'):r[w]=sim(m,dates,la,sa,cfg,cand['hold_bars'],cand['stride_bars'],cand['tp'],cand['sl'],w)
  e=r['eval2025'];en=e['longs']>=4 and e['shorts']>=4;r['passes_alpha_pool']=en and r['test2024']['ratio']>=2.5 and e['ratio']>=2.5;r['passes_live_grade']=en and r['test2024']['ratio']>=3 and e['ratio']>=3;r['passes_2026_target']=r['ytd2026']['trades']>=8 and r['ytd2026']['ratio']>=5
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'base_candidate':cand['name'],'protocol':'fixed base/exit; train-fit Kimchi gate; test-only selection; sealed eval/2026; 6bp/side','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate-json',required=True);p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');a=p.parse_args();cj=a.candidate_json;delattr(a,'candidate_json');o=run(Config(**vars(a)),cj);print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:5]},indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
