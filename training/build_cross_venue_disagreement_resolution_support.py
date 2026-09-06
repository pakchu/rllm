"""Build source-support clocks for CVDR-6 without post-entry outcomes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import preregister_cross_venue_disagreement_resolution_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

NONPRICE=Path('data/options_crowding_deleveraging_relay_sources_v4_2023_2026');PRICE=Path('data/options_oi_chase_exhaustion_sources_2023_2026')
CLOCK=Path('data/cross_venue_disagreement_resolution_relay_clocks_2023_2026.csv.gz');CONTROLDIR=Path('data/cross_venue_disagreement_resolution_relay_controls_2023_2026');RESULT=Path('results/cross_venue_disagreement_resolution_relay_support_2026-08-08.json')
SPLITS=base.SPLITS;MIN={'train':16,'test':24,'eval':24,'final':16};CONTROLS=('bvol_expanding_only','dvol_expanding_only','no_lower_return_bound','no_upper_return_bound','direction_flip');ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=('candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','bvol_body','dvol_body','hour_return','prior_abs_return_q40','prior_abs_return_q75')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def features()->pd.DataFrame:
 b,d,o,f=base.load_sources(NONPRICE);j=base.joined_features(b,d,o,f);p=pd.read_csv(PRICE/'btc_completed_hour.csv.gz',compression='gzip');p['decision_time']=pd.to_datetime(p.decision_time,utc=True,format='mixed');p['open']=pd.to_numeric(p.open,errors='coerce');p['close']=pd.to_numeric(p.close,errors='coerce');p['price_valid']=p.source_valid.astype(str).str.lower().eq('true');p['hour_return']=p.close/p.open-1;a=p.hour_return.abs().where(p.price_valid);p['q40']=a.shift(1).rolling(720,min_periods=672).quantile(.4);p['q75']=a.shift(1).rolling(720,min_periods=672).quantile(.75);j=j.merge(p[['decision_time','price_valid','hour_return','q40','q75']],on='decision_time',validate='one_to_one');j['base_valid']=j.bvol_valid&j.price_valid&np.isfinite(j[['bvol_open','bvol_close','dvol_open','dvol_close','hour_return']]).all(axis=1)&j[['bvol_open','bvol_close','dvol_open','dvol_close']].gt(0).all(axis=1)&j.hour_return.ne(0);return j
def clock(j:pd.DataFrame,control:str='primary')->pd.DataFrame:
 b,d=j.bvol_body,j.dvol_body
 if control=='bvol_expanding_only':vol=b.gt(0)
 elif control=='dvol_expanding_only':vol=d.gt(0)
 else:vol=b.ne(0)&d.ne(0)&np.sign(b).eq(-np.sign(d))
 ret=j.hour_return.ne(0)
 if control!='no_lower_return_bound':ret&=j.q40.notna()&j.hour_return.abs().ge(j.q40)
 if control!='no_upper_return_bound':ret&=j.q75.notna()&j.hour_return.abs().le(j.q75)
 active=j.base_valid&vol&ret;on=active&~active.shift(1,fill_value=False)&j.base_valid.shift(1,fill_value=False)&j.decision_time.diff().eq(pd.Timedelta(hours=1));rows=[];next_allowed=None
 for i in j.index[on]:
  decision=pd.Timestamp(j.at[i,'decision_time']);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  side=int(np.sign(j.at[i,'hour_return']));side=-side if control=='direction_flip' else side;next_allowed=exit_;rows.append({'candidate':'CVDR-6','control':control,'split':split,'decision_time':decision,'feature_available_time':decision,'entry_time':entry,'exit_time':exit_,'side':side,'bvol_body':float(b.at[i]),'dvol_body':float(d.at[i]),'hour_return':float(j.at[i,'hour_return']),'prior_abs_return_q40':float(j.at[i,'q40']),'prior_abs_return_q75':float(j.at[i,'q75'])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime('%Y-%m').value_counts();return {'events':len(x),'longs':lo,'shorts':sh,'minority_side_share':min(lo,sh)/len(x),'max_month_share':int(m.max())/len(x)}
def run()->dict:
 j=features();primary=clock(j);controls={n:clock(j,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROLDIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROLDIR/f'{n}.csv.gz')
 st={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in st.items():checks[f'{n}_minimum_events']=x['events']>=MIN[n];checks[f'{n}_side_balance']=x['minority_side_share']>=.2;checks[f'{n}_month_concentration']=x['max_month_share']<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());passed=all(checks.values());core={'protocol_version':'cvdr_6_source_support_v1','policy_id':'CVDR-6','preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':sha(prereg.DEFAULT_OUTPUT),'manifest_hash':reg['manifest_hash']},'source_manifests':{'nonprice':{'path':str(NONPRICE/'manifest.json'),'sha256':sha(NONPRICE/'manifest.json')},'completed_hour_price':{'path':str(PRICE/'manifest.json'),'sha256':sha(PRICE/'manifest.json')}},'completed_preentry_feature_price_opened':True,'postentry_return_pnl_execution_price_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(primary)},'controls':{n:{'path':str(CONTROLDIR/f'{n}.csv.gz'),'sha256':sha(CONTROLDIR/f'{n}.csv.gz'),'rows':len(x)} for n,x in controls.items()},'support':st,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':ECONOMIC_OUTCOMES_AUTHORIZED,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n');return r
if __name__=='__main__':argparse.ArgumentParser().parse_args();r=run();print(json.dumps({'passed':r['support_passed'],'support':r['support']},indent=2))
