"""Source-only support for frozen HVHVS-8."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_high_volatility_har_variation_surprise_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE='/home/pakchu/rllm/.env';START=pd.Timestamp('2023-04-01T00:00:00Z');END=pd.Timestamp('2026-08-01T00:00:00Z');PREREG_SHA='a6a6f386218949d9e5d37d3168738970fea38cfd0ed09be5c0a7f0b0fc9ce44f'
SPLITS={'train':(pd.Timestamp('2023-07-01T00:00:00Z'),pd.Timestamp('2024-01-01T00:00:00Z')),'test':(pd.Timestamp('2024-01-01T00:00:00Z'),pd.Timestamp('2025-01-01T00:00:00Z')),'eval':(pd.Timestamp('2025-01-01T00:00:00Z'),pd.Timestamp('2026-01-01T00:00:00Z')),'final':(pd.Timestamp('2026-01-01T00:00:00Z'),pd.Timestamp('2026-08-01T00:00:00Z'))};MIN={'train':8,'test':12,'eval':12,'final':8}
CONTROLS=('no_surprise_tail','no_efficiency_gate','raw_positive_surprise','one_block_stale_geometry','direction_flip','forced_long')
ROOT=Path('data/high_volatility_har_variation_surprise_sources_2023_2026');PANEL=ROOT/'states.csv.gz';MANIFEST=ROOT/'manifest.json';CLOCK=Path('data/high_volatility_har_variation_surprise_relay_clocks_2023_2026.csv.gz');CDIR=Path('data/high_volatility_har_variation_surprise_relay_controls_2023_2026');RESULT=Path('results/high_volatility_har_variation_surprise_relay_support_2026-08-10.json')
QUERY="SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
def sha(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()
def ch(x:Any):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def rank(s:pd.Series):
 a=pd.to_numeric(s,errors='coerce').to_numpy(float);o=np.full(len(a),np.nan);h=[]
 for i,v in enumerate(a):
  p=np.asarray(h[-270:])
  if math.isfinite(v) and len(p)>=180:o[i]=(np.sum(p<v)+.5*np.sum(p==v))/len(p)
  if math.isfinite(v):h.append(v)
 return pd.Series(o,index=s.index)
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={'connect_timeout':10})
def materialize():
 from sqlalchemy import text
 db=engine()
 with db.connect() as c:f=pd.read_sql_query(text(QUERY),c,params={'start':START.to_pydatetime(),'end':END.to_pydatetime()})
 db.dispose();f.ts=pd.to_datetime(f.ts,utc=True)
 for c in ('open','high','low','close'):f[c]=pd.to_numeric(f[c],errors='coerce')
 if f.ts.duplicated().any():raise RuntimeError('duplicate HVHVS source timestamps')
 f=f.set_index('ts').sort_index();rows=[]
 for d in pd.date_range(START.ceil('8h'),END,freq='8h',inclusive='left'):
  idx=pd.date_range(d-pd.Timedelta('8h'),d,freq='1min',inclusive='left');w=f.reindex(idx);ok=len(w)==480 and np.isfinite(w).all().all() and w.gt(0).all().all() and w.high.ge(w[['open','close']].max(axis=1)).all() and w.low.le(w[['open','close']].min(axis=1)).all() and w.high.ge(w.low).all()
  if ok:
   lr=np.diff(np.log(w.close.astype(float)));rv=float(np.sqrt(np.square(lr).sum()));br=float(np.log(w.close.iloc[-1]/w.open.iloc[0]));late=float(np.log(w.close.iloc[-1]/w.open.iloc[-120]));tv=float(np.abs(lr).sum());eff=abs(br)/tv;ok=min(rv,tv,eff)>0
  if not ok:rv=br=late=eff=np.nan
  rows.append({'decision_time':d,'source_valid':ok,'realized_variation':rv,'block_return':br,'late_return':late,'path_efficiency':eff})
 x=pd.DataFrame(rows);rv=x.realized_variation.where(x.source_valid);x['har_forecast']=(rv.shift(1)+rv.shift(1).rolling(3,min_periods=3).mean()+rv.shift(1).rolling(21,min_periods=21).mean())/3;x['variation_surprise']=np.log(rv/x.har_forecast);x.loc[~x.variation_surprise.gt(0),'variation_surprise']=np.nan;x['surprise_rank']=rank(x.variation_surprise);x['efficiency_rank']=rank(x.path_efficiency.where(x.source_valid));ROOT.mkdir(parents=True,exist_ok=True);_write_gzip_csv(x,PANEL);core={'protocol_version':'hvhvs_source_v1','query':QUERY,'window':[START.isoformat(),END.isoformat()],'outcomes_opened':False,'candidate_incidence_opened_before_materialization':False,'output':{'path':str(PANEL),'sha256':sha(PANEL),'rows':len(x),'valid_rows':int(x.source_valid.sum())}};m={**core,'manifest_hash':ch(core)};MANIFEST.write_text(json.dumps(m,indent=2)+'\n');return x,m
def active(f,control):
 br,late,sr,er=(f.block_return,f.late_return,f.surprise_rank,f.efficiency_rank);positive=f.variation_surprise.gt(0)
 if control=='one_block_stale_geometry':br,late,sr,er,positive=[x.shift(1) for x in (br,late,sr,er,positive)]
 surprise=pd.Series(True,index=f.index) if control=='no_surprise_tail' else sr.ge(.75);surprise=positive if control=='raw_positive_surprise' else surprise;eff=pd.Series(True,index=f.index) if control=='no_efficiency_gate' else er.ge(.6);agree=br.ne(0)&late.ne(0)&np.sign(br).eq(np.sign(late));e=f.source_valid&positive&surprise&eff&agree;return e&~e.shift(1,fill_value=False)&f.source_valid.shift(1,fill_value=False),np.sign(br)
def clock(f,control='primary'):
 on,side=active(f,control);rows=[];next_=None
 for i in f.index[on]:
  d=pd.Timestamp(f.at[i,'decision_time']);en=d+pd.Timedelta('5m');ex=en+pd.Timedelta('8h')
  if next_ is not None and en<next_:continue
  split=next((n for n,(a,b) in SPLITS.items() if en>=a and ex<=b),None)
  if not split:continue
  sd=int(side.at[i]);sd=-sd if control=='direction_flip' else 1 if control=='forced_long' else sd;next_=ex;rows.append({'candidate':'HVHVS-8','control':control,'split':split,'decision_time':d,'feature_available_time':d,'entry_time':en,'exit_time':ex,'side':sd,'variation_surprise':float(f.at[i,'variation_surprise']),'surprise_rank':float(f.at[i,'surprise_rank']),'path_efficiency':float(f.at[i,'path_efficiency']),'efficiency_rank':float(f.at[i,'efficiency_rank'])})
 return pd.DataFrame(rows,columns=['candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','variation_surprise','surprise_rank','path_efficiency','efficiency_rank'])
def stats(c,n):
 x=c[c.split.eq(n)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());return {'events':len(x),'longs':l,'shorts':s,'minority_side_share':min(l,s)/len(x),'max_month_share':int(x.entry_time.dt.strftime('%Y-%m').value_counts().max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('HVHVS prereg drift')
 f,m=materialize();p=clock(f);cs={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CDIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(p,CLOCK)
 for n,x in cs.items():_write_gzip_csv(x,CDIR/f'{n}.csv.gz')
 su={n:stats(p,n) for n in SPLITS};checks={}
 for n,x in su.items():checks[n+'_minimum_events']=x['events']>=MIN[n];checks[n+'_side_balance']=x['minority_side_share']>=.2;checks[n+'_month_concentration']=x['max_month_share']<=.45
 passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={'protocol_version':'hvhvs_8_source_support_v1','policy_id':'HVHVS-8','preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':sha(prereg.DEFAULT_OUTPUT),'manifest_hash':reg['manifest_hash']},'source_manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST),'manifest_hash':m['manifest_hash']},'completed_preentry_sources_opened':True,'postentry_return_pnl_execution_price_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(p)},'controls':{n:{'path':str(CDIR/f'{n}.csv.gz'),'sha256':sha(CDIR/f'{n}.csv.gz'),'rows':len(x),'promotion_authorized':False} for n,x in cs.items()},'support':su,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':ch(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');return r
if __name__=='__main__':print(json.dumps({'passed':(r:=run())['support_passed'],'support':r['support']},indent=2))
