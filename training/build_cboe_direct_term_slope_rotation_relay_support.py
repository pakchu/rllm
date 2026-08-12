"""Build outcome-blind source support for CVDTSR-24."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import numpy as np,pandas as pd
from training import preregister_cboe_direct_term_slope_rotation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE='/home/pakchu/rllm/.env';PREREG_SHA='7844b2295c0bb104455f9bba1c44392625f705ae4920a4fa0406387d01b56add';NY=ZoneInfo('America/New_York')
SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in prereg.build()['stages'].items()};MIN={'train':8,'test':12,'eval':12,'final':8};CONTROLS=('no_rotation_tail','no_btc_variation_gate','one_session_stale_rotation','direction_flip','forced_long')
ROOT=Path('data/cboe_direct_term_slope_rotation_relay_sources_2021_2026');FEATURES=ROOT/'preentry_features.csv.gz';MANIFEST=ROOT/'manifest.json';CLOCK=Path('data/cboe_direct_term_slope_rotation_relay_clocks_2023_2026.csv.gz');CONTROL_DIR=Path('data/cboe_direct_term_slope_rotation_relay_controls_2023_2026');RESULT=Path('results/cboe_direct_term_slope_rotation_relay_support_2026-08-12.json');BUILDER=Path(__file__).relative_to(Path.cwd())
QUERY="SELECT ts,open,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
COLS=('candidate','control','split','source_date','next_source_date','decision_time','feature_available_time','entry_time','exit_time','side','direct_slope','slope_rotation','rotation_rank','btc_variation','btc_variation_rank')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def rank(s:pd.Series,n:int,m:int)->pd.Series:
 out=pd.Series(np.nan,index=s.index);h=[]
 for i,v in pd.to_numeric(s,errors='coerce').items():
  a=np.asarray(h[-n:],float)
  if math.isfinite(v) and len(a)>=m:out.at[i]=(np.sum(a<v)+.5*np.sum(a==v))/len(a)
  if math.isfinite(v):h.append(float(v))
 return out
def decision(d):return (pd.Timestamp(d).normalize().tz_localize(NY)+pd.Timedelta(hours=9,minutes=30)).tz_convert('UTC')
def load_surface():
 if sha(prereg.SURFACE)!=prereg.SURFACE_SHA or sha(prereg.MANIFEST)!=prereg.MANIFEST_SHA:raise RuntimeError('CVDTSR source hash drift')
 f=pd.read_csv(prereg.SURFACE,compression='gzip');expected=['observation_date','SKEW_close','VVIX_close','VIX9D_close','VIX_close','VIX3M_close']
 if f.columns.tolist()!=expected:raise RuntimeError('CVDTSR surface schema drift')
 f.observation_date=pd.to_datetime(f.observation_date,format='%Y-%m-%d');f=f.sort_values('observation_date').reset_index(drop=True)
 if f.observation_date.duplicated().any():raise RuntimeError('CVDTSR duplicate dates')
 for c in expected[1:]:f[c]=pd.to_numeric(f[c],errors='coerce')
 if not np.isfinite(f[expected[1:]]).all().all() or not f[expected[1:]].gt(0).all().all():raise RuntimeError('CVDTSR invalid surface')
 f['direct_slope']=np.log(f.VIX9D_close/f.VIX3M_close);f['slope_rotation']=f.direct_slope.diff();f['rotation_rank']=rank(f.slope_rotation.abs(),252,126);return f
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={'connect_timeout':10})
def load_bars(start,end):
 from sqlalchemy import text
 e=engine()
 try:
  with e.connect() as c:f=pd.read_sql_query(text(QUERY),c,params={'start':start.to_pydatetime(),'end':end.to_pydatetime()})
 finally:e.dispose()
 f.ts=pd.to_datetime(f.ts,utc=True);f=f.sort_values('ts');expected=pd.date_range(start,end,freq='1min',inclusive='left')
 if len(f)!=len(expected) or not f.ts.reset_index(drop=True).equals(pd.Series(expected,name='ts')):raise RuntimeError('CVDTSR BTC grid drift')
 for c in ('open','close'):f[c]=pd.to_numeric(f[c],errors='coerce')
 if not np.isfinite(f[['open','close']]).all().all() or not f[['open','close']].gt(0).all().all():raise RuntimeError('CVDTSR invalid BTC')
 return f.set_index('ts')
def build_features(surface,bars):
 rows=[]
 for i in range(1,len(surface)-1):
  d=decision(surface.at[i+1,'observation_date']);idx=pd.date_range(d-pd.Timedelta('24h'),d,freq='1min',inclusive='left');w=bars.reindex(idx);variation=float(np.square(np.log(w.close.to_numpy(float)/w.open.to_numpy(float))).sum()) if len(w)==1440 and not w.isna().any().any() else math.nan
  rows.append({'source_date':surface.at[i,'observation_date'],'next_source_date':surface.at[i+1,'observation_date'],'decision_time':d,'direct_slope':float(surface.at[i,'direct_slope']),'slope_rotation':float(surface.at[i,'slope_rotation']),'rotation_rank':float(surface.at[i,'rotation_rank']),'btc_variation':variation})
 f=pd.DataFrame(rows);f['btc_variation_rank']=rank(f.btc_variation,270,180);f['source_valid']=np.isfinite(f[['slope_rotation','rotation_rank','btc_variation','btc_variation_rank']]).all(axis=1)&f.slope_rotation.ne(0)&f.btc_variation.gt(0);return f
def states(f,control):
 used=f.copy()
 if control=='one_session_stale_rotation':used[['direct_slope','slope_rotation','rotation_rank']]=f[['direct_slope','slope_rotation','rotation_rank']].shift(1)
 rg=pd.Series(True,index=f.index) if control=='no_rotation_tail' else used.rotation_rank.ge(.65);vg=pd.Series(True,index=f.index) if control=='no_btc_variation_gate' else f.btc_variation_rank.ge(.65);eligible=f.source_valid&rg&vg&used.slope_rotation.ne(0);onset=eligible&~eligible.shift(1,fill_value=False)&f.source_valid.shift(1,fill_value=False);side=-np.sign(used.slope_rotation).fillna(0).astype(int)
 if control=='direction_flip':side=-side
 if control=='forced_long':side=side.where(side.eq(0),1)
 return onset&side.ne(0),side,used
def clock(f,control='primary'):
 active,side,used=states(f,control);rows=[];reserved=None
 for i in f.index[active]:
  d=pd.Timestamp(f.at[i,'decision_time']);entry=d+pd.Timedelta('5m');exit_=entry+pd.Timedelta('24h')
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({'candidate':'CVDTSR-24','control':control,'split':split,'source_date':pd.Timestamp(used.at[i,'source_date']).date().isoformat(),'next_source_date':pd.Timestamp(f.at[i,'next_source_date']).date().isoformat(),'decision_time':d,'feature_available_time':d,'entry_time':entry,'exit_time':exit_,'side':int(side.at[i]),'direct_slope':float(used.at[i,'direct_slope']),'slope_rotation':float(used.at[i,'slope_rotation']),'rotation_rank':float(used.at[i,'rotation_rank']),'btc_variation':float(f.at[i,'btc_variation']),'btc_variation_rank':float(f.at[i,'btc_variation_rank'])})
 return pd.DataFrame(rows,columns=COLS)
def stats(c,n):
 x=c[c.split.eq(n)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime('%Y-%m').value_counts();return {'events':len(x),'longs':lo,'shorts':sh,'minority_side_share':min(lo,sh)/len(x),'max_month_share':int(m.max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('CVDTSR prereg drift')
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);surface=load_surface();first=decision(surface.at[2,'observation_date'])-pd.Timedelta('24h');last=decision(surface.at[len(surface)-1,'observation_date']);bars=load_bars(first,last);f=build_features(surface,bars);primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};ROOT.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(f,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f'{n}.csv.gz')
 source_core={'protocol_version':'cvdtsr_24_sources_v1','surface':{'path':str(prereg.SURFACE),'sha256':sha(prereg.SURFACE),'manifest':str(prereg.MANIFEST),'manifest_sha256':sha(prereg.MANIFEST)},'btc_query':QUERY,'btc_rows':len(bars),'features':{'path':str(FEATURES),'sha256':sha(FEATURES),'rows':len(f)},'builder':{'path':str(BUILDER),'sha256':sha(BUILDER)},'outcomes_opened':False,'gross9_rows_opened':False};sm={**source_core,'manifest_hash':chash(source_core)};MANIFEST.write_text(json.dumps(sm,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f'{n}_minimum_events',x['events']>=MIN[n]),(f'{n}_side_balance',x['minority_side_share']>=.2),(f'{n}_month_concentration',x['max_month_share']<=.45))};passed=all(checks.values());core={'protocol_version':'cvdtsr_24_source_support_v1','policy_id':'CVDTSR-24','preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':reg['manifest_hash']},'source_manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST),'manifest_hash':sm['manifest_hash']},'completed_preentry_sources_opened':True,'postentry_return_pnl_execution_price_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(primary)},'controls':{n:{'path':str(CONTROL_DIR/f'{n}.csv.gz'),'sha256':sha(CONTROL_DIR/f'{n}.csv.gz'),'rows':len(x),'promotion_authorized':False} for n,x in controls.items()},'support':support,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n');return r
if __name__=='__main__':print(json.dumps({'passed':run()['support_passed'],'result':str(RESULT)}))
