"""Source-only support for frozen CVEH-6."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_cross_venue_efficiency_handoff_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE='/home/pakchu/rllm/.env';START=pd.Timestamp('2023-04-01T00:00:00Z');END=pd.Timestamp('2026-08-01T00:00:00Z');PREREG_SHA='7d40bb2759b395cd6fd486637b9d706ba08c5524e932857559fc545f195117af'
SPLITS={'train':(pd.Timestamp('2023-07-01T00:00:00Z'),pd.Timestamp('2024-01-01T00:00:00Z')),'test':(pd.Timestamp('2024-01-01T00:00:00Z'),pd.Timestamp('2025-01-01T00:00:00Z')),'eval':(pd.Timestamp('2025-01-01T00:00:00Z'),pd.Timestamp('2026-01-01T00:00:00Z')),'final':(pd.Timestamp('2026-01-01T00:00:00Z'),pd.Timestamp('2026-08-01T00:00:00Z'))};MIN={'train':8,'test':12,'eval':12,'final':8}
CONTROLS=('no_volume_handoff','no_efficiency_asymmetry','spot_efficiency_only','one_boundary_stale_geometry','direction_flip','forced_long','forced_short')
ROOT=Path('data/cross_venue_efficiency_handoff_sources_2023_2026');PANEL=ROOT/'states.csv.gz';MANIFEST=ROOT/'manifest.json';CLOCK=Path('data/cross_venue_efficiency_handoff_relay_clocks_2023_2026.csv.gz');CDIR=Path('data/cross_venue_efficiency_handoff_relay_controls_2023_2026');RESULT=Path('results/cross_venue_efficiency_handoff_relay_support_2026-08-10.json')
QUERY="SELECT ts,open,high,low,close,volume FROM {table} WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
def sha(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()
def ch(x:Any):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def rank(s:pd.Series):
 a=pd.to_numeric(s,errors='coerce').to_numpy(float);o=np.full(len(a),np.nan);h=[]
 for i,v in enumerate(a):
  p=np.asarray(h[-270:]);
  if math.isfinite(v) and len(p)>=180:o[i]=(np.sum(p<v)+.5*np.sum(p==v))/len(p)
  if math.isfinite(v):h.append(v)
 return pd.Series(o,index=s.index)
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={'connect_timeout':10})
def prep(x):
 x=x.copy();x.ts=pd.to_datetime(x.ts,utc=True)
 for c in ('open','high','low','close','volume'):x[c]=pd.to_numeric(x[c],errors='coerce')
 return x.drop_duplicates('ts',keep=False).set_index('ts').sort_index()
def valid(w):return len(w)==480 and np.isfinite(w[['open','high','low','close','volume']]).all().all() and w[['open','high','low','close']].gt(0).all().all() and w.volume.ge(0).all() and w.high.ge(w[['open','close']].max(axis=1)).all() and w.low.le(w[['open','close']].min(axis=1)).all() and w.high.ge(w.low).all()
def metrics(w):
 e=w.iloc[240:360];l=w.iloc[360:];r=float(np.log(l.close.iloc[-1]/l.open.iloc[0]));tv=float(np.abs(np.diff(np.log(l.close.astype(float)))).sum());a=float(np.log(l.volume.sum()/e.volume.sum()));return r,abs(r)/tv,a
def materialize():
 from sqlalchemy import text
 db=engine()
 with db.connect() as c:
  p=pd.read_sql_query(text(QUERY.format(table='bars_binance')),c,params={'start':START.to_pydatetime(),'end':END.to_pydatetime()});s=pd.read_sql_query(text(QUERY.format(table='bars_binance_spot')),c,params={'start':START.to_pydatetime(),'end':END.to_pydatetime()})
 db.dispose();p=prep(p);s=prep(s);rows=[]
 for d in pd.date_range(START.ceil('8h'),END,freq='8h',inclusive='left'):
  idx=pd.date_range(d-pd.Timedelta('8h'),d,freq='1min',inclusive='left');pw=p.reindex(idx);sw=s.reindex(idx);ok=valid(pw) and valid(sw) and min(pw.iloc[240:360].volume.sum(),pw.iloc[360:].volume.sum(),sw.iloc[240:360].volume.sum(),sw.iloc[360:].volume.sum())>0
  if ok:
   pr,pe,pa=metrics(pw);sr,se,sa=metrics(sw);rv=float(np.square(np.diff(np.log(pw.close.astype(float)))).sum());ok=min(pe,se,rv)>0
  if not ok:pr=pe=pa=sr=se=sa=rv=np.nan
  rows.append({'decision_time':d,'source_valid':ok,'spot_return':sr,'perp_return':pr,'spot_efficiency':se,'perp_efficiency':pe,'cash_handoff':sa-pa,'realized_variation':rv})
 f=pd.DataFrame(rows)
 for c in ('spot_efficiency','perp_efficiency','cash_handoff','realized_variation'):f[c+'_rank']=rank(f[c].where(f.source_valid))
 ROOT.mkdir(parents=True,exist_ok=True);_write_gzip_csv(f,PANEL);core={'protocol_version':'cveh_source_v1','queries':{'perp':QUERY.format(table='bars_binance'),'spot':QUERY.format(table='bars_binance_spot')},'window':[START.isoformat(),END.isoformat()],'outcomes_opened':False,'candidate_incidence_opened_before_materialization':False,'output':{'path':str(PANEL),'sha256':sha(PANEL),'rows':len(f),'valid_rows':int(f.source_valid.sum())}};m={**core,'manifest_hash':ch(core)};MANIFEST.write_text(json.dumps(m,indent=2)+'\n');return f,m
def active(f,control):
 sr,pr,se,pe,h,rv=(f.spot_return,f.perp_return,f.spot_efficiency_rank,f.perp_efficiency_rank,f.cash_handoff_rank,f.realized_variation_rank)
 if control=='one_boundary_stale_geometry':sr,pr,se,pe,h,rv=[x.shift(1) for x in (sr,pr,se,pe,h,rv)]
 agree=sr.ne(0)&pr.ne(0)&np.sign(sr).eq(np.sign(pr));eff=pd.Series(True,index=f.index) if control=='no_efficiency_asymmetry' else se.ge(.8)&pe.le(.5);hand=pd.Series(True,index=f.index) if control=='no_volume_handoff' else h.ge(.75)
 if control=='spot_efficiency_only':eff=se.ge(.8)
 e=f.source_valid&rv.ge(.65)&agree&eff&hand;return e&~e.shift(1,fill_value=False)&f.source_valid.shift(1,fill_value=False),np.sign(sr)
def clock(f,control='primary'):
 on,side=active(f,control);rows=[];next_=None
 for i in f.index[on]:
  d=pd.Timestamp(f.at[i,'decision_time']);en=d+pd.Timedelta('5m');ex=en+pd.Timedelta('6h')
  if next_ is not None and en<next_:continue
  split=next((n for n,(a,b) in SPLITS.items() if en>=a and ex<=b),None)
  if not split:continue
  sd=int(side.at[i]);sd=-sd if control=='direction_flip' else 1 if control=='forced_long' else -1 if control=='forced_short' else sd;next_=ex;rows.append({'candidate':'CVEH-6','control':control,'split':split,'decision_time':d,'feature_available_time':d,'entry_time':en,'exit_time':ex,'side':sd})
 return pd.DataFrame(rows,columns=['candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side'])
def stats(c,n):
 x=c[c.split.eq(n)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());return {'events':len(x),'longs':l,'shorts':s,'minority_side_share':min(l,s)/len(x),'max_month_share':int(x.entry_time.dt.strftime('%Y-%m').value_counts().max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('CVEH prereg drift')
 f,m=materialize();p=clock(f);cs={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CDIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(p,CLOCK)
 for n,x in cs.items():_write_gzip_csv(x,CDIR/f'{n}.csv.gz')
 su={n:stats(p,n) for n in SPLITS};checks={}
 for n,x in su.items():checks[n+'_minimum_events']=x['events']>=MIN[n];checks[n+'_side_balance']=x['minority_side_share']>=.2;checks[n+'_month_concentration']=x['max_month_share']<=.45
 passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={'protocol_version':'cveh_6_source_support_v1','policy_id':'CVEH-6','preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':sha(prereg.DEFAULT_OUTPUT),'manifest_hash':reg['manifest_hash']},'source_manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST),'manifest_hash':m['manifest_hash']},'completed_preentry_sources_opened':True,'postentry_return_pnl_execution_price_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(p)},'controls':{n:{'path':str(CDIR/f'{n}.csv.gz'),'sha256':sha(CDIR/f'{n}.csv.gz'),'rows':len(x),'promotion_authorized':False} for n,x in cs.items()},'support':su,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':ch(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');return r
if __name__=='__main__':print(json.dumps({'passed':(r:=run())['support_passed'],'support':r['support']},indent=2))
