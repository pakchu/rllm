"""Source-only support for frozen CAIIC-8."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_cross_asset_impact_isolation_continuation as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE='/home/pakchu/rllm/.env';START=pd.Timestamp('2023-01-01T00:00Z');END=pd.Timestamp('2026-08-01T00:00Z');PREREG_SHA='9a01373831179b84757bb22adceb992fb609947f2323a8d97b23d77daaa9d6c5';SYMBOLS=('BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT','DOGEUSDT','ADAUSDT')
SPLITS={'train':(pd.Timestamp('2023-07-01T00:00Z'),pd.Timestamp('2024-01-01T00:00Z')),'test':(pd.Timestamp('2024-01-01T00:00Z'),pd.Timestamp('2025-01-01T00:00Z')),'eval':(pd.Timestamp('2025-01-01T00:00Z'),pd.Timestamp('2026-01-01T00:00Z')),'final':(pd.Timestamp('2026-01-01T00:00Z'),END)};MIN={'train':8,'test':12,'eval':12,'final':8};CONTROLS=('btc_only_impact_tail','impact_isolation_without_return_tail','btc_return_tail_only','one_hour_stale_impact','direction_flip','forced_long')
QUERY="""SELECT symbol,date_bin('1 hour',ts,TIMESTAMPTZ '1970-01-01') AS hour_start,(array_agg(open ORDER BY ts))[1] AS hour_open,(array_agg(close ORDER BY ts DESC))[1] AS hour_close,sum(quote_asset_volume) AS quote_turnover,count(*) AS physical_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close) AND low<=least(open,close) AND high>=low AND quote_asset_volume>=0) AS coherent FROM bars_binance WHERE symbol=ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end GROUP BY symbol,hour_start ORDER BY hour_start,symbol"""
ROOT=Path('data/cross_asset_impact_isolation_continuation_sources_2023_2026');FEATURES=ROOT/'hourly_impact.csv.gz';MANIFEST=ROOT/'manifest.json';CLOCK=Path('data/cross_asset_impact_isolation_continuation_clocks_2023_2026.csv.gz');CDIR=Path('data/cross_asset_impact_isolation_continuation_controls_2023_2026');RESULT=Path('results/cross_asset_impact_isolation_continuation_support_2026-08-10.json')
def sha(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()
def ch(x:Any):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def rank(v):
 out=pd.Series(np.nan,index=v.index,dtype=float);h=[]
 for i,x in pd.to_numeric(v,errors='coerce').items():
  p=h[-2160:]
  if math.isfinite(x) and len(p)>=1440:
   a=np.asarray(p);out.at[i]=(np.sum(a<x)+.5*np.sum(a==x))/len(a)
  if math.isfinite(x):h.append(float(x))
 return out
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={'connect_timeout':10})
def load():
 from sqlalchemy import text
 db=engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={'symbols':list(SYMBOLS),'start':START.to_pydatetime(),'end':END.to_pydatetime()})
 finally:db.dispose()
def features(raw):
 f=raw.copy()
 for c in ('hour_start','first_ts','last_ts'):f[c]=pd.to_datetime(f[c],utc=True)
 for c in ('hour_open','hour_close','quote_turnover','physical_rows','distinct_rows'):f[c]=pd.to_numeric(f[c],errors='coerce')
 grid=pd.date_range(START,END,freq='1h',inclusive='left');zs={};rets={};valids={}
 for symbol in SYMBOLS:
  x=f[f.symbol.eq(symbol)].drop_duplicates('hour_start',keep=False).set_index('hour_start').reindex(grid);valid=np.isfinite(x[['hour_open','hour_close','quote_turnover','physical_rows','distinct_rows']]).all(axis=1)&x.hour_open.gt(0)&x.hour_close.gt(0)&x.quote_turnover.gt(0)&x.physical_rows.eq(60)&x.distinct_rows.eq(60)&x.coherent.fillna(False).astype(bool)&x.first_ts.eq(pd.Series(grid,index=grid))&x.last_ts.eq(pd.Series(grid+pd.Timedelta('59m'),index=grid));ret=np.log(x.hour_close/x.hour_open).where(valid);impact=np.log(ret.abs()/x.quote_turnover).where(valid&ret.ne(0));mu=impact.shift(1).rolling(720,min_periods=672).mean();sd=impact.shift(1).rolling(720,min_periods=672).std(ddof=1);zs[symbol]=((impact-mu)/sd).where(valid&(sd>0));rets[symbol]=ret;valids[symbol]=valid
 z=pd.DataFrame(zs,index=grid);ok=z.notna().all(axis=1)&pd.DataFrame(valids,index=grid).all(axis=1);btc=rets['BTCUSDT'].where(ok);alt=z[list(SYMBOLS[1:])].median(axis=1);q=(z.BTCUSDT-alt).where(ok);o=pd.DataFrame({'decision_time':grid+pd.Timedelta('1h'),'feature_available_time':grid+pd.Timedelta('1h'),'source_valid':ok.to_numpy(),'btc_return':btc.to_numpy(),'btc_impact_z':z.BTCUSDT.where(ok).to_numpy(),'alt_median_impact_z':alt.to_numpy(),'impact_isolation':q.to_numpy()});o['btc_abs_return_rank']=rank(o.btc_return.abs().where(o.source_valid));o['btc_impact_rank']=rank(o.btc_impact_z.where(o.source_valid));o['isolation_rank']=rank(o.impact_isolation.where(o.source_valid));return o
def active(f,control='primary'):
 state=f.source_valid&f.btc_return.ne(0)&f.isolation_rank.ge(.9)&f.btc_abs_return_rank.ge(.8);side=f.btc_return
 if control=='btc_only_impact_tail':state=f.source_valid&f.btc_return.ne(0)&f.btc_impact_rank.ge(.9)&f.btc_abs_return_rank.ge(.8);on=state&f.btc_impact_rank.shift(1).lt(.9)
 elif control=='impact_isolation_without_return_tail':state=f.source_valid&f.btc_return.ne(0)&f.isolation_rank.ge(.9);on=state&f.isolation_rank.shift(1).lt(.9)
 elif control=='btc_return_tail_only':state=f.source_valid&f.btc_return.ne(0)&f.btc_abs_return_rank.ge(.8);on=state&f.btc_abs_return_rank.shift(1).lt(.8)
 elif control=='one_hour_stale_impact':state=state.shift(1,fill_value=False);side=side.shift(1);on=state&~state.shift(1,fill_value=False)
 else:on=state&f.isolation_rank.shift(1).lt(.9)
 s=np.sign(side);s=-s if control=='direction_flip' else pd.Series(1,index=f.index) if control=='forced_long' else s;return on&side.ne(0),pd.Series(s,index=f.index).fillna(0).astype(int)
def clock(f,control='primary'):
 on,side=active(f,control);rows=[];reserved=None
 for i in f.index[on&side.ne(0)]:
  d=pd.Timestamp(f.at[i,'decision_time']);en=d+pd.Timedelta('5m');ex=en+pd.Timedelta('8h')
  if reserved is not None and en<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if en>=a and ex<=b),None)
  if not split:continue
  reserved=ex;rows.append({'candidate':'CAIIC-8','control':control,'split':split,'decision_time':d,'feature_available_time':d,'entry_time':en,'exit_time':ex,'side':int(side.at[i]),'btc_return':float(f.at[i,'btc_return']),'btc_abs_return_rank':float(f.at[i,'btc_abs_return_rank']),'impact_isolation':float(f.at[i,'impact_isolation']),'isolation_rank':float(f.at[i,'isolation_rank'])})
 return pd.DataFrame(rows,columns=['candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','btc_return','btc_abs_return_rank','impact_isolation','isolation_rank'])
def stats(c,n):
 x=c[c.split.eq(n)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());return {'events':len(x),'longs':l,'shorts':s,'minority_side_share':min(l,s)/len(x),'max_month_share':int(x.entry_time.dt.strftime('%Y-%m').value_counts().max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('CAIIC prereg drift')
 raw=load();f=features(raw);p=clock(f);cs={n:clock(f,n) for n in CONTROLS};ROOT.mkdir(parents=True,exist_ok=True);CDIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(f,FEATURES);_write_gzip_csv(p,CLOCK)
 for n,x in cs.items():_write_gzip_csv(x,CDIR/f'{n}.csv.gz')
 sc={'protocol_version':'caiic_source_v1','query_sha256':hashlib.sha256(QUERY.encode()).hexdigest(),'window':[START.isoformat(),END.isoformat()],'symbols':list(SYMBOLS),'aggregate_rows':len(raw),'features':{'path':str(FEATURES),'sha256':sha(FEATURES),'rows':len(f)},'outcomes_opened':False,'gross9_rows_opened':False};sm={**sc,'manifest_hash':ch(sc)};MANIFEST.write_text(json.dumps(sm,indent=2)+'\n');su={n:stats(p,n) for n in SPLITS};checks={}
 for n,x in su.items():checks[n+'_minimum_events']=x['events']>=MIN[n];checks[n+'_side_balance']=x['minority_side_share']>=.2;checks[n+'_month_concentration']=x['max_month_share']<=.45
 passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={'protocol_version':'caiic_8_source_support_v1','policy_id':'CAIIC-8','preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':reg['manifest_hash']},'source_manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST),'manifest_hash':sm['manifest_hash']},'completed_preentry_sources_opened':True,'postentry_return_pnl_execution_price_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(p)},'controls':{n:{'path':str(CDIR/f'{n}.csv.gz'),'sha256':sha(CDIR/f'{n}.csv.gz'),'rows':len(x),'promotion_authorized':False} for n,x in cs.items()},'support':su,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':ch(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');return r
if __name__=='__main__':print(json.dumps({'passed':(r:=run())['support_passed'],'support':r['support']},indent=2))
