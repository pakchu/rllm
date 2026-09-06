"""Deterministic source-only support for HVKPLC-8."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_korean_price_leadership_continuation as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE='/home/pakchu/rllm/.env';START=pd.Timestamp('2023-04-01T00:00:00Z');END=pd.Timestamp('2026-08-01T00:00:00Z')
PREREG_SHA='470f733e9925e1557e308f44aeac7c8181ce15deb86a81b66098961c3a7058b2';REGISTRATION=prereg.build();POLICY=REGISTRATION['policy'];STAGES={k:tuple(map(pd.Timestamp,v)) for k,v in REGISTRATION['stages'].items()};GATES=REGISTRATION['source_support_gates']
ROOT=Path('data/high_volatility_korean_price_leadership_continuation_sources_2023_2026');PANEL=ROOT/'states.csv.gz';MANIFEST=ROOT/'manifest.json';CLOCK=Path('data/high_volatility_korean_price_leadership_continuation_clocks_2023_2026.csv.gz');SPLIT_DIR=Path('data/high_volatility_korean_price_leadership_continuation_split_clocks_2023_2026');RESULT=Path('results/high_volatility_korean_price_leadership_continuation_support_2026-08-16.json')
QUERY="""WITH crypto AS (
SELECT date_bin('8 hours',b.ts,TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '8 hours' AS decision_time,
sum(ln(b.close/b.open)) AS binance_return,sum(ln(u.close/u.open)) AS upbit_krw_return,
sum(power(ln(b.close/b.open),2)) AS minute_squared_return,count(*) AS source_rows,count(DISTINCT b.ts) AS distinct_rows,
min(b.ts) AS first_ts,max(b.ts) AS last_ts,
bool_and(b.open>0 AND b.high>0 AND b.low>0 AND b.close>0 AND b.high>=greatest(b.open,b.close,b.low) AND b.low<=least(b.open,b.close,b.high)
AND u.open>0 AND u.high>0 AND u.low>0 AND u.close>0 AND u.high>=greatest(u.open,u.close,u.low) AND u.low<=least(u.open,u.close,u.high)) AS coherent
FROM bars_binance b JOIN bars_upbit u ON u.symbol='KRW-BTC' AND u.interval='1m' AND u.ts=b.ts
WHERE b.symbol='BTCUSDT' AND b.interval='1m' AND b.ts>=:start AND b.ts<:end GROUP BY 1)
SELECT c.*,f0.open AS usdkrw_open,f1.close AS usdkrw_close FROM crypto c
JOIN bars_polygon f0 ON f0.symbol='USDKRW' AND f0.interval='1m' AND f0.ts=c.decision_time-INTERVAL '8 hours'
JOIN bars_polygon f1 ON f1.symbol='USDKRW' AND f1.interval='1m' AND f1.ts=c.decision_time-INTERVAL '1 minute'
ORDER BY c.decision_time"""

def sha256(path:str|Path)->str:return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def prior_rank(series:pd.Series)->pd.Series:
 values=pd.to_numeric(series,errors='coerce').to_numpy(float);out=np.full(len(values),np.nan);history=[]
 for i,value in enumerate(values):
  prior=np.asarray(history[-POLICY['history_cycles']:],float)
  if math.isfinite(value) and len(prior)>=POLICY['minimum_history_cycles']:out[i]=(np.sum(prior<value)+.5*np.sum(prior==value))/len(prior)
  if math.isfinite(value):history.append(float(value))
 return pd.Series(out,index=series.index)
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={'connect_timeout':10})
def load_source()->pd.DataFrame:
 from sqlalchemy import text
 db=engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={'start':START,'end':END})
 finally:db.dispose()
def build_panel(frame:pd.DataFrame)->pd.DataFrame:
 expected=['decision_time','binance_return','upbit_krw_return','minute_squared_return','source_rows','distinct_rows','first_ts','last_ts','coherent','usdkrw_open','usdkrw_close']
 if frame.columns.tolist()!=expected:raise RuntimeError('HVKPLC-8 source schema drift')
 frame=frame.copy()
 for c in ('decision_time','first_ts','last_ts'):frame[c]=pd.to_datetime(frame[c],utc=True,errors='raise')
 for c in ('binance_return','upbit_krw_return','minute_squared_return','source_rows','distinct_rows','usdkrw_open','usdkrw_close'):frame[c]=pd.to_numeric(frame[c],errors='raise')
 start=frame.decision_time-pd.Timedelta('8h')
 frame['source_valid']=(np.isfinite(frame[['binance_return','upbit_krw_return','minute_squared_return','usdkrw_open','usdkrw_close']]).all(axis=1)&frame.source_rows.eq(480)&frame.distinct_rows.eq(480)&frame.first_ts.eq(start)&frame.last_ts.eq(frame.decision_time-pd.Timedelta('1m'))&frame.coherent.eq(True)&frame.minute_squared_return.gt(0)&frame.usdkrw_open.gt(0)&frame.usdkrw_close.gt(0))
 frame['usdkrw_return']=np.log(frame.usdkrw_close/frame.usdkrw_open);frame['upbit_implied_usd_return']=frame.upbit_krw_return-frame.usdkrw_return;frame['realized_variation']=np.sqrt(frame.minute_squared_return)
 frame['variation_rank']=prior_rank(frame.realized_variation.where(frame.source_valid))
 agreement=frame.binance_return.ne(0)&frame.upbit_implied_usd_return.ne(0)&np.sign(frame.binance_return).eq(np.sign(frame.upbit_implied_usd_return))
 leadership=np.sign(frame.binance_return)*(frame.upbit_implied_usd_return-frame.binance_return)
 frame['eligible']=frame.source_valid&agreement&leadership.gt(0)&frame.variation_rank.ge(POLICY['variation_rank_min'])
 frame['onset']=frame.eligible&~frame.eligible.shift(1,fill_value=False)&frame.source_valid.shift(1,fill_value=False);frame['feature_available_time']=frame.decision_time
 return frame
def stage_for(entry,exit_):return next((n for n,(s,e) in STAGES.items() if s<=entry and exit_<=e),None)
def build_clock(panel):
 rows=[];reserved=None
 for row in panel.loc[panel.onset].itertuples(index=False):
  decision=pd.Timestamp(row.decision_time);entry=decision+pd.Timedelta(minutes=POLICY['entry_delay_minutes']);exit_=entry+pd.Timedelta(hours=POLICY['hold_hours'])
  if reserved is not None and entry<reserved:continue
  split=stage_for(entry,exit_)
  if split is None:continue
  side=int(np.sign(row.binance_return))
  if side not in (-1,1):raise RuntimeError('HVKPLC-8 side drift')
  reserved=exit_;rows.append({'candidate':prereg.POLICY_ID,'control':'primary','split':split,'decision_time':decision,'feature_available_time':row.feature_available_time,'entry_time':entry,'exit_time':exit_,'side':side,'binance_return':row.binance_return,'upbit_krw_return':row.upbit_krw_return,'usdkrw_return':row.usdkrw_return,'upbit_implied_usd_return':row.upbit_implied_usd_return,'realized_variation':row.realized_variation,'variation_rank':row.variation_rank})
 cols=['candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','binance_return','upbit_krw_return','usdkrw_return','upbit_implied_usd_return','realized_variation','variation_rank'];return pd.DataFrame(rows,columns=cols)
def stats(clock,split):
 x=clock.loc[clock.split.eq(split)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 l=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime('%Y-%m').value_counts();return {'events':len(x),'longs':l,'shorts':sh,'minority_side_share':min(l,sh)/len(x),'max_month_share':int(m.max())/len(x)}
def run():
 if sha256(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('HVKPLC-8 preregistration hash drift')
 prereg.validate(json.loads(prereg.DEFAULT_OUTPUT.read_text()));raw=load_source();panel=build_panel(raw);clock=build_clock(panel);ROOT.mkdir(parents=True,exist_ok=True);SPLIT_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(panel,PANEL);_write_gzip_csv(clock,CLOCK);splits={n:clock.loc[clock.split.eq(n)].copy() for n in STAGES}
 for n,x in splits.items():_write_gzip_csv(x,SPLIT_DIR/f'{n}.csv.gz')
 source_core={'protocol_version':'hvkplc_8_sources_v1','query':QUERY,'query_sha256':hashlib.sha256(QUERY.encode()).hexdigest(),'tables':['bars_binance','bars_upbit','bars_polygon'],'window':[START.isoformat(),END.isoformat()],'physical_rows':len(raw),'panel':{'path':str(PANEL),'sha256':sha256(PANEL),'rows':len(panel),'valid_rows':int(panel.source_valid.sum())},'outcomes_opened':False,'execution_prices_opened':False,'funding_opened':False,'gross9_rows_opened':False,'no_imputation':True};manifest={**source_core,'manifest_hash':prereg.canonical_hash(source_core)};MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
 support={n:stats(clock,n) for n in STAGES};checks={k:v for n,x in support.items() for k,v in ((f'{n}_minimum_events',x['events']>=GATES['minimum_events'][n]),(f'{n}_side_balance',x['minority_side_share']>=GATES['minority_side_share_min']),(f'{n}_month_concentration',x['max_month_share']<=GATES['max_month_share']))};passed=all(checks.values());core={'protocol_version':'hvkplc_8_source_support_v1','policy_id':prereg.POLICY_ID,'preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':REGISTRATION['manifest_hash']},'source_manifest':{'path':str(MANIFEST),'sha256':sha256(MANIFEST),'manifest_hash':manifest['manifest_hash']},'completed_preentry_sources_opened':True,'candidate_incidence_opened':True,'postentry_return_pnl_execution_price_opened':False,'held_interval_funding_values_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha256(CLOCK),'rows':len(clock)},'split_artifacts':{n:{'path':str(SPLIT_DIR/f'{n}.csv.gz'),'sha256':sha256(SPLIT_DIR/f'{n}.csv.gz'),'rows':len(x)} for n,x in splits.items()},'support':support,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_gross9_novelty' if passed else 'terminal_source_support_reject'};result={**core,'manifest_hash':prereg.canonical_hash(core)};RESULT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
if __name__=='__main__':
 r=run();print(json.dumps({'passed':r['support_passed'],'support':r['support']},indent=2))
