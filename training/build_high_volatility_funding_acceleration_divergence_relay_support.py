"""Outcome-blind source support for frozen HVFADR-8."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_funding_acceleration_divergence_relay as prereg
ENV_FILE='/home/pakchu/rllm/.env';START=pd.Timestamp('2023-01-01T00:00:00Z');END=pd.Timestamp('2026-08-01T00:00:00Z');PREREG_SHA='0d8805ddff0627e9703270320d596509b115261f3a88727bb664504676fc5d44';REG=prereg.build();P=REG['policy'];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG['stages'].items()};GATES=REG['source_support_gates'];CONTROLS=tuple(REG['diagnostic_controls']['names'])
FUND_QUERY="""SELECT funding_time,funding_rate FROM funding_rates_binance WHERE symbol='BTCUSDT' AND funding_time>=:start AND funding_time<:end ORDER BY funding_time""";BAR_QUERY="""SELECT date_bin('8 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '8 hours' AS decision_time,(array_agg(open ORDER BY ts))[1] AS first_open,(array_agg(close ORDER BY ts DESC))[1] AS last_close,sum(power(ln(close/open),2)) AS minute_squared_return,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""
ROOT=Path('data/high_volatility_funding_acceleration_divergence_relay_sources_2023_2026');PANEL=ROOT/'settlement_states.csv.gz';MANIFEST=ROOT/'manifest.json';CLOCK=Path('data/high_volatility_funding_acceleration_divergence_relay_clocks_2023_2026.csv.gz');SPLIT_DIR=Path('data/high_volatility_funding_acceleration_divergence_relay_split_clocks_2023_2026');CONTROL_DIR=Path('data/high_volatility_funding_acceleration_divergence_relay_controls_2023_2026');RESULT=Path('results/high_volatility_funding_acceleration_divergence_relay_support_2026-08-13.json');BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=('decision_time','feature_available_time','source_valid','funding_rate','previous_funding_rate','funding_change','change_rank','completed_return','realized_variation','variation_rank','divergence','eligible');CLOCK_COLUMNS=('candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','funding_change','change_rank','completed_return','realized_variation','variation_rank')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return prereg.canonical_hash(x)
def causal(series):
 vals=pd.to_numeric(series,errors='coerce').to_numpy(float);out=np.full(len(vals),np.nan);hist=[]
 for i,v in enumerate(vals):
  prior=np.asarray(hist[-P['history_cycles']:],float)
  if math.isfinite(v) and len(prior)>=P['minimum_history_cycles']:out[i]=float((np.sum(prior<v)+.5*np.sum(prior==v))/len(prior))
  if math.isfinite(v):hist.append(float(v))
 return pd.Series(out,index=series.index)
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={'connect_timeout':10})
def load_source():
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c:
   funding=pd.read_sql_query(text(FUND_QUERY),c,params={'start':START,'end':END});bars=pd.read_sql_query(text(BAR_QUERY),c,params={'start':START,'end':END})
 finally:db.dispose()
 return funding,bars
def prepare(raw):
 funding,bars=raw
 if funding.columns.tolist()!=['funding_time','funding_rate'] or bars.columns.tolist()!=['decision_time','first_open','last_close','minute_squared_return','source_rows','distinct_rows','first_ts','last_ts','coherent']:raise RuntimeError('HVFADR source schema drift')
 f=funding.copy();f.funding_time=pd.to_datetime(f.funding_time,utc=True,errors='coerce');f.funding_rate=pd.to_numeric(f.funding_rate,errors='coerce')
 if f.funding_time.isna().any() or f.funding_time.duplicated().any() or not np.isfinite(f.funding_rate).all():raise RuntimeError('HVFADR invalid funding key')
 b=bars.copy()
 for c in ('decision_time','first_ts','last_ts'):b[c]=pd.to_datetime(b[c],utc=True,errors='coerce')
 for c in ('first_open','last_close','minute_squared_return','source_rows','distinct_rows'):b[c]=pd.to_numeric(b[c],errors='coerce')
 if b.decision_time.isna().any() or b.decision_time.duplicated().any():raise RuntimeError('HVFADR invalid bars key')
 start=b.decision_time-pd.Timedelta('8h');b['bar_valid']=np.isfinite(b[['first_open','last_close','minute_squared_return','source_rows','distinct_rows']]).all(axis=1)&b.first_open.gt(0)&b.last_close.gt(0)&b.minute_squared_return.gt(0)&b.source_rows.eq(480)&b.distinct_rows.eq(480)&b.first_ts.eq(start)&b.last_ts.eq(b.decision_time-pd.Timedelta('1m'))&b.coherent.eq(True);b['completed_return']=np.log(b.last_close/b.first_open);b['realized_variation']=np.sqrt(b.minute_squared_return);return f.sort_values('funding_time'),b.set_index('decision_time').sort_index()
def build_panel(raw):
 f,bars=prepare(raw);x=f.copy();x['previous_time']=x.funding_time.shift(1);x['previous_funding_rate']=x.funding_rate.shift(1);x['funding_change']=x.funding_rate-x.previous_funding_rate;x=x.join(bars[['bar_valid','completed_return','realized_variation']],on='funding_time');x['source_valid']=x.previous_time.notna()&x.funding_time.sub(x.previous_time).eq(pd.Timedelta(hours=P['required_settlement_gap_hours']))&x.bar_valid.eq(True)&np.isfinite(x[['funding_change','completed_return','realized_variation']]).all(axis=1)&x.funding_change.ne(0)&x.completed_return.ne(0)&x.realized_variation.gt(0);v=x.source_valid.eq(True);x['change_rank']=causal(x.funding_change.abs().where(v));x['variation_rank']=causal(x.realized_variation.where(v));x['divergence']=np.sign(x.completed_return).eq(-np.sign(x.funding_change));x['eligible']=v&x.change_rank.ge(P['change_extremity_rank_min'])&x.variation_rank.ge(P['variation_rank_min'])&x.divergence;x['decision_time']=x.funding_time;x['feature_available_time']=x.funding_time;return x.loc[:,PANEL_COLUMNS]
def onset(state,valid):
 out=pd.Series(False,index=state.index);prior=None
 for i in state.index:
  if not bool(valid.at[i]):continue
  if bool(state.at[i]) and prior is not None:out.at[i]=not bool(state.at[prior])
  prior=i
 return out
def active(panel,control='primary'):
 if control not in ('primary',*CONTROLS):raise ValueError(control)
 used=panel.copy();change=used.funding_change;rank=used.change_rank
 if control=='one_settlement_stale_change':change=change.shift(1);rank=rank.shift(1)
 valid=used.source_valid.eq(True)&np.isfinite(change)&np.isfinite(rank);div=np.sign(used.completed_return).eq(-np.sign(change));state=valid&div&rank.ge(P['change_extremity_rank_min'])&used.variation_rank.ge(P['variation_rank_min'])
 if control=='no_change_tail':state=valid&div&used.variation_rank.ge(P['variation_rank_min'])
 elif control=='no_variation_gate':state=valid&div&rank.ge(P['change_extremity_rank_min'])
 selected=onset(state,used.source_valid.eq(True));side=np.sign(used.completed_return).fillna(0).astype(int)
 if control=='funding_change_direction':side=np.sign(change).fillna(0).astype(int)
 elif control=='direction_flip':side=-side
 elif control=='forced_long':side=side.where(side.eq(0),1)
 return selected&side.ne(0),side,used
def build_clock(panel,control='primary'):
 selected,side,used=active(panel,control);rows=[];reserved=None
 for i in panel.index[selected]:
  decision=pd.Timestamp(panel.at[i,'decision_time']);entry=decision+pd.Timedelta(minutes=P['entry_delay_minutes']);exit_=entry+pd.Timedelta(hours=P['hold_hours'])
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({'candidate':prereg.POLICY_ID,'control':control,'split':split,'decision_time':decision,'feature_available_time':used.at[i,'feature_available_time'],'entry_time':entry,'exit_time':exit_,'side':int(side.at[i]),'funding_change':float(used.at[i,'funding_change']),'change_rank':float(used.at[i,'change_rank']),'completed_return':float(used.at[i,'completed_return']),'realized_variation':float(used.at[i,'realized_variation']),'variation_rank':float(used.at[i,'variation_rank'])})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(clock,split):
 x=clock[clock.split.eq(split)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime('%Y-%m').value_counts();return {'events':len(x),'longs':l,'shorts':s,'minority_side_share':min(l,s)/len(x),'max_month_share':int(m.max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('HVFADR prereg drift')
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};common.immutable(PANEL,common.csv_gz(panel));common.immutable(CLOCK,common.csv_gz(primary))
 for n,x in controls.items():common.immutable(CONTROL_DIR/f'{n}.csv.gz',common.csv_gz(x))
 for n,x in splits.items():common.immutable(SPLIT_DIR/f'{n}.csv.gz',common.csv_gz(x))
 source_core={'protocol_version':'hvfadr_8_sources_v1','queries':{'funding':FUND_QUERY,'bars':BAR_QUERY},'query_sha256':{'funding':hashlib.sha256(FUND_QUERY.encode()).hexdigest(),'bars':hashlib.sha256(BAR_QUERY.encode()).hexdigest()},'tables':['funding_rates_binance','bars_binance'],'symbol':'BTCUSDT','window':[START.isoformat(),END.isoformat()],'physical_rows':{'funding':len(raw[0]),'bars':len(raw[1])},'builder':{'path':str(BUILDER),'sha256':sha(BUILDER)},'panel':{'path':str(PANEL),'sha256':sha(PANEL),'rows':len(panel),'valid_rows':int(panel.source_valid.sum())},'outcomes_opened':False,'held_interval_funding_opened':False,'gross9_rows_opened':False,'no_imputation':True};manifest={**source_core,'manifest_hash':chash(source_core)};common.immutable(MANIFEST,common.json_bytes(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f'{n}_minimum_events',x['events']>=GATES['minimum_events'][n]),(f'{n}_side_balance',x['minority_side_share']>=GATES['minority_side_share_min']),(f'{n}_month_concentration',x['max_month_share']<=GATES['max_month_share']))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={'protocol_version':'hvfadr_8_source_support_v1','policy_id':prereg.POLICY_ID,'preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':reg['manifest_hash']},'source_manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST),'manifest_hash':manifest['manifest_hash']},'completed_preentry_sources_opened':True,'candidate_incidence_opened':True,'postentry_return_pnl_execution_price_opened':False,'held_interval_funding_values_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(primary)},'split_artifacts':{n:{'path':str(SPLIT_DIR/f'{n}.csv.gz'),'sha256':sha(SPLIT_DIR/f'{n}.csv.gz'),'rows':len(x)} for n,x in splits.items()},'controls':{n:{'path':str(CONTROL_DIR/f'{n}.csv.gz'),'sha256':sha(CONTROL_DIR/f'{n}.csv.gz'),'rows':len(x),'promotion_authorized':False} for n,x in controls.items()},'support':support,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':chash(core)};common.immutable(RESULT,common.json_bytes(r));return r
if __name__=='__main__':print(json.dumps({'passed':(r:=run())['support_passed'],'support':r['support']}))
