"""Outcome-blind source support for frozen HVCAFRU-8."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_cross_alt_funding_residual_unwind_relay as prereg
ENV_FILE='/home/pakchu/rllm/.env';START=pd.Timestamp('2023-01-01T00:00:00Z');END=pd.Timestamp('2026-08-01T00:00:00Z');PREREG_SHA='b99250927a410d8bdb710d18566515db5944287532c5bec81192333307167b73';REG=prereg.build();P=REG['policy'];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG['stages'].items()};GATES=REG['source_support_gates'];CONTROLS=tuple(REG['diagnostic_controls']['names']);SYMBOLS=('BTCUSDT',*prereg.ALTS)
FUND_QUERY="""SELECT funding_time,symbol,funding_rate FROM funding_rates_binance WHERE symbol=ANY(:symbols) AND funding_time>=:start AND funding_time<:end ORDER BY funding_time,symbol""";BAR_QUERY="""SELECT date_bin('8 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS block_start,sum(power(ln(close/open),2)) AS minute_squared_return,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""
ROOT=Path('data/high_volatility_cross_alt_funding_residual_unwind_relay_sources_2023_2026');PANEL=ROOT/'settlement_states.csv.gz';MANIFEST=ROOT/'manifest.json';CLOCK=Path('data/high_volatility_cross_alt_funding_residual_unwind_relay_clocks_2023_2026.csv.gz');SPLIT_DIR=Path('data/high_volatility_cross_alt_funding_residual_unwind_relay_split_clocks_2023_2026');CONTROL_DIR=Path('data/high_volatility_cross_alt_funding_residual_unwind_relay_controls_2023_2026');RESULT=Path('results/high_volatility_cross_alt_funding_residual_unwind_relay_support_2026-08-13.json');BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=('decision_time','feature_available_time','source_valid','consensus_side','consensus_breadth','consensus_strength','broad_side','broad_breadth','raw_side','raw_breadth','raw_strength','btc_realized_variation','variation_rank','eligible');CLOCK_COLUMNS=('candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','consensus_breadth','consensus_strength','btc_realized_variation','variation_rank')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return prereg.canonical_hash(x)
def causal(series):
 vals=pd.to_numeric(series,errors='coerce').to_numpy(float);out=np.full(len(vals),np.nan);hist=[]
 for i,v in enumerate(vals):
  prior=np.asarray(hist[-P['history_settlements']:],float)
  if math.isfinite(v) and len(prior)>=P['minimum_history_settlements']:out[i]=float((np.sum(prior<v)+.5*np.sum(prior==v))/len(prior))
  if math.isfinite(v):hist.append(float(v))
 return pd.Series(out,index=series.index)
def consensus(signs,ranks=None,threshold=0.):
 s=np.asarray(signs,float);tail=np.isfinite(s)&(s!=0)
 if ranks is not None:
  r=np.asarray(ranks,float);tail&=np.isfinite(r)&(r>=threshold)
 else:r=np.ones(len(s))
 pos=int(np.sum(tail&(s>0)));neg=int(np.sum(tail&(s<0)))
 if max(pos,neg)<P['minimum_consensus_breadth'] or pos==neg:return 0,max(pos,neg),math.nan
 side=1 if pos>neg else -1;chosen=r[tail&(np.sign(s)==side)];return side,len(chosen),float(np.median(chosen))
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={'connect_timeout':10})
def load_source():
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c:
   funding=pd.read_sql_query(text(FUND_QUERY),c,params={'symbols':list(SYMBOLS),'start':START,'end':END});bars=pd.read_sql_query(text(BAR_QUERY),c,params={'start':START,'end':END})
 finally:db.dispose()
 return funding,bars
def prepare_funding(raw):
 if raw.columns.tolist()!=['funding_time','symbol','funding_rate']:raise RuntimeError('HVCAFRU funding schema drift')
 x=raw.copy();x.funding_time=pd.to_datetime(x.funding_time,utc=True,errors='coerce');x.funding_rate=pd.to_numeric(x.funding_rate,errors='coerce')
 if x[['funding_time','symbol']].isna().any().any() or x.duplicated(['funding_time','symbol']).any() or not np.isfinite(x.funding_rate).all():raise RuntimeError('HVCAFRU invalid funding key')
 return x.pivot(index='funding_time',columns='symbol',values='funding_rate').sort_index().reindex(columns=SYMBOLS)
def prepare_bars(raw):
 required=['block_start','minute_squared_return','source_rows','distinct_rows','first_ts','last_ts','coherent']
 if raw.columns.tolist()!=required:raise RuntimeError('HVCAFRU bars schema drift')
 x=raw.copy()
 for c in ('block_start','first_ts','last_ts'):x[c]=pd.to_datetime(x[c],utc=True,errors='coerce')
 for c in ('minute_squared_return','source_rows','distinct_rows'):x[c]=pd.to_numeric(x[c],errors='coerce')
 if x.block_start.isna().any() or x.block_start.duplicated().any():raise RuntimeError('HVCAFRU invalid bars key')
 x['valid']=np.isfinite(x[['minute_squared_return','source_rows','distinct_rows']]).all(axis=1)&x.minute_squared_return.ge(0)&x.source_rows.eq(480)&x.distinct_rows.eq(480)&x.first_ts.eq(x.block_start)&x.last_ts.eq(x.block_start+pd.Timedelta('479m'))&x.coherent.eq(True);x['decision_time']=x.block_start+pd.Timedelta('8h');return x.set_index('decision_time').sort_index()
def build_panel(raw):
 funding=prepare_funding(raw[0]);bars=prepare_bars(raw[1]);decisions=pd.date_range(START+pd.Timedelta('8h'),END,freq='8h',inclusive='left');funding=funding.reindex(decisions);bars=bars.reindex(decisions);block_sq=bars.minute_squared_return.where(bars.valid.eq(True));var=np.sqrt(block_sq.rolling(3,min_periods=3).sum());source=funding.notna().all(axis=1)&bars.valid.eq(True)&block_sq.rolling(3,min_periods=3).count().eq(3);spreads=funding.loc[:,list(prereg.ALTS)].sub(funding.BTCUSDT,axis=0);ranks=pd.DataFrame(index=decisions,columns=prereg.ALTS,dtype=float);rawranks=pd.DataFrame(index=decisions,columns=prereg.ALTS,dtype=float)
 for alt in prereg.ALTS:ranks[alt]=causal(spreads[alt].abs().where(source));rawranks[alt]=causal(funding[alt].abs().where(source))
 rows=[]
 for d in decisions:
  ok=bool(source.loc[d] and np.isfinite(ranks.loc[d]).all() and math.isfinite(float(var.loc[d])) and var.loc[d]>0);ss=np.sign(spreads.loc[d].to_numpy(float));fs=np.sign(funding.loc[d,list(prereg.ALTS)].to_numpy(float));side,breadth,strength=consensus(ss,ranks.loc[d].to_numpy(float),P['spread_extremity_rank_min']) if ok else (0,0,math.nan);bside,bbreadth,_=consensus(ss) if ok else (0,0,math.nan);rawside,rawbreadth,rawstrength=consensus(fs,rawranks.loc[d].to_numpy(float),P['spread_extremity_rank_min']) if ok else (0,0,math.nan);rows.append({'decision_time':d,'feature_available_time':d,'source_valid':ok,'consensus_side':side,'consensus_breadth':breadth,'consensus_strength':strength,'broad_side':bside,'broad_breadth':bbreadth,'raw_side':rawside,'raw_breadth':rawbreadth,'raw_strength':rawstrength,'btc_realized_variation':float(var.loc[d]) if ok else math.nan})
 panel=pd.DataFrame(rows);v=panel.source_valid.eq(True);panel['variation_rank']=causal(panel.btc_realized_variation.where(v));panel['eligible']=v&panel.consensus_breadth.ge(P['minimum_consensus_breadth'])&panel.variation_rank.ge(P['variation_rank_min']);return panel.loc[:,PANEL_COLUMNS]
def onset(state,valid):
 out=pd.Series(False,index=state.index);prior=None
 for i in state.index:
  if not bool(valid.at[i]):continue
  if bool(state.at[i]) and prior is not None:out.at[i]=not bool(state.at[prior])
  prior=i
 return out
def active(panel,control='primary'):
 if control not in ('primary',*CONTROLS):raise ValueError(control)
 used=panel.copy();side=used.consensus_side;breadth=used.consensus_breadth;strength=used.consensus_strength
 if control=='no_spread_tail':side=used.broad_side;breadth=used.broad_breadth;strength=pd.Series(1.,index=used.index)
 elif control=='raw_alt_funding_consensus':side=used.raw_side;breadth=used.raw_breadth;strength=used.raw_strength
 if control=='one_settlement_stale_spreads':side=side.shift(1);breadth=breadth.shift(1);strength=strength.shift(1)
 valid=used.source_valid.eq(True)&np.isfinite(strength);state=valid&breadth.ge(P['minimum_consensus_breadth'])&used.variation_rank.ge(P['variation_rank_min'])
 if control=='no_variation_gate':state=valid&breadth.ge(P['minimum_consensus_breadth'])
 selected=onset(state,used.source_valid.eq(True));side=-pd.to_numeric(side,errors='coerce').fillna(0).astype(int)
 if control=='direction_flip':side=-side
 elif control=='forced_long':side=side.where(side.eq(0),1)
 return selected&side.ne(0),side,used
def build_clock(panel,control='primary'):
 selected,side,used=active(panel,control);rows=[];reserved=None
 for i in panel.index[selected]:
  decision=pd.Timestamp(panel.at[i,'decision_time']);entry=decision+pd.Timedelta(minutes=P['entry_delay_minutes']);exit_=entry+pd.Timedelta(hours=P['hold_hours'])
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({'candidate':prereg.POLICY_ID,'control':control,'split':split,'decision_time':decision,'feature_available_time':used.at[i,'feature_available_time'],'entry_time':entry,'exit_time':exit_,'side':int(side.at[i]),'consensus_breadth':float(used.at[i,'consensus_breadth']),'consensus_strength':float(used.at[i,'consensus_strength']),'btc_realized_variation':float(used.at[i,'btc_realized_variation']),'variation_rank':float(used.at[i,'variation_rank'])})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(clock,split):
 x=clock[clock.split.eq(split)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime('%Y-%m').value_counts();return {'events':len(x),'longs':l,'shorts':s,'minority_side_share':min(l,s)/len(x),'max_month_share':int(m.max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('HVCAFRU prereg drift')
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};common.immutable(PANEL,common.csv_gz(panel));common.immutable(CLOCK,common.csv_gz(primary))
 for n,x in controls.items():common.immutable(CONTROL_DIR/f'{n}.csv.gz',common.csv_gz(x))
 for n,x in splits.items():common.immutable(SPLIT_DIR/f'{n}.csv.gz',common.csv_gz(x))
 source_core={'protocol_version':'hvcafru_8_sources_v1','queries':{'funding':FUND_QUERY,'bars':BAR_QUERY},'query_sha256':{'funding':hashlib.sha256(FUND_QUERY.encode()).hexdigest(),'bars':hashlib.sha256(BAR_QUERY.encode()).hexdigest()},'tables':['funding_rates_binance','bars_binance'],'symbols':list(SYMBOLS),'window':[START.isoformat(),END.isoformat()],'physical_rows':{'funding':len(raw[0]),'bars':len(raw[1])},'builder':{'path':str(BUILDER),'sha256':sha(BUILDER)},'panel':{'path':str(PANEL),'sha256':sha(PANEL),'rows':len(panel),'valid_rows':int(panel.source_valid.sum())},'outcomes_opened':False,'gross9_rows_opened':False,'no_imputation':True};manifest={**source_core,'manifest_hash':chash(source_core)};common.immutable(MANIFEST,common.json_bytes(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f'{n}_minimum_events',x['events']>=GATES['minimum_events'][n]),(f'{n}_side_balance',x['minority_side_share']>=GATES['minority_side_share_min']),(f'{n}_month_concentration',x['max_month_share']<=GATES['max_month_share']))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={'protocol_version':'hvcafru_8_source_support_v1','policy_id':prereg.POLICY_ID,'preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':reg['manifest_hash']},'source_manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST),'manifest_hash':manifest['manifest_hash']},'completed_preentry_sources_opened':True,'candidate_incidence_opened':True,'postentry_return_pnl_execution_price_opened':False,'funding_signal_values_opened':True,'held_interval_funding_values_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(primary)},'split_artifacts':{n:{'path':str(SPLIT_DIR/f'{n}.csv.gz'),'sha256':sha(SPLIT_DIR/f'{n}.csv.gz'),'rows':len(x)} for n,x in splits.items()},'controls':{n:{'path':str(CONTROL_DIR/f'{n}.csv.gz'),'sha256':sha(CONTROL_DIR/f'{n}.csv.gz'),'rows':len(x),'promotion_authorized':False} for n,x in controls.items()},'support':support,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':chash(core)};common.immutable(RESULT,common.json_bytes(r));return r
if __name__=='__main__':print(json.dumps({'passed':(r:=run())['support_passed'],'support':r['support']}))
