"""Outcome-blind source support for frozen HVCASPCPSC-8."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_cross_alt_serial_persistence_premium_sponsorship_relay as prereg
ENV_FILE='/home/pakchu/rllm/.env';START=pd.Timestamp('2023-01-01T00:00:00Z');END=pd.Timestamp('2026-08-01T00:00:00Z');PREREG_SHA='98d0b3ae230334542aadb217cf53411c0291f0f78149d83616d1af74646ecfe4';REG=prereg.build();P=REG['policy'];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG['stages'].items()};GATES=REG['source_support_gates'];CONTROLS=tuple(REG['diagnostic_controls']['names']);SYMBOLS=('BTCUSDT',*prereg.ALTS)
QUERY="""WITH five AS (SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS bar_start,symbol,ln((array_agg(close ORDER BY ts DESC))[1]/(array_agg(open ORDER BY ts))[1]) AS bar_return,sum(power(ln(close/open),2)) AS minute_squared_return,count(*) AS minute_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol=ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1,2), tagged AS (SELECT *,date_bin('8 hours',bar_start,TIMESTAMPTZ '1970-01-01 03:00:00+00') AS block_start FROM five), lagged AS (SELECT *,lag(bar_return) OVER (PARTITION BY block_start,symbol ORDER BY bar_start) AS prior_return FROM tagged) SELECT block_start,symbol,sum(bar_return) AS displacement,corr(prior_return,bar_return) FILTER (WHERE prior_return IS NOT NULL) AS serial_persistence,sum(minute_squared_return) AS minute_squared_return,sum(minute_rows) AS source_rows,count(*) AS five_minute_bars,min(bar_start) AS first_bar,max(bar_start) AS last_bar,bool_and(minute_rows=5 AND distinct_rows=5 AND first_ts=bar_start AND last_ts=bar_start+INTERVAL '4 minutes' AND coherent) AS coherent FROM lagged GROUP BY 1,2 ORDER BY 1,2"""
PREMIUM_QUERY="""SELECT ts,open,high,low,close FROM bars_binance_premium WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT=Path('data/high_volatility_cross_alt_serial_persistence_premium_sponsorship_relay_sources_2023_2026');PANEL=ROOT/'block_states.csv.gz';MANIFEST=ROOT/'manifest.json';CLOCK=Path('data/high_volatility_cross_alt_serial_persistence_premium_sponsorship_relay_clocks_2023_2026.csv.gz');SPLIT_DIR=Path('data/high_volatility_cross_alt_serial_persistence_premium_sponsorship_relay_split_clocks_2023_2026');CONTROL_DIR=Path('data/high_volatility_cross_alt_serial_persistence_premium_sponsorship_relay_controls_2023_2026');RESULT=Path('results/high_volatility_cross_alt_serial_persistence_premium_sponsorship_relay_support_2026-08-16.json');BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=('decision_time','feature_available_time','source_valid','consensus_side','consensus_breadth','consensus_strength','broad_side','broad_breadth','negative_side','negative_breadth','negative_strength','btc_realized_variation','variation_rank','premium_open','premium_close','premium_displacement','eligible');CLOCK_COLUMNS=('candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','consensus_breadth','consensus_strength','btc_realized_variation','variation_rank','premium_open','premium_close','premium_displacement')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return prereg.canonical_hash(x)
def causal(series):
 vals=pd.to_numeric(series,errors='coerce').to_numpy(float);out=np.full(len(vals),np.nan);hist=[]
 for i,v in enumerate(vals):
  prior=np.asarray(hist[-P['history_decisions']:],float)
  if math.isfinite(v) and len(prior)>=P['minimum_history_decisions']:out[i]=float((np.sum(prior<v)+.5*np.sum(prior==v))/len(prior))
  if math.isfinite(v):hist.append(float(v))
 return pd.Series(out,index=series.index)
def consensus(signs,ranks,qualifier,threshold):
 s=np.asarray(signs,float);r=np.asarray(ranks,float);tail=np.asarray(qualifier,bool)&np.isfinite(s)&np.isfinite(r)&(s!=0)&(r>=threshold);pos=int(np.sum(tail&(s>0)));neg=int(np.sum(tail&(s<0)))
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
   bars=pd.read_sql_query(text(QUERY),c,params={'symbols':list(SYMBOLS),'start':START,'end':END})
   premium=pd.read_sql_query(text(PREMIUM_QUERY),c,params={'start':START,'end':END})
   return bars,premium
 finally:db.dispose()
def prepare(raw):
 required=['block_start','symbol','displacement','serial_persistence','minute_squared_return','source_rows','five_minute_bars','first_bar','last_bar','coherent']
 if raw.columns.tolist()!=required:raise RuntimeError('HVCASPCPSC source schema drift')
 x=raw.copy()
 for c in ('block_start','first_bar','last_bar'):x[c]=pd.to_datetime(x[c],utc=True,errors='coerce')
 for c in ('displacement','serial_persistence','minute_squared_return','source_rows','five_minute_bars'):x[c]=pd.to_numeric(x[c],errors='coerce')
 if x[['block_start','symbol']].isna().any().any() or x.duplicated(['block_start','symbol']).any():raise RuntimeError('HVCASPCPSC invalid key')
 x['row_valid']=np.isfinite(x[['displacement','serial_persistence','minute_squared_return','source_rows','five_minute_bars']]).all(axis=1)&x.displacement.ne(0)&x.minute_squared_return.gt(0)&x.source_rows.eq(480)&x.five_minute_bars.eq(96)&x.first_bar.eq(x.block_start)&x.last_bar.eq(x.block_start+pd.Timedelta('475m'))&x.coherent.eq(True);x['decision_time']=x.block_start+pd.Timedelta('8h');return x.set_index(['decision_time','symbol']).sort_index()
def build_panel(raw):
 bars_raw,premium_raw=raw;x=prepare(bars_raw);decisions=pd.date_range(START+pd.Timedelta('3h'),END,freq='8h',inclusive='left');full=x.reindex(pd.MultiIndex.from_product([decisions,SYMBOLS],names=['decision_time','symbol']));valid=full.row_valid.unstack('symbol').reindex(columns=SYMBOLS);disp=full.displacement.unstack('symbol').reindex(columns=SYMBOLS);pers=full.serial_persistence.unstack('symbol').reindex(columns=SYMBOLS);sq=full.minute_squared_return.unstack('symbol').reindex(columns=SYMBOLS)
 premium=premium_raw.copy()
 if premium.columns.tolist()!=['ts','open','high','low','close']:raise RuntimeError('HVCASPCPSC premium schema drift')
 premium['ts']=pd.to_datetime(premium.ts,utc=True,errors='raise')
 for c in ('open','high','low','close'):premium[c]=pd.to_numeric(premium[c],errors='raise')
 if premium.ts.duplicated().any() or not np.isfinite(premium[['open','high','low','close']]).all().all():raise RuntimeError('HVCASPCPSC invalid premium source')
 premium=premium.set_index('ts').sort_index();ranks=pd.DataFrame(index=decisions,columns=prereg.ALTS,dtype=float);negative_ranks=pd.DataFrame(index=decisions,columns=prereg.ALTS,dtype=float)
 for alt in prereg.ALTS:ranks[alt]=causal(pers[alt].where(valid[alt]));negative_ranks[alt]=causal((-pers[alt]).where(valid[alt]))
 btcvar=np.sqrt(sq.BTCUSDT.where(valid.BTCUSDT).rolling(3,min_periods=3).sum());rows=[]
 for d in decisions:
  idx=pd.date_range(d-pd.Timedelta('8h'),d,freq='1min',inclusive='left');pw=premium.reindex(idx);premium_ok=len(pw)==480 and np.isfinite(pw).all().all() and pw.high.ge(pw[['open','close']].max(axis=1)).all() and pw.low.le(pw[['open','close']].min(axis=1)).all() and pw.high.ge(pw.low).all();premium_open=float(pw.open.iloc[0]) if premium_ok else math.nan;premium_close=float(pw.close.iloc[-1]) if premium_ok else math.nan;premium_displacement=premium_close-premium_open if premium_ok else math.nan;premium_ok=bool(premium_ok and math.isfinite(premium_displacement) and premium_displacement!=0);ok=bool(valid.loc[d].eq(True).all() and np.isfinite(ranks.loc[d]).all() and math.isfinite(float(btcvar.loc[d])) and btcvar.loc[d]>0 and premium_ok);sign=np.sign(disp.loc[d,list(prereg.ALTS)].to_numpy(float));pv=pers.loc[d,list(prereg.ALTS)].to_numpy(float);rr=ranks.loc[d].to_numpy(float);nr=negative_ranks.loc[d].to_numpy(float);side,breadth,st=consensus(sign,rr,pv>0,P['persistence_rank_min']) if ok else (0,0,math.nan);bside,bbreadth,_=consensus(sign,np.ones(6),pv>0,0.) if ok else (0,0,math.nan);nside,nbreadth,nst=consensus(sign,nr,pv<0,P['persistence_rank_min']) if ok else (0,0,math.nan);rows.append({'decision_time':d,'feature_available_time':d,'source_valid':ok,'consensus_side':side,'consensus_breadth':breadth,'consensus_strength':st,'broad_side':bside,'broad_breadth':bbreadth,'negative_side':nside,'negative_breadth':nbreadth,'negative_strength':nst,'btc_realized_variation':float(btcvar.loc[d]) if ok else math.nan,'premium_open':premium_open,'premium_close':premium_close,'premium_displacement':premium_displacement})
 panel=pd.DataFrame(rows);v=panel.source_valid.eq(True);panel['variation_rank']=causal(panel.btc_realized_variation.where(v));panel['eligible']=v&panel.consensus_breadth.ge(P['minimum_consensus_breadth'])&panel.variation_rank.ge(P['variation_rank_min'])&np.sign(panel.premium_displacement).eq(panel.consensus_side);return panel.loc[:,PANEL_COLUMNS]
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
 if control=='no_persistence_tail':side=used.broad_side;breadth=used.broad_breadth;strength=pd.Series(1.,index=used.index)
 elif control=='negative_autocorrelation_consensus':side=used.negative_side;breadth=used.negative_breadth;strength=used.negative_strength
 if control=='one_block_stale_persistence':side=side.shift(1);breadth=breadth.shift(1);strength=strength.shift(1)
 valid=used.source_valid.eq(True)&np.isfinite(strength);premium_gate=pd.Series(True,index=used.index) if control=='no_premium_sponsorship_gate' else np.sign(used.premium_displacement).eq(side);state=valid&breadth.ge(P['minimum_consensus_breadth'])&used.variation_rank.ge(P['variation_rank_min'])&premium_gate
 if control=='no_variation_gate':state=valid&breadth.ge(P['minimum_consensus_breadth'])&premium_gate
 selected=onset(state,used.source_valid.eq(True));side=pd.to_numeric(side,errors='coerce').fillna(0).astype(int)
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
  reserved=exit_;rows.append({'candidate':prereg.POLICY_ID,'control':control,'split':split,'decision_time':decision,'feature_available_time':used.at[i,'feature_available_time'],'entry_time':entry,'exit_time':exit_,'side':int(side.at[i]),'consensus_breadth':float(used.at[i,'consensus_breadth']),'consensus_strength':float(used.at[i,'consensus_strength']),'btc_realized_variation':float(used.at[i,'btc_realized_variation']),'variation_rank':float(used.at[i,'variation_rank']),'premium_open':float(used.at[i,'premium_open']),'premium_close':float(used.at[i,'premium_close']),'premium_displacement':float(used.at[i,'premium_displacement'])})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(clock,split):
 x=clock[clock.split.eq(split)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime('%Y-%m').value_counts();return {'events':len(x),'longs':l,'shorts':s,'minority_side_share':min(l,s)/len(x),'max_month_share':int(m.max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('HVCASPCPSC prereg drift')
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};common.immutable(PANEL,common.csv_gz(panel));common.immutable(CLOCK,common.csv_gz(primary))
 for n,x in controls.items():common.immutable(CONTROL_DIR/f'{n}.csv.gz',common.csv_gz(x))
 for n,x in splits.items():common.immutable(SPLIT_DIR/f'{n}.csv.gz',common.csv_gz(x))
 source_core={'protocol_version':'hvcaspc_psc_8_sources_v1','query':QUERY,'query_sha256':hashlib.sha256(QUERY.encode()).hexdigest(),'premium_query':PREMIUM_QUERY,'premium_query_sha256':hashlib.sha256(PREMIUM_QUERY.encode()).hexdigest(),'tables':['bars_binance','bars_binance_premium'],'symbols':list(SYMBOLS),'window':[START.isoformat(),END.isoformat()],'physical_rows':{'bars_aggregates':len(raw[0]),'premium':len(raw[1])},'builder':{'path':str(BUILDER),'sha256':sha(BUILDER)},'panel':{'path':str(PANEL),'sha256':sha(PANEL),'rows':len(panel),'valid_rows':int(panel.source_valid.sum())},'outcomes_opened':False,'gross9_rows_opened':False,'no_imputation':True};manifest={**source_core,'manifest_hash':chash(source_core)};common.immutable(MANIFEST,common.json_bytes(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f'{n}_minimum_events',x['events']>=GATES['minimum_events'][n]),(f'{n}_side_balance',x['minority_side_share']>=GATES['minority_side_share_min']),(f'{n}_month_concentration',x['max_month_share']<=GATES['max_month_share']))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={'protocol_version':'hvcaspc_psc_8_source_support_v1','policy_id':prereg.POLICY_ID,'preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':reg['manifest_hash']},'source_manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST),'manifest_hash':manifest['manifest_hash']},'completed_preentry_sources_opened':True,'candidate_incidence_opened':True,'postentry_return_pnl_execution_price_opened':False,'funding_values_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(primary)},'split_artifacts':{n:{'path':str(SPLIT_DIR/f'{n}.csv.gz'),'sha256':sha(SPLIT_DIR/f'{n}.csv.gz'),'rows':len(x)} for n,x in splits.items()},'controls':{n:{'path':str(CONTROL_DIR/f'{n}.csv.gz'),'sha256':sha(CONTROL_DIR/f'{n}.csv.gz'),'rows':len(x),'promotion_authorized':False} for n,x in controls.items()},'support':support,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':chash(core)};common.immutable(RESULT,common.json_bytes(r));return r
if __name__=='__main__':print(json.dumps({'passed':(r:=run())['support_passed'],'support':r['support']}))
