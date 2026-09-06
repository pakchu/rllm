"""Outcome-blind source support for frozen HVDQOFS-12."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_daily_quarter_opening_flow_surprise_relay as prereg
ENV_FILE='/home/pakchu/rllm/.env';START=pd.Timestamp('2023-01-01T00:00:00Z');END=pd.Timestamp('2026-08-01T00:00:00Z');PREREG_SHA='2dd4dee646209a26f2348c5cb7a7a3aec3aac0d8b97bc936dd1396289e2df006';REG=prereg.build();P=REG['policy'];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG['stages'].items()};GATES=REG['source_support_gates'];CONTROLS=tuple(REG['diagnostic_controls']['names']);SYMBOLS=('BTCUSDT',*prereg.ALTS)
QUERY="""SELECT date_bin('1 day',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS source_day,symbol,sum(quote_asset_volume) FILTER (WHERE extract(minute FROM ts)::int%15<5) AS opening_quote_turnover,sum(2*taker_buy_quote-quote_asset_volume) FILTER (WHERE extract(minute FROM ts)::int%15<5) AS opening_signed_taker_quote,sum(power(ln(close/open),2)) AS minute_squared_return,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,count(*) FILTER (WHERE extract(minute FROM ts)::int%15<5) AS opening_rows,min(ts) AS first_ts,max(ts) AS last_ts,min(ts) FILTER (WHERE extract(minute FROM ts)::int%15<5) AS first_opening_ts,max(ts) FILTER (WHERE extract(minute FROM ts)::int%15<5) AS last_opening_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high) AND quote_asset_volume>=0 AND taker_buy_quote>=0 AND taker_buy_quote<=quote_asset_volume) AS coherent FROM bars_binance WHERE symbol=ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1,2 ORDER BY 1,2"""
ROOT=Path('data/high_volatility_daily_quarter_opening_flow_surprise_relay_sources_2023_2026');PANEL=ROOT/'daily_opening_flow_states.csv.gz';MANIFEST=ROOT/'manifest.json';CLOCK=Path('data/high_volatility_daily_quarter_opening_flow_surprise_relay_clocks_2023_2026.csv.gz');SPLIT_DIR=Path('data/high_volatility_daily_quarter_opening_flow_surprise_relay_split_clocks_2023_2026');CONTROL_DIR=Path('data/high_volatility_daily_quarter_opening_flow_surprise_relay_controls_2023_2026');RESULT=Path('results/high_volatility_daily_quarter_opening_flow_surprise_relay_support_2026-08-13.json');BUILDER=Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS=('source_day','decision_time','feature_available_time','source_valid','consensus_side','consensus_breadth','consensus_strength','strength_rank','raw_side','raw_breadth','raw_strength','raw_strength_rank','btc_realized_variation','variation_rank','eligible');CLOCK_COLUMNS=('candidate','control','split','source_day','decision_time','feature_available_time','entry_time','exit_time','side','consensus_breadth','consensus_strength','strength_rank','btc_realized_variation','variation_rank')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return prereg.canonical_hash(x)
def causal(series,lookback,minimum,kind):
 vals=pd.to_numeric(series,errors='coerce').to_numpy(float);out=np.full(len(vals),np.nan);hist=[]
 for i,v in enumerate(vals):
  prior=np.asarray(hist[-lookback:],float)
  if math.isfinite(v) and len(prior)>=minimum:out[i]=float(np.median(prior)) if kind=='median' else float((np.sum(prior<v)+.5*np.sum(prior==v))/len(prior))
  if math.isfinite(v):hist.append(float(v))
 return pd.Series(out,index=series.index)
def geometry(values,breadth=4):
 x=np.asarray(values,float)
 if x.shape!=(6,) or not np.isfinite(x).all() or np.any(x==0):return 0,0,math.nan
 pos=int(np.sum(x>0));neg=int(np.sum(x<0));side=1 if pos>=breadth else -1 if neg>=breadth else 0
 if not side:return 0,max(pos,neg),math.nan
 z=np.abs(x[np.sign(x)==side]);return side,len(z),float(np.median(z))
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={'connect_timeout':10})
def load_source():
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={'symbols':list(SYMBOLS),'start':START,'end':END})
 finally:db.dispose()
def prepare(raw):
 required=['source_day','symbol','opening_quote_turnover','opening_signed_taker_quote','minute_squared_return','source_rows','distinct_rows','opening_rows','first_ts','last_ts','first_opening_ts','last_opening_ts','coherent']
 if raw.columns.tolist()!=required:raise RuntimeError('HVDQOFS source schema drift')
 x=raw.copy()
 for c in ('source_day','first_ts','last_ts','first_opening_ts','last_opening_ts'):x[c]=pd.to_datetime(x[c],utc=True,errors='coerce')
 nums=('opening_quote_turnover','opening_signed_taker_quote','minute_squared_return','source_rows','distinct_rows','opening_rows')
 for c in nums:x[c]=pd.to_numeric(x[c],errors='coerce')
 if x[['source_day','symbol']].isna().any().any() or x.duplicated(['source_day','symbol']).any():raise RuntimeError('HVDQOFS invalid key')
 x['row_valid']=np.isfinite(x[list(nums)]).all(axis=1)&x.opening_quote_turnover.gt(0)&x.minute_squared_return.ge(0)&x.source_rows.eq(1440)&x.distinct_rows.eq(1440)&x.opening_rows.eq(480)&x.first_ts.eq(x.source_day)&x.last_ts.eq(x.source_day+pd.Timedelta('1439m'))&x.first_opening_ts.eq(x.source_day)&x.last_opening_ts.eq(x.source_day+pd.Timedelta('1429m'))&x.coherent.eq(True);x['opening_flow']=x.opening_signed_taker_quote/x.opening_quote_turnover;return x.set_index(['source_day','symbol']).sort_index()
def build_panel(raw):
 x=prepare(raw);days=pd.date_range(START,END,freq='1D',inclusive='left');full=x.reindex(pd.MultiIndex.from_product([days,SYMBOLS],names=['source_day','symbol']));valid=full.row_valid.unstack('symbol').reindex(columns=SYMBOLS);flow=full.opening_flow.unstack('symbol').reindex(columns=SYMBOLS);sq=full.minute_squared_return.unstack('symbol').reindex(columns=SYMBOLS);baselines=pd.DataFrame(index=days,columns=prereg.ALTS,dtype=float)
 for alt in prereg.ALTS:baselines[alt]=causal(flow[alt].where(valid[alt]),P['baseline_days'],P['minimum_baseline_days'],'median')
 innovations=flow.loc[:,list(prereg.ALTS)]-baselines;rows=[]
 for d in days:
  ok=bool(valid.loc[d].eq(True).all() and np.isfinite(innovations.loc[d]).all());side,breadth,strength=geometry(innovations.loc[d].to_numpy(float)) if ok else (0,0,math.nan);rawside,rawbreadth,rawstrength=geometry(flow.loc[d,list(prereg.ALTS)].to_numpy(float)) if ok else (0,0,math.nan);variation=float(math.sqrt(sq.at[d,'BTCUSDT'])) if ok else math.nan;ok=bool(ok and side and math.isfinite(strength) and math.isfinite(variation) and variation>0);rows.append({'source_day':d,'decision_time':d+pd.Timedelta('1D'),'feature_available_time':d+pd.Timedelta('1D'),'source_valid':ok,'consensus_side':side,'consensus_breadth':breadth,'consensus_strength':strength,'raw_side':rawside,'raw_breadth':rawbreadth,'raw_strength':rawstrength,'btc_realized_variation':variation})
 p=pd.DataFrame(rows);v=p.source_valid.eq(True);p['strength_rank']=causal(p.consensus_strength.where(v),P['history_days'],P['minimum_history_days'],'rank');p['raw_strength_rank']=causal(p.raw_strength.where(v),P['history_days'],P['minimum_history_days'],'rank');p['variation_rank']=causal(p.btc_realized_variation.where(v),P['history_days'],P['minimum_history_days'],'rank');p['eligible']=v&p.strength_rank.ge(P['strength_rank_min'])&p.variation_rank.ge(P['variation_rank_min']);return p.loc[:,PANEL_COLUMNS]
def active(panel,control='primary'):
 if control not in ('primary',*CONTROLS):raise ValueError(control)
 used=panel.copy();side=used.consensus_side;breadth=used.consensus_breadth;strength=used.consensus_strength;rank=used.strength_rank
 if control=='raw_opening_flow_level':side=used.raw_side;breadth=used.raw_breadth;strength=used.raw_strength;rank=used.raw_strength_rank
 if control=='one_day_stale_surprise':side=side.shift(1);breadth=breadth.shift(1);strength=strength.shift(1);rank=rank.shift(1)
 valid=used.source_valid.eq(True)&np.isfinite(strength);tail=pd.Series(True,index=used.index) if control=='no_strength_tail' else rank.ge(P['strength_rank_min']);variation=pd.Series(True,index=used.index) if control=='no_variation_gate' else used.variation_rank.ge(P['variation_rank_min']);selected=valid&breadth.ge(P['minimum_consensus_breadth'])&tail&variation;side=pd.to_numeric(side,errors='coerce').fillna(0).astype(int)
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
  reserved=exit_;rows.append({'candidate':prereg.POLICY_ID,'control':control,'split':split,'source_day':panel.at[i,'source_day'],'decision_time':decision,'feature_available_time':used.at[i,'feature_available_time'],'entry_time':entry,'exit_time':exit_,'side':int(side.at[i]),'consensus_breadth':float(panel.at[i,'consensus_breadth']),'consensus_strength':float(panel.at[i,'consensus_strength']),'strength_rank':float(panel.at[i,'strength_rank']),'btc_realized_variation':float(panel.at[i,'btc_realized_variation']),'variation_rank':float(panel.at[i,'variation_rank'])})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(clock,split):
 x=clock[clock.split.eq(split)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime('%Y-%m').value_counts();return {'events':len(x),'longs':l,'shorts':s,'minority_side_share':min(l,s)/len(x),'max_month_share':int(m.max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('HVDQOFS prereg drift')
 raw=load_source();panel=build_panel(raw);primary=build_clock(panel);controls={n:build_clock(panel,n) for n in CONTROLS};splits={n:primary[primary.split.eq(n)].copy() for n in SPLITS};common.immutable(PANEL,common.csv_gz(panel));common.immutable(CLOCK,common.csv_gz(primary))
 for n,x in controls.items():common.immutable(CONTROL_DIR/f'{n}.csv.gz',common.csv_gz(x))
 for n,x in splits.items():common.immutable(SPLIT_DIR/f'{n}.csv.gz',common.csv_gz(x))
 source_core={'protocol_version':'hvdqofs_12_sources_v1','query':QUERY,'query_sha256':hashlib.sha256(QUERY.encode()).hexdigest(),'table':'bars_binance','symbols':list(SYMBOLS),'window':[START.isoformat(),END.isoformat()],'physical_rows':len(raw),'builder':{'path':str(BUILDER),'sha256':sha(BUILDER)},'panel':{'path':str(PANEL),'sha256':sha(PANEL),'rows':len(panel),'valid_rows':int(panel.source_valid.sum())},'outcomes_opened':False,'gross9_rows_opened':False,'no_imputation':True};manifest={**source_core,'manifest_hash':chash(source_core)};common.immutable(MANIFEST,common.json_bytes(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f'{n}_minimum_events',x['events']>=GATES['minimum_events'][n]),(f'{n}_side_balance',x['minority_side_share']>=GATES['minority_side_share_min']),(f'{n}_month_concentration',x['max_month_share']<=GATES['max_month_share']))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={'protocol_version':'hvdqofs_12_source_support_v1','policy_id':prereg.POLICY_ID,'preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':reg['manifest_hash']},'source_manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST),'manifest_hash':manifest['manifest_hash']},'completed_preentry_sources_opened':True,'candidate_incidence_opened':True,'postentry_return_pnl_execution_price_opened':False,'funding_values_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(primary)},'split_artifacts':{n:{'path':str(SPLIT_DIR/f'{n}.csv.gz'),'sha256':sha(SPLIT_DIR/f'{n}.csv.gz'),'rows':len(x)} for n,x in splits.items()},'controls':{n:{'path':str(CONTROL_DIR/f'{n}.csv.gz'),'sha256':sha(CONTROL_DIR/f'{n}.csv.gz'),'rows':len(x),'promotion_authorized':False} for n,x in controls.items()},'support':support,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':chash(core)};common.immutable(RESULT,common.json_bytes(r));return r
if __name__=='__main__':print(json.dumps({'passed':(r:=run())['support_passed'],'support':r['support']}))
