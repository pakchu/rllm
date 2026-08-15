"""Deterministic source-only support for HVCAMLC-8."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np,pandas as pd
from training import preregister_high_volatility_cross_alt_magnitude_leadership_continuation as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE='/home/pakchu/rllm/.env';START=pd.Timestamp('2023-04-01T00:00:00Z');END=pd.Timestamp('2026-08-01T00:00:00Z');PREREG_SHA='016b0dc79fe6d85dd5dd6d09a2a7d16acc243353efd6f5859ae1ad8c171ef059';REG=prereg.build();P=REG['policy'];STAGES={k:tuple(map(pd.Timestamp,v)) for k,v in REG['stages'].items()};GATES=REG['source_support_gates'];ALTS=tuple(REG['features']['universe']);SYMBOLS=('BTCUSDT',*ALTS)
ROOT=Path('data/high_volatility_cross_alt_magnitude_leadership_continuation_sources_2023_2026');PANEL=ROOT/'states.csv.gz';MANIFEST=ROOT/'manifest.json';CLOCK=Path('data/high_volatility_cross_alt_magnitude_leadership_continuation_clocks_2023_2026.csv.gz');SPLIT_DIR=Path('data/high_volatility_cross_alt_magnitude_leadership_continuation_split_clocks_2023_2026');RESULT=Path('results/high_volatility_cross_alt_magnitude_leadership_continuation_support_2026-08-16.json')
QUERY="""SELECT date_bin('8 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS block_start,symbol,sum(ln(close/open)) AS completed_return,sum(power(ln(close/open),2)) AS minute_squared_return,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol=ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1,2 ORDER BY 1,2"""
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prior_rank(series):
 vals=pd.to_numeric(series,errors='coerce').to_numpy(float);out=np.full(len(vals),np.nan);hist=[]
 for i,v in enumerate(vals):
  prior=np.asarray(hist[-P['history_cycles']:],float)
  if math.isfinite(v) and len(prior)>=P['minimum_history_cycles']:out[i]=(np.sum(prior<v)+.5*np.sum(prior==v))/len(prior)
  if math.isfinite(v):hist.append(float(v))
 return pd.Series(out,index=series.index)
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={'connect_timeout':10})
def load_source():
 from sqlalchemy import text
 db=engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={'symbols':list(SYMBOLS),'start':START,'end':END})
 finally:db.dispose()
def build_panel(raw):
 expected=['block_start','symbol','completed_return','minute_squared_return','source_rows','distinct_rows','first_ts','last_ts','coherent']
 if raw.columns.tolist()!=expected:raise RuntimeError('HVCAMLC-8 source schema drift')
 x=raw.copy()
 for c in ('block_start','first_ts','last_ts'):x[c]=pd.to_datetime(x[c],utc=True,errors='raise')
 for c in ('completed_return','minute_squared_return','source_rows','distinct_rows'):x[c]=pd.to_numeric(x[c],errors='raise')
 x['row_valid']=(np.isfinite(x[['completed_return','minute_squared_return','source_rows','distinct_rows']]).all(axis=1)&x.completed_return.ne(0)&x.minute_squared_return.gt(0)&x.source_rows.eq(480)&x.distinct_rows.eq(480)&x.first_ts.eq(x.block_start)&x.last_ts.eq(x.block_start+pd.Timedelta('479m'))&x.coherent.eq(True));x['decision_time']=x.block_start+pd.Timedelta('8h')
 idx=pd.date_range(START.ceil('8h'),END,freq='8h',inclusive='left');full=x.set_index(['decision_time','symbol']).reindex(pd.MultiIndex.from_product([idx,SYMBOLS],names=['decision_time','symbol']));valid=full.row_valid.unstack().reindex(columns=SYMBOLS);ret=full.completed_return.unstack().reindex(columns=SYMBOLS);sq=full.minute_squared_return.unstack().reindex(columns=SYMBOLS);rows=[]
 for d in idx:
  ok=bool(valid.loc[d].eq(True).all());btc=float(ret.at[d,'BTCUSDT']) if ok else math.nan;alts=ret.loc[d,list(ALTS)].to_numpy(float) if ok else np.full(6,np.nan);side=int(np.sign(btc)) if ok else 0;breadth=int(np.sum(np.sign(alts)==side)) if ok else 0;median_abs=float(np.median(np.abs(alts))) if ok else math.nan;var=math.sqrt(float(sq.at[d,'BTCUSDT'])) if ok else math.nan;rows.append({'decision_time':d,'feature_available_time':d,'source_valid':ok,'btc_return':btc,'alt_consensus_breadth':breadth,'median_absolute_alt_return':median_abs,'btc_absolute_return':abs(btc) if ok else math.nan,'realized_variation':var})
 panel=pd.DataFrame(rows);panel['variation_rank']=prior_rank(panel.realized_variation.where(panel.source_valid));panel['eligible']=panel.source_valid&panel.alt_consensus_breadth.ge(5)&panel.median_absolute_alt_return.gt(panel.btc_absolute_return)&panel.variation_rank.ge(P['variation_rank_min']);panel['onset']=panel.eligible&~panel.eligible.shift(1,fill_value=False)&panel.source_valid.shift(1,fill_value=False);return panel
def stage_for(entry,exit_):return next((n for n,(a,b) in STAGES.items() if a<=entry and exit_<=b),None)
def build_clock(panel):
 rows=[];reserved=None
 for row in panel.loc[panel.onset].itertuples(index=False):
  d=pd.Timestamp(row.decision_time);entry=d+pd.Timedelta(minutes=P['entry_delay_minutes']);exit_=entry+pd.Timedelta(hours=P['hold_hours'])
  if reserved is not None and entry<reserved:continue
  split=stage_for(entry,exit_)
  if split is None:continue
  side=int(np.sign(row.btc_return));reserved=exit_;rows.append({'candidate':prereg.POLICY_ID,'control':'primary','split':split,'decision_time':d,'feature_available_time':row.feature_available_time,'entry_time':entry,'exit_time':exit_,'side':side,'btc_return':row.btc_return,'alt_consensus_breadth':row.alt_consensus_breadth,'median_absolute_alt_return':row.median_absolute_alt_return,'btc_absolute_return':row.btc_absolute_return,'realized_variation':row.realized_variation,'variation_rank':row.variation_rank})
 cols=['candidate','control','split','decision_time','feature_available_time','entry_time','exit_time','side','btc_return','alt_consensus_breadth','median_absolute_alt_return','btc_absolute_return','realized_variation','variation_rank'];return pd.DataFrame(rows,columns=cols)
def stats(c,n):
 x=c[c.split.eq(n)]
 if x.empty:return {'events':0,'longs':0,'shorts':0,'minority_side_share':0.,'max_month_share':0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime('%Y-%m').value_counts();return {'events':len(x),'longs':l,'shorts':s,'minority_side_share':min(l,s)/len(x),'max_month_share':int(m.max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError('HVCAMLC-8 prereg hash drift')
 prereg.validate(json.loads(prereg.DEFAULT_OUTPUT.read_text()));raw=load_source();panel=build_panel(raw);clock=build_clock(panel);ROOT.mkdir(parents=True,exist_ok=True);SPLIT_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(panel,PANEL);_write_gzip_csv(clock,CLOCK);splits={n:clock[clock.split.eq(n)].copy() for n in STAGES}
 for n,x in splits.items():_write_gzip_csv(x,SPLIT_DIR/f'{n}.csv.gz')
 sc={'protocol_version':'hvcamlc_8_sources_v1','query':QUERY,'query_sha256':hashlib.sha256(QUERY.encode()).hexdigest(),'tables':['bars_binance'],'symbols':list(SYMBOLS),'window':[START.isoformat(),END.isoformat()],'physical_rows':len(raw),'panel':{'path':str(PANEL),'sha256':sha(PANEL),'rows':len(panel),'valid_rows':int(panel.source_valid.sum())},'outcomes_opened':False,'execution_prices_opened':False,'funding_opened':False,'gross9_rows_opened':False,'no_imputation':True};manifest={**sc,'manifest_hash':prereg.canonical_hash(sc)};MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n');support={n:stats(clock,n) for n in STAGES};checks={k:v for n,x in support.items() for k,v in ((f'{n}_minimum_events',x['events']>=GATES['minimum_events'][n]),(f'{n}_side_balance',x['minority_side_share']>=GATES['minority_side_share_min']),(f'{n}_month_concentration',x['max_month_share']<=GATES['max_month_share']))};passed=all(checks.values());core={'protocol_version':'hvcamlc_8_source_support_v1','policy_id':prereg.POLICY_ID,'preregistration':{'path':str(prereg.DEFAULT_OUTPUT),'sha256':PREREG_SHA,'manifest_hash':REG['manifest_hash']},'source_manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST),'manifest_hash':manifest['manifest_hash']},'completed_preentry_sources_opened':True,'candidate_incidence_opened':True,'postentry_return_pnl_execution_price_opened':False,'held_interval_funding_values_opened':False,'gross9_rows_opened':False,'clock':{'path':str(CLOCK),'sha256':sha(CLOCK),'rows':len(clock)},'split_artifacts':{n:{'path':str(SPLIT_DIR/f'{n}.csv.gz'),'sha256':sha(SPLIT_DIR/f'{n}.csv.gz'),'rows':len(x)} for n,x in splits.items()},'support':support,'support_checks':checks,'support_passed':passed,'advance_to_gross9_novelty':passed,'advance_to_economic_outcomes':False,'decision':'pass_to_gross9_novelty' if passed else 'terminal_source_support_reject'};r={**core,'manifest_hash':prereg.canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');return r
if __name__=='__main__':r=run();print(json.dumps({'passed':r['support_passed'],'support':r['support']},indent=2))
