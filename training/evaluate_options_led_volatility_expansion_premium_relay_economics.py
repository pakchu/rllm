"""Sequential strict economic evaluator for the frozen OVEPR-24 clock."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training.audit_rank7_fresh_kimchi_fixed_portfolio import subaccount_bar_path,synchronized_portfolio_stats
from training.search_inventory_purge_reclaim_alpha import Config as ExecutionConfig,Trade

POLICY_ID='OVEPR-24'; BAR=pd.Timedelta(minutes=5); LEVERAGE=.5; BASE_COST=.0006; STRESS_COST=.0010
PREREG=Path('results/options_led_volatility_expansion_premium_relay_preregistration_2026-08-08.json');PREREG_SHA='180e6be7f6889024896303be511a07b3a95b44dc225f4566a2edab7127022dd6'
SUPPORT=Path('results/options_led_volatility_expansion_premium_relay_support_2026-08-08.json');SUPPORT_SHA='afbf8157c2c85aec0470563cdfba1b45afe18472a617633d2140b6ec6c1c15a7'
NOVELTY=Path('results/options_led_volatility_expansion_premium_relay_gross9_novelty_2026-08-08.json');NOVELTY_SHA='b6a5128aa259907df36b39d484f7d7bc3f142134b0d68e5cb22a0ef64ddfdd03'
CLOCK=Path('data/options_led_volatility_expansion_premium_relay_clocks_2023_2026.csv.gz');CLOCK_SHA='b79bd105784db59980a83d1e1e75e3334e954f76e0f06a6d44eca1dc017e6bf1'
CONTROL_DIR=Path('data/options_led_volatility_expansion_premium_relay_controls_2023_2026')
MARKET=Path('data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz');MARKET_SHA='a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c'
FUNDING=Path('data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz');FUNDING_SHA='4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7'
STAGES={
 'train':('train','2023-07-01T00:00:00Z','2024-01-01T00:00:00Z'),
 'test':('test','2024-01-01T00:00:00Z','2025-01-01T00:00:00Z'),
 'eval':('eval','2025-01-01T00:00:00Z','2026-01-01T00:00:00Z'),
 'final':('final','2026-01-01T00:00:00Z','2026-07-01T00:00:00Z')}
PREDECESSOR={'test':'train','eval':'test','final':'eval'}
OUTPUTS={s:Path(f'results/options_led_volatility_expansion_premium_relay_{s}_economics_2026-08-08.json') for s in STAGES}
CONTROLS=('no_deribit_lead','deribit_fall_mirror','no_premium_efficiency','direction_flip','extra_latency_1h','deterministic_random_side')
CLUSTER_DRAWS=100_000;CLUSTER_SEED=20260808

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def load_json(path:Path)->dict[str,Any]:return json.loads(path.read_text())
def verify_controls(stage:str)->dict[str,Any]:
 if sha(PREREG)!=PREREG_SHA or sha(SUPPORT)!=SUPPORT_SHA or sha(NOVELTY)!=NOVELTY_SHA or sha(CLOCK)!=CLOCK_SHA:raise RuntimeError('OVEPR frozen predecessor hash drift')
 n=load_json(NOVELTY)
 if n.get('advance_to_economic_outcomes') is not True or n.get('evidence_boundary',{}).get('outcomes_opened') is not False:raise RuntimeError('OVEPR novelty did not authorize economics')
 if stage in PREDECESSOR:
  p=OUTPUTS[PREDECESSOR[stage]]
  if not p.is_file():raise RuntimeError(f'missing passing predecessor: {p}')
  d=load_json(p);core={k:v for k,v in d.items() if k!='manifest_hash'}
  if d.get('manifest_hash')!=chash(core) or d.get('passed') is not True:raise RuntimeError('economic predecessor did not pass')
 return n

def _utc(v:Any)->pd.Timestamp:
 t=pd.Timestamp(v);return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
def _stream_gzip(path:Path,start:pd.Timestamp,end:pd.Timestamp,columns:tuple[str,...],date_col:str,include_end:bool)->pd.DataFrame:
 rows=[]
 with gzip.open(path,'rt',newline='') as f:
  reader=csv.DictReader(f)
  if not set(columns).issubset(reader.fieldnames or []):raise RuntimeError(f'schema drift: {path}')
  for row in reader:
   t=_utc(row[date_col])
   if t>end or (t==end and not include_end):break
   if t>=start:rows.append({c:row[c] for c in columns})
 out=pd.DataFrame(rows,columns=columns)
 if out.empty:raise RuntimeError(f'empty physical prefix: {path}')
 out[date_col]=pd.to_datetime(out[date_col],utc=True,errors='raise')
 return out

def load_csv_sources(start:pd.Timestamp,end:pd.Timestamp)->tuple[pd.DataFrame,pd.DataFrame,dict[str,Any]]:
 if sha(MARKET)!=MARKET_SHA or sha(FUNDING)!=FUNDING_SHA:raise RuntimeError('economic source hash drift')
 m=_stream_gzip(MARKET,start,end,('date','open','high','low','close'),'date',True)
 for c in ('open','high','low','close'):m[c]=pd.to_numeric(m[c],errors='raise')
 f=_stream_gzip(FUNDING,start,end,('date','funding_rate'),'date',False);f['funding_rate']=pd.to_numeric(f.funding_rate,errors='raise')
 return m,f,{'mode':'hash_bound_gzip_physical_prefix','market_sha256':MARKET_SHA,'funding_sha256':FUNDING_SHA}

def load_postgres_sources(start:pd.Timestamp,end:pd.Timestamp)->tuple[pd.DataFrame,pd.DataFrame,dict[str,Any]]:
 from sqlalchemy import bindparam,create_engine,text
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file('.env');engine=create_engine(postgres_url_from_env('.env'),connect_args={'connect_timeout':10})
 stmt=text("""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,(array_agg(close ORDER BY ts DESC))[1] AS close,count(*) AS source_rows FROM bars_binance WHERE interval='1m' AND symbol=:symbol AND ts>=:start AND ts<:query_end GROUP BY 1 ORDER BY 1""")
 fund=text("""SELECT funding_time AS date,funding_rate FROM funding_rates_binance WHERE symbol=:symbol AND funding_time>=:start AND funding_time<:end ORDER BY funding_time""")
 with engine.connect() as conn:
  m=pd.read_sql_query(stmt,conn,params={'symbol':'BTCUSDT','start':start.to_pydatetime(),'query_end':(end+BAR).to_pydatetime()});f=pd.read_sql_query(fund,conn,params={'symbol':'BTCUSDT','start':start.to_pydatetime(),'end':end.to_pydatetime()})
 engine.dispose();m['date']=pd.to_datetime(m.date,utc=True);f['date']=pd.to_datetime(f.date,utc=True)
 if not m.source_rows.eq(5).all():raise RuntimeError('Postgres 5m candle source is incomplete')
 return m[['date','open','high','low','close']],f[['date','funding_rate']],{'mode':'postgres_exact_1m_to_5m','table':'bars_binance','funding_table':'funding_rates_binance','symbol':'BTCUSDT'}

def load_sources(stage:str,start:pd.Timestamp,end:pd.Timestamp):return load_postgres_sources(start,end) if stage=='final' else load_csv_sources(start,end)
def load_clock(path:Path,split:str,start:pd.Timestamp,end:pd.Timestamp)->pd.DataFrame:
 d=pd.read_csv(path,compression='gzip');d=d[d['split'].eq(split)].copy()
 for c in ('entry_time','exit_time'):d[c]=pd.to_datetime(d[c],utc=True,errors='raise')
 d['side']=pd.to_numeric(d.side,errors='raise').astype(int);d=d[(d.entry_time>=start)&(d.exit_time<=end)].sort_values('entry_time').reset_index(drop=True)
 if d.empty or set(d.side)-{-1,1}:raise RuntimeError(f'invalid stage clock: {path}')
 return d

def validate_market(m:pd.DataFrame,start:pd.Timestamp,end:pd.Timestamp)->None:
 expected=pd.date_range(start,end,freq='5min',inclusive='both');dates=pd.DatetimeIndex(m.date)
 if not dates.equals(expected):raise RuntimeError(f'market grid incomplete: {len(dates)} != {len(expected)}')
 vals=m[['open','high','low','close']].to_numpy(float)
 if not np.isfinite(vals).all() or (vals<=0).any():raise RuntimeError('invalid market OHLC')
 if (m.high< m[['open','close']].max(axis=1)).any() or (m.low>m[['open','close']].min(axis=1)).any():raise RuntimeError('market envelope invalid')

def build_trades(clock:pd.DataFrame,m:pd.DataFrame,f:pd.DataFrame)->list[Trade]:
 dates=pd.DatetimeIndex(m.date);pos={t:i for i,t in enumerate(dates)};opens=m.open.to_numpy(float);highs=m.high.to_numpy(float);lows=m.low.to_numpy(float)
 ft=pd.DatetimeIndex(f.date).asi8;fr=pd.to_numeric(f.funding_rate,errors='raise').to_numpy(float);out=[]
 for r in clock.itertuples(index=False):
  if r.entry_time not in pos or r.exit_time not in pos:raise RuntimeError('clock is absent from exact market opens')
  a=pos[r.entry_time];b=pos[r.exit_time];side=int(r.side);entry=opens[a];gross=side*(opens[b]/entry-1);left=np.searchsorted(ft,r.entry_time.value,'left');right=np.searchsorted(ft,r.exit_time.value,'right');ff=1-LEVERAGE*side*fr[left:right]
  if not np.isfinite(ff).all() or (ff<=0).any():raise RuntimeError('invalid funding factor')
  held=slice(a,b);fav=(highs[held].max() if side>0 else lows[held].min());adv=(lows[held].min() if side>0 else highs[held].max())
  out.append(Trade(signal_position=a-1,entry_position=a,exit_position=b,side=side,gross_return=float(gross),price_factor=float(1+LEVERAGE*gross),funding_factor=float(np.prod(ff)) if len(ff) else 1.0,funding_debit_factor=float(np.prod(np.minimum(ff,1))) if len(ff) else 1.0,favorable_price_factor=float(1+LEVERAGE*side*(fav/entry-1)),adverse_price_factor=float(1+LEVERAGE*side*(adv/entry-1)),entry_date=str(r.entry_time)))
 return out

def cfg(cost:float)->ExecutionConfig:return ExecutionConfig(input_csv='',metrics_csv='',funding_csv='',output='',manifest_output='',leverage=LEVERAGE,fee_rate=cost,slippage_rate=0.0)
def metrics(trades:list[Trade],m:pd.DataFrame,f:pd.DataFrame,start:pd.Timestamp,end:pd.Timestamp,cost:float)->dict[str,Any]:
 path=subaccount_bar_path(m,f,trades,cfg(cost),start=str(start),end=str(end),hold_bars=lambda t:t.exit_position-t.entry_position)
 raw=synchronized_portfolio_stats({'ovepr':path},{'ovepr':1.0},start=str(start),end=str(end),trade_counts={'ovepr':len(trades)})
 gross=[t.gross_return for t in trades];net=[(1-LEVERAGE*cost)*t.price_factor*t.funding_factor*(1-LEVERAGE*cost)-1 for t in trades]
 return {'absolute_return_pct':raw['absolute_return_pct'],'cagr_pct':raw['cagr_pct'],'strict_mdd_pct':raw['synchronized_strict_mdd_pct'],'cagr_to_strict_mdd':raw['cagr_to_synchronized_strict_mdd'],'final_equity':raw['final_equity'],'trades':len(trades),'longs':sum(t.side>0 for t in trades),'shorts':sum(t.side<0 for t in trades),'mean_gross_underlying_bp':float(np.mean(gross)*1e4),'mean_net_bp':float(np.mean(net)*1e4),'win_rate':float(np.mean(np.asarray(net)>0))}
def cluster_p(trades:list[Trade],cost:float)->dict[str,Any]:
 clusters={}
 for t in trades:
  entry=_utc(t.entry_date);iso=entry.isocalendar();key=f'{int(iso.year):04d}-W{int(iso.week):02d}';effect=math.log((1-LEVERAGE*cost)*t.price_factor*t.funding_factor*(1-LEVERAGE*cost));clusters[key]=clusters.get(key,0)+effect
 vals=np.array(list(clusters.values()));obs=float(vals.sum());rng=np.random.default_rng(CLUSTER_SEED);draws=rng.choice(np.array([-1.,1.]),size=(CLUSTER_DRAWS,len(vals)));null=draws@vals;p=(1+int((null>=obs).sum()))/(CLUSTER_DRAWS+1)
 return {'method':'one_sided_UTC_week_cluster_signflip_monte_carlo','clusters':len(vals),'draws':CLUSTER_DRAWS,'seed':CLUSTER_SEED,'observed_log_effect':obs,'pvalue':p}
def evaluate_clock(clock:pd.DataFrame,m:pd.DataFrame,f:pd.DataFrame,start:pd.Timestamp,end:pd.Timestamp)->dict[str,Any]:
 trades=build_trades(clock,m,f);base=metrics(trades,m,f,start,end,BASE_COST);stress=metrics(trades,m,f,start,end,STRESS_COST);mid=start+(end-start)/2;halves={}
 for name,a,b in [('first',start,mid),('second',mid,end)]:
  sub=clock[(clock.entry_time>=a)&(clock.exit_time<=b)];halves[name]=metrics(build_trades(sub,m,f),m,f,a,b,BASE_COST)
 return {'base':base,'stress':stress,'cluster_signflip':cluster_p(trades,BASE_COST),'calendar_halves':halves}
def run(stage:str,output:Path|None=None)->dict[str,Any]:
 if stage not in STAGES:raise ValueError(stage)
 novelty=verify_controls(stage);split,s0,e0=STAGES[stage];start=_utc(s0);end=_utc(e0);m,f,source=load_sources(stage,start,end);validate_market(m,start,end);primary=load_clock(CLOCK,split,start,end);result=evaluate_clock(primary,m,f,start,end)
 controls={}
 support=load_json(SUPPORT)
 for name in CONTROLS:
  p=CONTROL_DIR/f'{name}.csv.gz';expected=support['controls'][name]['sha256']
  if sha(p)!=expected:raise RuntimeError(f'control hash drift: {name}')
  controls[name]=evaluate_clock(load_clock(p,split,start,end),m,f,start,end)
 b=result['base'];s=result['stress'];checks={'absolute_return_positive':b['absolute_return_pct']>0,'cagr_to_strict_mdd_min_3':b['cagr_to_strict_mdd']>=3,'strict_mdd_max_15':b['strict_mdd_pct']<=15,'mean_gross_move_min_20bp':b['mean_gross_underlying_bp']>=20,'cluster_signflip_p_max_0_1':result['cluster_signflip']['pvalue']<=.1,'stress_absolute_return_positive':s['absolute_return_pct']>0,'stress_cagr_to_strict_mdd_min_2_5':s['cagr_to_strict_mdd']>=2.5,'each_calendar_half_positive':all(x['absolute_return_pct']>0 for x in result['calendar_halves'].values())};passed=all(checks.values())
 core={'protocol_version':'ovepr_24_sequential_economics_v1','policy_id':POLICY_ID,'stage':stage,'window':[s0,e0],'predecessor':None if stage=='train' else {'stage':PREDECESSOR[stage],'path':str(OUTPUTS[PREDECESSOR[stage]]),'sha256':sha(OUTPUTS[PREDECESSOR[stage]])},'novelty_authorization':{'path':str(NOVELTY),'sha256':NOVELTY_SHA,'manifest_hash':novelty['manifest_hash']},'source':source,'physical_rows_opened':{'market':len(m),'funding':len(f),'primary_clock':len(primary)},'future_stage_outcomes_opened':False,'primary':result,'controls_diagnostic_only':controls,'checks':checks,'passed':passed,'advance_to_next_stage':passed and stage!='final','decision':'pass' if passed else 'terminal_reject_no_repair'};out={**core,'manifest_hash':chash(core)};dest=output or OUTPUTS[stage];dest.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n');return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=tuple(STAGES),required=True);p.add_argument('--output',type=Path);a=p.parse_args();r=run(a.stage,a.output);print(json.dumps({'stage':a.stage,'passed':r['passed'],'output':str(a.output or OUTPUTS[a.stage])}))
