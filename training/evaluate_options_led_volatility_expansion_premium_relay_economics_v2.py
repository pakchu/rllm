"""Fresh V2 sequential OVEPR economics with exact half-open boundary accounting."""
from __future__ import annotations
import argparse,csv,gzip,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import evaluate_options_led_volatility_expansion_premium_relay_economics as v1

POLICY_ID=v1.POLICY_ID;BAR=v1.BAR;LEVERAGE=v1.LEVERAGE;BASE_COST=v1.BASE_COST;STRESS_COST=v1.STRESS_COST
STAGES=v1.STAGES;PREDECESSOR=v1.PREDECESSOR;CONTROLS=v1.CONTROLS;CONTROL_DIR=v1.CONTROL_DIR;CLOCK=v1.CLOCK
NOVELTY=v1.NOVELTY;NOVELTY_SHA=v1.NOVELTY_SHA;SUPPORT=v1.SUPPORT
FAILURE=Path('results/options_led_volatility_expansion_premium_relay_train_economic_attempt_terminal_failure_2026-08-08.json');FAILURE_SHA='080a2c93b8e34999a1520d1c972b033f615393a27426a4623dddf2519d6f8d17'
TRAIN_FUNDING=Path('data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz');TRAIN_FUNDING_SHA='3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6'
OUTPUTS={s:Path(f'results/options_led_volatility_expansion_premium_relay_{s}_economics_v2_2026-08-08.json') for s in STAGES}
CLUSTER_DRAWS=v1.CLUSTER_DRAWS;CLUSTER_SEED=v1.CLUSTER_SEED
sha=v1.sha;chash=v1.chash;load_json=v1.load_json;_utc=v1._utc;load_clock=v1.load_clock;validate_market=v1.validate_market

def verify(stage:str)->dict[str,Any]:
 n=v1.verify_controls('train')
 if sha(FAILURE)!=FAILURE_SHA:raise RuntimeError('V1 terminal failure receipt drift')
 failure=load_json(FAILURE)
 if failure.get('economic_metrics_computed') is not False or failure.get('attempt_disposition')!='TERMINAL_FAILURE_NO_RETRY_UNDER_V1':raise RuntimeError('V1 failure is not eligible for infrastructure successor')
 if stage in PREDECESSOR:
  p=OUTPUTS[PREDECESSOR[stage]]
  if not p.is_file():raise RuntimeError(f'missing V2 predecessor: {p}')
  d=load_json(p);core={k:v for k,v in d.items() if k!='manifest_hash'}
  if d.get('manifest_hash')!=chash(core) or d.get('passed') is not True:raise RuntimeError('V2 predecessor did not pass')
 return n

def _stream(path:Path,start:pd.Timestamp,end:pd.Timestamp,columns:tuple[str,...],date_col:str,include_end:bool)->pd.DataFrame:
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
 out[date_col]=pd.to_datetime(out[date_col],utc=True,errors='raise');return out

def load_train_funding(start:pd.Timestamp,end:pd.Timestamp)->pd.DataFrame:
 if sha(TRAIN_FUNDING)!=TRAIN_FUNDING_SHA:raise RuntimeError('train funding mark source hash drift')
 rows=[]
 with gzip.open(TRAIN_FUNDING,'rt',newline='') as f:
  reader=csv.DictReader(f);required={'funding_time_utc','funding_rate','settlement_mark_price'}
  if not required.issubset(reader.fieldnames or []):raise RuntimeError('train funding mark schema drift')
  for row in reader:
   t=_utc(row['funding_time_utc'])
   if t>=end:break
   if t>=start:rows.append({'date':t,'funding_rate':row['funding_rate'],'mark_price':row['settlement_mark_price']})
 out=pd.DataFrame(rows,columns=['date','funding_rate','mark_price'])
 if out.empty:raise RuntimeError('empty train funding mark prefix')
 for c in ('funding_rate','mark_price'):out[c]=pd.to_numeric(out[c],errors='raise')
 return out

def load_csv_market(start:pd.Timestamp,end:pd.Timestamp)->pd.DataFrame:
 if sha(v1.MARKET)!=v1.MARKET_SHA:raise RuntimeError('economic market source hash drift')
 m=_stream(v1.MARKET,start,end,('date','open','high','low','close'),'date',True)
 for c in ('open','high','low','close'):m[c]=pd.to_numeric(m[c],errors='raise')
 return m

def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file('.env');return create_engine(postgres_url_from_env('.env'),connect_args={'connect_timeout':10})

def load_postgres_funding(start:pd.Timestamp,end:pd.Timestamp)->pd.DataFrame:
 from sqlalchemy import text
 engine=postgres_engine();funds=text("""SELECT funding_time AS date,funding_rate,mark_price FROM funding_rates_binance WHERE symbol=:symbol AND funding_time>=:start AND funding_time<:end ORDER BY funding_time""")
 with engine.connect() as c:f=pd.read_sql_query(funds,c,params={'symbol':'BTCUSDT','start':start.to_pydatetime(),'end':end.to_pydatetime()})
 engine.dispose();f.date=pd.to_datetime(f.date,utc=True);return f[['date','funding_rate','mark_price']]

def load_postgres(start:pd.Timestamp,end:pd.Timestamp):
 from sqlalchemy import text
 engine=postgres_engine()
 bars=text("""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,(array_agg(close ORDER BY ts DESC))[1] AS close,count(*) AS source_rows FROM bars_binance WHERE interval='1m' AND symbol=:symbol AND ts>=:start AND ts<:query_end GROUP BY 1 ORDER BY 1""")
 funds=text("""SELECT funding_time AS date,funding_rate,mark_price FROM funding_rates_binance WHERE symbol=:symbol AND funding_time>=:start AND funding_time<:end ORDER BY funding_time""")
 with engine.connect() as c:
  m=pd.read_sql_query(bars,c,params={'symbol':'BTCUSDT','start':start.to_pydatetime(),'query_end':(end+BAR).to_pydatetime()});f=pd.read_sql_query(funds,c,params={'symbol':'BTCUSDT','start':start.to_pydatetime(),'end':end.to_pydatetime()})
 engine.dispose();m.date=pd.to_datetime(m.date,utc=True);f.date=pd.to_datetime(f.date,utc=True)
 if not m.source_rows.eq(5).all():raise RuntimeError('Postgres 5m source incomplete')
 return m[['date','open','high','low','close']],f[['date','funding_rate','mark_price']],{'mode':'postgres_exact_1m_to_5m','tables':['bars_binance','funding_rates_binance'],'symbol':'BTCUSDT'}
def load_sources(stage,start,end):
 if stage=='final':return load_postgres(start,end)
 m=load_csv_market(start,end)
 if stage=='train':
  f=load_train_funding(start,end);source={'mode':'hash_bound_gzip_physical_prefix','market_sha256':v1.MARKET_SHA,'funding_marks_sha256':TRAIN_FUNDING_SHA}
 else:
  f=load_postgres_funding(start,end);source={'mode':'hash_bound_gzip_market_plus_postgres_exact_funding','market_sha256':v1.MARKET_SHA,'funding_table':'funding_rates_binance','symbol':'BTCUSDT'}
 return m,f,source

def validate_funding(f:pd.DataFrame,start:pd.Timestamp,end:pd.Timestamp)->None:
 if list(f.columns)!=['date','funding_rate','mark_price']:raise RuntimeError('funding schema drift')
 if f.empty:raise RuntimeError('funding source is empty')
 if f.date.duplicated().any() or not f.date.is_monotonic_increasing:raise RuntimeError('funding clock invalid')
 vals=f[['funding_rate','mark_price']].to_numpy(float)
 if not np.isfinite(vals).all() or (f.mark_price<=0).any() or not ((f.date>=start)&(f.date<end)).all():raise RuntimeError('funding values invalid')
 if f.date.iloc[0]!=start or f.date.iloc[-1]<end-pd.Timedelta(hours=8):raise RuntimeError('funding boundary incomplete')
 if len(f)>1 and f.date.diff().iloc[1:].max()>pd.Timedelta(hours=8):raise RuntimeError('funding gap exceeds eight hours')

def simulate(clock:pd.DataFrame,m:pd.DataFrame,f:pd.DataFrame,start:pd.Timestamp,end:pd.Timestamp,cost:float)->dict[str,Any]:
 dates=pd.DatetimeIndex(m.date);positions={t:i for i,t in enumerate(dates)};opens=m.open.to_numpy(float);highs=m.high.to_numpy(float);lows=m.low.to_numpy(float)
 clock=clock.sort_values('entry_time').reset_index(drop=True);previous=None
 equity=peak=1.0;mdd=0.0;trade_rows=[]
 for r in clock.itertuples(index=False):
  if previous is not None and r.entry_time<previous:raise RuntimeError('clock intervals overlap')
  previous=r.exit_time
  if r.entry_time not in positions or r.exit_time not in positions:raise RuntimeError('clock absent from exact market opens')
  a=positions[r.entry_time];b=positions[r.exit_time];side=int(r.side);entry=float(opens[a]);pre=float(equity);quantity=side*LEVERAGE*pre/entry;entry_fee=abs(quantity)*entry*cost;funding_cash=0.0
  stage_funding=f[(f.date>=r.entry_time)&(f.date<r.exit_time)];fund_by_time={t:float(-quantity*mp*rate) for t,rate,mp in zip(stage_funding.date,stage_funding.funding_rate,stage_funding.mark_price)}
  for j in range(a,b):
   funding_cash+=fund_by_time.get(dates[j],0.0);cash=pre-entry_fee+funding_cash
   favorable=float(highs[j] if side>0 else lows[j]);adverse=float(lows[j] if side>0 else highs[j])
   favorable_equity=cash+quantity*(favorable-entry);adverse_equity=cash+quantity*(adverse-entry)-abs(quantity)*adverse*cost
   if min(favorable_equity,adverse_equity)<=0:raise RuntimeError('nonpositive intratrade equity')
   peak=max(peak,favorable_equity);mdd=max(mdd,1-adverse_equity/peak)
  exit_price=float(opens[b]);exit_fee=abs(quantity)*exit_price*cost;equity=pre-entry_fee+quantity*(exit_price-entry)+funding_cash-exit_fee
  if equity<=0:raise RuntimeError('nonpositive realized equity')
  mdd=max(mdd,1-equity/peak);peak=max(peak,equity)
  gross=side*(exit_price/entry-1);trade_rows.append({'entry_time':r.entry_time,'exit_time':r.exit_time,'side':side,'gross_return':gross,'net_factor':equity/pre,'funding_cash_over_pre_equity':funding_cash/pre})
 years=(end-start).total_seconds()/(365.25*86400);absolute=(equity-1)*100;cagr=(equity**(1/years)-1)*100;mdd_pct=mdd*100;gross=[x['gross_return'] for x in trade_rows];net=[x['net_factor']-1 for x in trade_rows]
 return {'absolute_return_pct':absolute,'cagr_pct':cagr,'strict_mdd_pct':mdd_pct,'cagr_to_strict_mdd':cagr/mdd_pct if mdd_pct>1e-12 else 0.0,'final_equity':equity,'trades':len(trade_rows),'longs':sum(x['side']>0 for x in trade_rows),'shorts':sum(x['side']<0 for x in trade_rows),'mean_gross_underlying_bp':float(np.mean(gross)*1e4) if gross else 0.0,'mean_net_bp':float(np.mean(net)*1e4) if net else 0.0,'win_rate':float(np.mean(np.asarray(net)>0)) if net else 0.0,'trade_rows':trade_rows}

def cluster_p(rows:list[dict[str,Any]])->dict[str,Any]:
 clusters={}
 for r in rows:
  iso=r['entry_time'].isocalendar();key=(int(iso.year),int(iso.week));clusters[key]=clusters.get(key,0)+math.log(r['net_factor'])
 vals=np.array(list(clusters.values()));obs=float(vals.sum());rng=np.random.default_rng(CLUSTER_SEED);null=rng.choice(np.array([-1.,1.]),size=(CLUSTER_DRAWS,len(vals)))@vals;p=(1+int((null>=obs).sum()))/(CLUSTER_DRAWS+1)
 return {'method':'one_sided_UTC_week_cluster_signflip_monte_carlo','clusters':len(vals),'draws':CLUSTER_DRAWS,'seed':CLUSTER_SEED,'observed_log_effect':obs,'pvalue':p}
def public_metric(d):return {k:v for k,v in d.items() if k!='trade_rows'}
def evaluate_primary(clock,m,f,start,end):
 base=simulate(clock,m,f,start,end,BASE_COST);stress=simulate(clock,m,f,start,end,STRESS_COST);mid=start+(end-start)/2;halves={}
 for name,a,b in [('first',start,mid),('second',mid,end)]:halves[name]=public_metric(simulate(clock[(clock.entry_time>=a)&(clock.exit_time<=b)],m,f,a,b,BASE_COST))
 return {'base':public_metric(base),'stress':public_metric(stress),'cluster_signflip':cluster_p(base['trade_rows']),'calendar_halves':halves}
def evaluate_control(clock,m,f,start,end):return {'base':public_metric(simulate(clock,m,f,start,end,BASE_COST)),'stress':public_metric(simulate(clock,m,f,start,end,STRESS_COST))}

def run(stage:str,output:Path|None=None)->dict[str,Any]:
 novelty=verify(stage);split,s0,e0=STAGES[stage];start=_utc(s0);end=_utc(e0);m,f,source=load_sources(stage,start,end);validate_market(m,start,end);validate_funding(f,start,end);primary_clock=load_clock(CLOCK,split,start,end);primary=evaluate_primary(primary_clock,m,f,start,end);support=load_json(SUPPORT);controls={}
 for name in CONTROLS:
  p=CONTROL_DIR/f'{name}.csv.gz'
  if sha(p)!=support['controls'][name]['sha256']:raise RuntimeError(f'control hash drift: {name}')
  controls[name]=evaluate_control(load_clock(p,split,start,end),m,f,start,end)
 b=primary['base'];s=primary['stress'];checks={'absolute_return_positive':b['absolute_return_pct']>0,'cagr_to_strict_mdd_min_3':b['cagr_to_strict_mdd']>=3,'strict_mdd_max_15':b['strict_mdd_pct']<=15,'mean_gross_move_min_20bp':b['mean_gross_underlying_bp']>=20,'cluster_signflip_p_max_0_1':primary['cluster_signflip']['pvalue']<=.1,'stress_absolute_return_positive':s['absolute_return_pct']>0,'stress_cagr_to_strict_mdd_min_2_5':s['cagr_to_strict_mdd']>=2.5,'each_calendar_half_positive':all(x['absolute_return_pct']>0 for x in primary['calendar_halves'].values())};passed=all(checks.values())
 core={'protocol_version':'ovepr_24_sequential_economics_v2','policy_id':POLICY_ID,'stage':stage,'window':[s0,e0],'v1_terminal_failure':{'path':str(FAILURE),'sha256':FAILURE_SHA,'candidate_metrics_observed':False},'predecessor':None if stage=='train' else {'stage':PREDECESSOR[stage],'path':str(OUTPUTS[PREDECESSOR[stage]]),'sha256':sha(OUTPUTS[PREDECESSOR[stage]])},'novelty_authorization':{'path':str(NOVELTY),'sha256':NOVELTY_SHA,'manifest_hash':novelty['manifest_hash']},'accounting':{'quantity':'side*0.5*pre_entry_equity/entry_open, fixed through exit','same_open_transition':'exit and exit cost first, then next entry and entry cost','funding':'cash=-fixed_quantity*settlement_mark*rate for entry<=time<exit','strict_mdd':'global peak, every held favorable then adverse OHLC, funding cash, virtual adverse exit cost, actual exit cost'},'source':source,'physical_rows_opened':{'market':len(m),'funding':len(f),'primary_clock':len(primary_clock)},'later_stage_outcomes_opened':False,'primary':primary,'controls_diagnostic_only':controls,'checks':checks,'passed':passed,'advance_to_next_stage':passed and stage!='final','decision':'pass' if passed else 'terminal_reject_no_repair'};out={**core,'manifest_hash':chash(core)};dest=output or OUTPUTS[stage];dest.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n');return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=tuple(STAGES),required=True);p.add_argument('--output',type=Path);a=p.parse_args();r=run(a.stage,a.output);print(json.dumps({'stage':a.stage,'passed':r['passed'],'output':str(a.output or OUTPUTS[a.stage])}))
