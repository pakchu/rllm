"""Deterministic outcome-blind source support for HVCASPCPQA-8."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_caspc_quiet_premium_activity as prereg

START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA="d3fc1e0d152abd2c87a568efa6c2236389bf787de2271c847713ab99c2783e01";REG=prereg.build()
STAGES={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"]
PREMIUM_QUERY="""WITH tagged AS (SELECT ts,date_bin('8 hours',ts-INTERVAL '3 hours',TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '11 hours' AS decision_time,close FROM bars_binance_premium WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end), path AS (SELECT *,lag(close) OVER(PARTITION BY decision_time ORDER BY ts) AS previous_close FROM tagged) SELECT decision_time,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,sum(abs(close-previous_close)) FILTER(WHERE previous_close IS NOT NULL) AS premium_total_variation,count(previous_close) AS adjacent_pairs,bool_and(close IS NOT NULL) AS coherent FROM path GROUP BY decision_time ORDER BY decision_time"""
BTC_QUERY="""SELECT date_bin('8 hours',ts-INTERVAL '3 hours',TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '11 hours' AS decision_time,(array_agg(open ORDER BY ts))[1] AS block_open,(array_agg(close ORDER BY ts DESC))[1] AS block_close,sum(power(ln(close/open),2)) AS squared_return_sum,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY decision_time ORDER BY decision_time"""
ROOT=Path("data/high_volatility_caspc_quiet_premium_activity_sources_2023_2026");PANEL=ROOT/"cycle_states.csv.gz";MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_caspc_quiet_premium_activity_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_caspc_quiet_premium_activity_split_clocks_2023_2026")
RESULT=Path("results/high_volatility_caspc_quiet_premium_activity_support_2026-08-16.json");BUILDER=Path(__file__).relative_to(Path.cwd())
BASE_CLOCK=Path("data/high_volatility_cross_alt_serial_persistence_consensus_relay_clocks_2023_2026.csv.gz")
PANEL_COLUMNS=("decision_time","feature_available_time","source_valid","premium_total_variation","btc_return","btc_realized_variation","relative_premium_activity","relative_activity_rank","btc_variation_rank","eligible","side")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","premium_total_variation","btc_return","btc_realized_variation","relative_premium_activity","relative_activity_rank","btc_variation_rank")
def sha(p:str|Path)->str:
 d=hashlib.sha256();
 with Path(p).open("rb") as h:
  for c in iter(lambda:h.read(1048576),b""):d.update(c)
 return d.hexdigest()
def rank(s:pd.Series,history_cycles:int=270,minimum_history_cycles:int=180)->pd.Series:
 vals=pd.to_numeric(s,errors="coerce").to_numpy(float);out=np.full(len(vals),np.nan);hist=[]
 for i,v in enumerate(vals):
  prior=np.asarray(hist[-history_cycles:],float)
  if math.isfinite(v) and len(prior)>=minimum_history_cycles:out[i]=(np.sum(prior<v)+.5*np.sum(prior==v))/len(prior)
  if math.isfinite(v):hist.append(float(v))
 return pd.Series(out,index=s.index)
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file("/home/pakchu/rllm/.env");return create_engine(postgres_url_from_env("/home/pakchu/rllm/.env"),connect_args={"connect_timeout":10})
def load_source():
 from sqlalchemy import text
 db=engine()
 try:
  with db.connect() as c:
   p=pd.read_sql_query(text(PREMIUM_QUERY),c,params={"start":START,"end":END});b=pd.read_sql_query(text(BTC_QUERY),c,params={"start":START,"end":END})
 finally:db.dispose()
 return p,b
def build_panel(raw):
 p,b=raw
 ep=["decision_time","source_rows","distinct_rows","first_ts","last_ts","premium_total_variation","adjacent_pairs","coherent"]
 eb=["decision_time","block_open","block_close","squared_return_sum","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if p.columns.tolist()!=ep or b.columns.tolist()!=eb:raise RuntimeError("HVCASPCPQA-8 schema drift")
 for f in (p,b):
  for c in ("decision_time","first_ts","last_ts"):f[c]=pd.to_datetime(f[c],utc=True,errors="raise")
 p=p.rename(columns={"source_rows":"premium_rows","distinct_rows":"premium_distinct","first_ts":"premium_first","last_ts":"premium_last","coherent":"premium_coherent"})
 b=b.rename(columns={"source_rows":"btc_rows","distinct_rows":"btc_distinct","first_ts":"btc_first","last_ts":"btc_last","coherent":"btc_coherent"})
 f=p.merge(b,on="decision_time",validate="one_to_one").sort_values("decision_time").reset_index(drop=True)
 for c in ("premium_total_variation","block_open","block_close","squared_return_sum","premium_rows","premium_distinct","adjacent_pairs","btc_rows","btc_distinct"):f[c]=pd.to_numeric(f[c],errors="coerce")
 start=f.decision_time-pd.Timedelta(hours=8);last=f.decision_time-pd.Timedelta(minutes=1)
 f["source_valid"]=(f.premium_rows.eq(480)&f.premium_distinct.eq(480)&f.adjacent_pairs.eq(479)&f.premium_first.eq(start)&f.premium_last.eq(last)&f.premium_coherent.eq(True)&f.btc_rows.eq(480)&f.btc_distinct.eq(480)&f.btc_first.eq(start)&f.btc_last.eq(last)&f.btc_coherent.eq(True)&np.isfinite(f[["premium_total_variation","block_open","block_close","squared_return_sum"]]).all(axis=1)&f.premium_total_variation.gt(0)&f.block_open.gt(0)&f.block_close.gt(0)&f.squared_return_sum.gt(0))
 f["btc_return"]=np.log(f.block_close/f.block_open).where(f.source_valid);f["btc_realized_variation"]=np.sqrt(f.squared_return_sum).where(f.source_valid)
 f["relative_premium_activity"]=(f.premium_total_variation/f.btc_realized_variation).where(f.source_valid)
 f["relative_activity_rank"]=rank(f.relative_premium_activity.where(f.source_valid));f["btc_variation_rank"]=rank(f.btc_realized_variation.where(f.source_valid))
 f["eligible"]=f.source_valid&f.relative_activity_rank.notna()
 f["side"]=0;f["feature_available_time"]=f.decision_time
 return f.loc[:,PANEL_COLUMNS]
def stage(e,x):return next((n for n,(a,b) in STAGES.items() if a<=e and x<=b),None)
def build_clock(p):
 states=p.set_index("decision_time");base=pd.read_csv(BASE_CLOCK)
 if "control" in base:base=base[base.control.eq("primary")].copy()
 for c in ("decision_time","feature_available_time","entry_time","exit_time"):base[c]=pd.to_datetime(base[c],utc=True)
 rows=[]
 for b in base.itertuples(index=False):
  d=pd.Timestamp(b.decision_time)
  if d not in states.index:continue
  r=states.loc[d]
  if isinstance(r,pd.DataFrame):raise RuntimeError("duplicate premium state")
  if not bool(r.eligible):continue
  if r.relative_activity_rank>.50:continue
  side=int(b.side)
  if side not in (-1,1) or r.feature_available_time>b.entry_time:raise RuntimeError("HVCASPCPQA-8 side/availability drift")
  rows.append({"candidate":prereg.POLICY_ID,"control":"primary","split":b.split,"decision_time":d,"feature_available_time":r.feature_available_time,"entry_time":b.entry_time,"exit_time":b.exit_time,"side":side,**{c:r[c] for c in PANEL_COLUMNS[3:9]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(c,s):
 q=c[c.split.eq(s)]
 if q.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(q.side.eq(1).sum());h=int(q.side.eq(-1).sum());m=q.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(q),"longs":l,"shorts":h,"minority_side_share":min(l,h)/len(q),"max_month_share":int(m.max())/len(q)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCASPCPQA-8 preregistration drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);raw=load_source();p=build_panel(raw);c=build_clock(p);splits={n:c[c.split.eq(n)].copy() for n in STAGES}
 common.immutable(PANEL,common.csv_gz(p));common.immutable(CLOCK,common.csv_gz(c))
 for n,v in splits.items():common.immutable(SPLIT_DIR/f"{n}.csv.gz",common.csv_gz(v))
 qs={"premium":PREMIUM_QUERY,"btc":BTC_QUERY};sc={"protocol_version":"hvcaspcpqa_8_sources_v1","queries":qs,"query_sha256":{k:hashlib.sha256(v.encode()).hexdigest() for k,v in qs.items()},"tables":["bars_binance_premium","bars_binance"],"window":[START.isoformat(),END.isoformat()],"physical_rows":{"premium_blocks":len(raw[0]),"btc_blocks":len(raw[1])},"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(p),"valid_rows":int(p.source_valid.sum())},"outcomes_opened":False,"execution_prices_opened":False,"held_interval_funding_opened":False,"gross9_rows_opened":False,"no_imputation":True};m={**sc,"manifest_hash":prereg.canonical_hash(sc)};common.immutable(MANIFEST,common.json_bytes(m))
 su={n:stats(c,n) for n in STAGES};checks={k:o for n,v in su.items() for k,o in ((f"{n}_minimum_events",v["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",v["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",v["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values())
 core={"protocol_version":"hvcaspcpqa_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":m["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"held_interval_funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(c)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":len(v)} for n,v in splits.items()},"support":su,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_gross9_novelty" if passed else "terminal_source_support_reject"};res={**core,"manifest_hash":prereg.canonical_hash(core)};common.immutable(RESULT,common.json_bytes(res));return res
if __name__=="__main__":
 r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
