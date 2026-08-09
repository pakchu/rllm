"""Build source-only support for frozen HVRDBR-6."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_high_volatility_record_discovery_breadth_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="4f7e4f9d00ea0951325faf48ae73c738a578db4fc92a527104ca55b589ce070a";START=pd.Timestamp("2023-01-01T00:00Z");END=pd.Timestamp("2026-08-01T00:00Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_discovery_share_gate","no_variation_gate","net_block_return_side","one_block_stale_features","direction_flip","same_clock_forced_long")
BAR_QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS bar_time,(array_agg(open ORDER BY ts))[1] AS bar_open,max(high) AS bar_high,min(low) AS bar_low,(array_agg(close ORDER BY ts DESC))[1] AS bar_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""
SOURCE_DIR=Path("data/high_volatility_record_discovery_breadth_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"eight_hour_discovery.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/high_volatility_record_discovery_breadth_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_record_discovery_breadth_relay_controls_2023_2026");RESULT=Path("results/high_volatility_record_discovery_breadth_relay_support_2026-08-09.json")
FEATURE_COLUMNS=("decision_time","feature_available_time","source_valid","high_discoveries","low_discoveries","breadth","discovery_share","realized_variation","block_return","discovery_share_rank","variation_rank")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side",*FEATURE_COLUMNS[3:])
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return prereg.canonical_hash(x)
def prior_midrank(v:pd.Series,lookback:int=270,minimum:int=252)->pd.Series:
 out=pd.Series(np.nan,index=v.index,dtype=float);h=[]
 for i,x in pd.to_numeric(v,errors="coerce").items():
  p=h[-lookback:]
  if math.isfinite(x) and len(p)>=minimum:
   a=np.asarray(p);out.at[i]=(np.sum(a<x)+.5*np.sum(a==x))/len(a)
  if math.isfinite(x):h.append(float(x))
 return out
def discovery_counts(high:np.ndarray,low:np.ndarray)->tuple[int,int]:
 if len(high)<2:return 0,0
 prior_high=np.maximum.accumulate(high)[:-1];prior_low=np.minimum.accumulate(low)[:-1]
 return int(np.sum(high[1:]>prior_high)),int(np.sum(low[1:]<prior_low))
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_bars()->pd.DataFrame:
 from sqlalchemy import text
 db=engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(BAR_QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:db.dispose()
def build_features(bars:pd.DataFrame)->pd.DataFrame:
 req={"bar_time","bar_open","bar_high","bar_low","bar_close","source_rows","distinct_rows","first_ts","last_ts","coherent"}
 if not req.issubset(bars):raise ValueError("HVRDBR source schema drift")
 b=bars.copy()
 for c in ("bar_time","first_ts","last_ts"):b[c]=pd.to_datetime(b[c],utc=True,errors="coerce")
 for c in ("bar_open","bar_high","bar_low","bar_close","source_rows","distinct_rows"):b[c]=pd.to_numeric(b[c],errors="coerce")
 b=b.sort_values("bar_time",kind="mergesort").set_index("bar_time");rows=[]
 for d in pd.date_range(START+pd.Timedelta(hours=8),END,freq="8h",inclusive="left"):
  idx=pd.date_range(d-pd.Timedelta(hours=8),d,freq="5min",inclusive="left");w=b.reindex(idx);expected=pd.Series(idx,index=idx)
  ok=np.isfinite(w[["bar_open","bar_high","bar_low","bar_close","source_rows","distinct_rows"]]).all(axis=1)&w[["bar_open","bar_high","bar_low","bar_close"]].gt(0).all(axis=1)&w.source_rows.eq(5)&w.distinct_rows.eq(5)&w.coherent.eq(True)&w.first_ts.eq(expected)&w.last_ts.eq(pd.Series(idx+pd.Timedelta(minutes=4),index=idx));valid=bool(len(w)==96 and ok.all())
  hi=lo=0;breadth=share=variation=block_return=np.nan
  if valid:
   hi,lo=discovery_counts(w.bar_high.to_numpy(float),w.bar_low.to_numpy(float));breadth=hi-lo;total=hi+lo;share=abs(breadth)/total if total>0 else np.nan;returns=np.log(w.bar_close.to_numpy(float)/w.bar_open.to_numpy(float));variation=float(np.sqrt(np.square(returns).sum()));block_return=float(np.log(w.bar_close.iloc[-1]/w.bar_open.iloc[0]));valid=bool(breadth!=0 and total>0 and variation>0 and block_return!=0 and np.isfinite([share,variation,block_return]).all())
  if not valid:breadth=share=variation=block_return=np.nan
  rows.append({"decision_time":d,"feature_available_time":d,"source_valid":valid,"high_discoveries":hi if valid else np.nan,"low_discoveries":lo if valid else np.nan,"breadth":breadth,"discovery_share":share,"realized_variation":variation,"block_return":block_return})
 out=pd.DataFrame(rows);out["discovery_share_rank"]=prior_midrank(out.discovery_share.where(out.source_valid));out["variation_rank"]=prior_midrank(out.realized_variation.where(out.source_valid));return out[list(FEATURE_COLUMNS)]
def conditions(f:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 u=f.shift(1) if control=="one_block_stale_features" else f;valid=u.source_valid.eq(True)&u.breadth.ne(0);dg=pd.Series(True,index=f.index) if control=="no_discovery_share_gate" else u.discovery_share_rank.ge(.65);vg=pd.Series(True,index=f.index) if control=="no_variation_gate" else u.variation_rank.ge(.65);active=valid&dg&vg;basis=u.block_return if control=="net_block_return_side" else u.breadth;side=np.sign(basis).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=f.index)
 return active,side,u
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,sides,u=conditions(f,control);rows=[]
 for i in f.index[active]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6);split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  source=u.loc[i];rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(sides.at[i]),**{c:source[c] for c in FEATURE_COLUMNS[3:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(c:pd.DataFrame,s:str)->dict[str,float|int]:
 x=c[c.split.eq(s)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());q=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVRDBR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);bars=load_bars();features=build_features(bars);primary=clock(features);controls={n:clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"hvrdbr_6_source_v1","query_sha256":hashlib.sha256(BAR_QUERY.encode()).hexdigest(),"table":"bars_binance","window":[START.isoformat(),END.isoformat()],"physical_rows_1m":int(pd.to_numeric(bars.source_rows).sum()),"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features),"valid_rows":int(features.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};sm={**sc,"manifest_hash":chash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"hvrdbr_6_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};out={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");return out
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
