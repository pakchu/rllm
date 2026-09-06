"""Build outcome-blind source support for preregistered CVCIR-24."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import numpy as np,pandas as pd
from training import preregister_cboe_volatility_curve_impulse_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_cboe_volatility_surface_regime_crossing_relay_support import strict_prior_midrank
CLOCK=Path("data/cboe_volatility_curve_impulse_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/cboe_volatility_curve_impulse_relay_controls_2023_2026");RESULT=Path("results/cboe_volatility_curve_impulse_relay_support_2026-08-08.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MIN={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_vix_high","front_only","broad_only","no_onset","direction_flip");NY=ZoneInfo("America/New_York");ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","observation_date","previous_observation_date","decision_time","feature_available_time","entry_time","exit_time","side","front","broad","delta_front","delta_broad","vix_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def features()->pd.DataFrame:
 if sha(prereg.SOURCE)!=prereg.SOURCE_SHA256 or sha(prereg.SOURCE_MANIFEST)!=prereg.SOURCE_MANIFEST_SHA256:raise RuntimeError("CVCIR frozen source changed")
 f=pd.read_csv(prereg.SOURCE,compression="gzip");expected=["observation_date","SKEW_close","VVIX_close","VIX9D_close","VIX_close","VIX3M_close"]
 if f.columns.tolist()!=expected:raise RuntimeError("CVCIR source schema changed")
 f.observation_date=pd.to_datetime(f.observation_date,format="%Y-%m-%d")
 if not f.observation_date.is_monotonic_increasing or f.observation_date.duplicated().any():raise RuntimeError("CVCIR source dates invalid")
 for c in expected[1:]:f[c]=pd.to_numeric(f[c],errors="coerce")
 if not np.isfinite(f[expected[1:]]).all().all() or not f[expected[1:]].gt(0).all().all():raise RuntimeError("CVCIR source values invalid")
 f["front"]=np.log(f.VIX9D_close/f.VIX_close);f["broad"]=np.log(f.VIX_close/f.VIX3M_close);f["delta_front"]=f.front.diff();f["delta_broad"]=f.broad.diff();f["vix_rank"]=strict_prior_midrank(np.log(f.VIX_close));return f
def raw_side(f:pd.DataFrame,control:str)->pd.Series:
 if control=="front_only":a=f.delta_front;b=f.delta_front
 elif control=="broad_only":a=f.delta_broad;b=f.delta_broad
 else:a=f.delta_front;b=f.delta_broad
 s=pd.Series(0,index=f.index,dtype=int);s.loc[a.lt(0)&b.lt(0)]=1;s.loc[a.gt(0)&b.gt(0)]=-1;return s
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 s=raw_side(f,control);eligible=s.ne(0)&f.vix_rank.notna()
 if control!="no_vix_high":eligible&=f.vix_rank.ge(.60)
 if control!="no_onset":eligible&=s.ne(s.shift(1,fill_value=0))
 rows=[];next_allowed=None
 for i in f.index[eligible]:
  if i+1>=len(f):continue
  nd=f.at[i+1,"observation_date"];entry=(pd.Timestamp(nd.date()).tz_localize(NY)+pd.Timedelta(hours=9,minutes=35)).tz_convert("UTC");exit_=entry+pd.Timedelta(hours=24)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(start,end) in SPLITS.items() if entry>=start and exit_<=end),None)
  if split is None:continue
  side=int(s.at[i]);side=-side if control=="direction_flip" else side;next_allowed=exit_;od=f.at[i,"observation_date"]
  rows.append({"candidate":"CVCIR-24","control":control,"split":split,"observation_date":od.date().isoformat(),"previous_observation_date":f.at[i-1,"observation_date"].date().isoformat(),"decision_time":pd.Timestamp(od.date()).tz_localize(NY)+pd.Timedelta(hours=16),"feature_available_time":entry,"entry_time":entry,"exit_time":exit_,"side":side,"front":float(f.at[i,"front"]),"broad":float(f.at[i,"broad"]),"delta_front":float(f.at[i,"delta_front"]),"delta_broad":float(f.at[i,"delta_broad"]),"vix_rank":float(f.at[i,"vix_rank"])})
 r=pd.DataFrame(rows,columns=COLUMNS)
 for c in ("decision_time","feature_available_time","entry_time","exit_time"):
  if not r.empty:r[c]=pd.to_datetime(r[c],utc=True)
 return r
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MIN[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"cvcir_24_source_support_v1","policy_id":"CVCIR-24","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(prereg.SOURCE_MANIFEST),"sha256":sha(prereg.SOURCE_MANIFEST)},"source_panel":{"path":str(prereg.SOURCE),"sha256":sha(prereg.SOURCE)},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.parent.mkdir(parents=True,exist_ok=True);RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
