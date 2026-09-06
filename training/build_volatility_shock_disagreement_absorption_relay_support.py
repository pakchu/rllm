"""Build outcome-blind source support for preregistered VSDAR-24."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import numpy as np,pandas as pd
from training import preregister_volatility_shock_disagreement_absorption_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_cboe_volatility_surface_regime_crossing_relay_support import strict_prior_midrank
CLOCK=Path("data/volatility_shock_disagreement_absorption_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/volatility_shock_disagreement_absorption_relay_controls_2023_2026");RESULT=Path("results/volatility_shock_disagreement_absorption_relay_support_2026-08-08.json");SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MIN={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_magnitude_gate","vix_shock_only","relative_convexity_only","same_sign_confirmation","direction_flip");NY=ZoneInfo("America/New_York");ECONOMIC_OUTCOMES_AUTHORIZED=False;COLUMNS=("candidate","control","split","observation_date","previous_observation_date","decision_time","feature_available_time","entry_time","exit_time","side","delta_log_vix","delta_relative_convexity","absolute_vix_change_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def features()->pd.DataFrame:
 if sha(prereg.SOURCE)!=prereg.SOURCE_SHA256 or sha(prereg.SOURCE_MANIFEST)!=prereg.SOURCE_MANIFEST_SHA256:raise RuntimeError("VSDAR source changed")
 f=pd.read_csv(prereg.SOURCE,compression="gzip");expected=["observation_date","SKEW_close","VVIX_close","VIX9D_close","VIX_close","VIX3M_close"]
 if f.columns.tolist()!=expected:raise RuntimeError("VSDAR schema changed")
 f.observation_date=pd.to_datetime(f.observation_date,format="%Y-%m-%d");vals=f[expected[1:]].apply(pd.to_numeric,errors="coerce")
 if not f.observation_date.is_monotonic_increasing or f.observation_date.duplicated().any() or not np.isfinite(vals).all().all() or not vals.gt(0).all().all():raise RuntimeError("VSDAR source invalid")
 v=np.log(vals.VIX_close);r=np.log(vals.VVIX_close/vals.VIX_close);f["delta_log_vix"]=v.diff();f["delta_relative_convexity"]=r.diff();f["absolute_vix_change_rank"]=strict_prior_midrank(f.delta_log_vix.abs());return f
def signal(f:pd.DataFrame,control:str)->pd.Series:
 dv=f.delta_log_vix;dr=f.delta_relative_convexity;rank=f.absolute_vix_change_rank;eligible=dv.ne(0)&dr.ne(0)&rank.notna();side=np.sign(dv).astype("Int64").fillna(0).astype(int)
 if control=="vix_shock_only":eligible=dv.ne(0)&rank.ge(.75)
 elif control=="relative_convexity_only":eligible=dr.ne(0)&rank.ge(.75);side=np.sign(dr).astype("Int64").fillna(0).astype(int)
 elif control=="same_sign_confirmation":eligible&=dv.mul(dr).gt(0)&rank.ge(.75)
 else:eligible&=dv.mul(dr).lt(0);eligible&=True if control=="no_magnitude_gate" else rank.ge(.75)
 side=side.where(eligible,0);return -side if control=="direction_flip" else side
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 s=signal(f,control);rows=[];next_allowed=None
 for i in f.index[s.ne(0)]:
  if i+1>=len(f):continue
  nd=f.at[i+1,"observation_date"];entry=(pd.Timestamp(nd.date()).tz_localize(NY)+pd.Timedelta(hours=9,minutes=35)).tz_convert("UTC");exit_=entry+pd.Timedelta(hours=24)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  od=f.at[i,"observation_date"];next_allowed=exit_;rows.append({"candidate":"VSDAR-24","control":control,"split":split,"observation_date":od.date().isoformat(),"previous_observation_date":f.at[i-1,"observation_date"].date().isoformat(),"decision_time":pd.Timestamp(od.date()).tz_localize(NY)+pd.Timedelta(hours=16),"feature_available_time":entry,"entry_time":entry,"exit_time":exit_,"side":int(s.at[i]),"delta_log_vix":float(f.at[i,"delta_log_vix"]),"delta_relative_convexity":float(f.at[i,"delta_relative_convexity"]),"absolute_vix_change_rank":float(f.at[i,"absolute_vix_change_rank"])})
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
 passed=all(checks.values());core={"protocol_version":"vsdar_24_source_support_v1","policy_id":"VSDAR-24","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(prereg.SOURCE_MANIFEST),"sha256":sha(prereg.SOURCE_MANIFEST)},"source_panel":{"path":str(prereg.SOURCE),"sha256":sha(prereg.SOURCE)},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.parent.mkdir(parents=True,exist_ok=True);RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
