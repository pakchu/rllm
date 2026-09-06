"""Build source-only support clocks for frozen CCBRR-12."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import numpy as np,pandas as pd
from training import preregister_cboe_convexity_beta_residual_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

CLOCK=Path("data/cboe_convexity_beta_residual_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/cboe_convexity_beta_residual_relay_controls_2023_2026");RESULT=Path("results/cboe_convexity_beta_residual_relay_support_2026-08-08.json");SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("raw_vvix_change","fixed_beta_one","no_magnitude_gate","one_session_stale_residual","direction_flip");NY=ZoneInfo("America/New_York");ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","observation_date","previous_observation_date","decision_time","feature_available_time","entry_time","exit_time","side","delta_log_vix","delta_log_vvix","rolling_intercept","rolling_beta","standardized_residual")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def causal_residual(x:pd.Series,y:pd.Series,lookback:int=252,minimum:int=126)->pd.DataFrame:
 xv=pd.to_numeric(x,errors="coerce").to_numpy(float);yv=pd.to_numeric(y,errors="coerce").to_numpy(float);a=np.full(len(xv),np.nan);b=a.copy();z=a.copy()
 for i in range(len(xv)):
  if not np.isfinite(xv[i]) or not np.isfinite(yv[i]):continue
  px=xv[max(0,i-lookback):i];py=yv[max(0,i-lookback):i];valid=np.isfinite(px)&np.isfinite(py);px=px[valid];py=py[valid]
  if len(px)<minimum:continue
  design=np.column_stack([np.ones(len(px)),px]);coef=np.linalg.lstsq(design,py,rcond=None)[0];resid=py-design@coef;scale=float(np.std(resid,ddof=1))
  if not np.isfinite(scale) or scale<=0:continue
  a[i]=coef[0];b[i]=coef[1];z[i]=(yv[i]-coef[0]-coef[1]*xv[i])/scale
 return pd.DataFrame({"rolling_intercept":a,"rolling_beta":b,"standardized_residual":z},index=x.index)
def causal_z(v:pd.Series,lookback:int=252,minimum:int=126)->pd.Series:
 x=pd.to_numeric(v,errors="coerce").to_numpy(float);o=np.full(len(x),np.nan)
 for i,current in enumerate(x):
  prior=x[max(0,i-lookback):i];prior=prior[np.isfinite(prior)]
  if np.isfinite(current) and len(prior)>=minimum:
   s=float(np.std(prior,ddof=1));o[i]=(current-float(np.mean(prior)))/s if s>0 else np.nan
 return pd.Series(o,index=v.index)
def features()->pd.DataFrame:
 if sha(prereg.SOURCE)!=prereg.SOURCE_SHA256 or sha(prereg.SOURCE_MANIFEST)!=prereg.SOURCE_MANIFEST_SHA256:raise RuntimeError("CCBRR source drift")
 f=pd.read_csv(prereg.SOURCE,compression="gzip");expected=["observation_date","SKEW_close","VVIX_close","VIX9D_close","VIX_close","VIX3M_close"]
 if f.columns.tolist()!=expected:raise RuntimeError("CCBRR schema drift")
 f.observation_date=pd.to_datetime(f.observation_date,format="%Y-%m-%d");vals=f[expected[1:]].apply(pd.to_numeric,errors="coerce")
 if not f.observation_date.is_monotonic_increasing or f.observation_date.duplicated().any() or not np.isfinite(vals).all().all() or not vals.gt(0).all().all():raise RuntimeError("CCBRR source invalid")
 f["delta_log_vix"]=np.log(vals.VIX_close).diff();f["delta_log_vvix"]=np.log(vals.VVIX_close).diff();f=pd.concat([f,causal_residual(f.delta_log_vix,f.delta_log_vvix)],axis=1);f["raw_vvix_z"]=causal_z(f.delta_log_vvix);f["fixed_beta_one_z"]=causal_z(f.delta_log_vvix-f.delta_log_vix);return f
def signal(f:pd.DataFrame,control:str)->pd.Series:
 z=f.standardized_residual
 if control=="raw_vvix_change":z=f.raw_vvix_z
 elif control=="fixed_beta_one":z=f.fixed_beta_one_z
 elif control=="one_session_stale_residual":z=f.standardized_residual.shift(1)
 eligible=np.isfinite(z)&z.ne(0);eligible&=True if control=="no_magnitude_gate" else z.abs().ge(1.);side=-np.sign(z).astype("Int64").fillna(0).astype(int);side=side.where(eligible,0);return -side if control=="direction_flip" else side
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 s=signal(f,control);rows=[];next_allowed=None
 for i in f.index[s.ne(0)]:
  if i+1>=len(f):continue
  nd=f.at[i+1,"observation_date"];entry=(pd.Timestamp(nd.date()).tz_localize(NY)+pd.Timedelta(hours=9,minutes=35)).tz_convert("UTC");exit_=entry+pd.Timedelta(hours=12)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  od=f.at[i,"observation_date"];next_allowed=exit_;rows.append({"candidate":"CCBRR-12","control":control,"split":split,"observation_date":od.date().isoformat(),"previous_observation_date":f.at[i-1,"observation_date"].date().isoformat(),"decision_time":pd.Timestamp(od.date()).tz_localize(NY)+pd.Timedelta(hours=16),"feature_available_time":entry,"entry_time":entry,"exit_time":exit_,"side":int(s.at[i]),"delta_log_vix":float(f.at[i,"delta_log_vix"]),"delta_log_vvix":float(f.at[i,"delta_log_vvix"]),"rolling_intercept":float(f.at[i,"rolling_intercept"]),"rolling_beta":float(f.at[i,"rolling_beta"]),"standardized_residual":float(f.at[i,"standardized_residual"])})
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
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"ccbrr_12_source_support_v1","policy_id":"CCBRR-12","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(prereg.SOURCE_MANIFEST),"sha256":sha(prereg.SOURCE_MANIFEST)},"source_panel":{"path":str(prereg.SOURCE),"sha256":sha(prereg.SOURCE)},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
