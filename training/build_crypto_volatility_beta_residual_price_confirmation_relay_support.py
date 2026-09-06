"""Build source-only support clocks for frozen CVBRPCR-12."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_options_crowding_deleveraging_relay_support_v4 as volbase
from training import build_cboe_convexity_beta_residual_relay_support as residualbase
from training import preregister_crypto_volatility_beta_residual_price_confirmation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PRICE_DIR=Path("data/options_oi_chase_exhaustion_sources_2023_2026");CLOCK=Path("data/crypto_volatility_beta_residual_price_confirmation_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/crypto_volatility_beta_residual_price_confirmation_relay_controls_2023_2026");RESULT=Path("results/crypto_volatility_beta_residual_price_confirmation_relay_support_2026-08-08.json")
SPLITS=volbase.SPLITS;MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("raw_dvol_change","fixed_beta_one","no_residual_gate","residual_direction","direction_flip");ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","delta_log_bvol","delta_log_dvol","rolling_intercept","rolling_beta","standardized_residual","price_return_24h")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def features()->pd.DataFrame:
 b,d,oi,funding=volbase.load_sources();bf=pd.DataFrame({"decision_time":pd.to_datetime(b.feature_available_time_utc,utc=True,format="mixed"),"bvol_close":pd.to_numeric(b.close,errors="coerce"),"bvol_valid":b.feature_valid.astype(str).str.lower().eq("true")});df=pd.DataFrame({"decision_time":pd.to_datetime(d.close_time,utc=True,format="mixed"),"dvol_close":pd.to_numeric(d.close,errors="coerce")});v=bf.merge(df,on="decision_time",validate="one_to_one").sort_values("decision_time").reset_index(drop=True);v["vol_valid"]=v.bvol_valid&np.isfinite(v[["bvol_close","dvol_close"]]).all(axis=1)&v[["bvol_close","dvol_close"]].gt(0).all(axis=1);v=v[v.decision_time.dt.hour.eq(0)&v.decision_time.dt.minute.eq(0)].copy().reset_index(drop=True);v["delta_log_bvol"]=np.log(v.bvol_close).diff();v["delta_log_dvol"]=np.log(v.dvol_close).diff();v=pd.concat([v,residualbase.causal_residual(v.delta_log_bvol,v.delta_log_dvol,lookback=90,minimum=60)],axis=1);v["raw_dvol_z"]=residualbase.causal_z(v.delta_log_dvol,lookback=90,minimum=60);v["fixed_beta_one_z"]=residualbase.causal_z(v.delta_log_dvol-v.delta_log_bvol,lookback=90,minimum=60)
 p=pd.read_csv(PRICE_DIR/"btc_completed_hour.csv.gz",compression="gzip");p["decision_time"]=pd.to_datetime(p.decision_time,utc=True,format="mixed");p["open"]=pd.to_numeric(p.open,errors="coerce");p["close"]=pd.to_numeric(p.close,errors="coerce");p["price_valid"]=p.source_valid.astype(str).str.lower().eq("true")&np.isfinite(p[["open","close"]]).all(axis=1)&p[["open","close"]].gt(0).all(axis=1);p=p.sort_values("decision_time").reset_index(drop=True);consecutive=p.decision_time.diff().eq(pd.Timedelta(hours=1));p["price_return_24h"]=np.log(p.close/p.open.shift(23));p["price_valid_day"]=p.price_valid.rolling(24,min_periods=24).sum().eq(24)&consecutive.rolling(23,min_periods=23).sum().eq(23)&np.isfinite(p.price_return_24h)&p.price_return_24h.ne(0);p=p[p.decision_time.dt.hour.eq(0)&p.decision_time.dt.minute.eq(0)][["decision_time","price_return_24h","price_valid_day"]]
 f=v.merge(p,on="decision_time",how="inner",validate="one_to_one");f["source_valid"]=f.vol_valid&f.price_valid_day&np.isfinite(f[["delta_log_bvol","delta_log_dvol","standardized_residual","price_return_24h"]]).all(axis=1);return f
def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 z=f.standardized_residual
 if control=="raw_dvol_change":z=f.raw_dvol_z
 elif control=="fixed_beta_one":z=f.fixed_beta_one_z
 eligible=f.source_valid&np.isfinite(z)&z.ne(0);eligible&=True if control=="no_residual_gate" else z.abs().ge(1.);direction=-np.sign(z) if control=="residual_direction" else np.sign(f.price_return_24h);return eligible,direction
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  ss=int(side.at[i]);ss=-ss if control=="direction_flip" else ss;next_allowed=exit_;rows.append({"candidate":"CVBRPCR-12","control":control,"split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":ss,"delta_log_bvol":float(f.at[i,"delta_log_bvol"]),"delta_log_dvol":float(f.at[i,"delta_log_dvol"]),"rolling_intercept":float(f.at[i,"rolling_intercept"]),"rolling_beta":float(f.at[i,"rolling_beta"]),"standardized_residual":float(f.at[i,"standardized_residual"]),"price_return_24h":float(f.at[i,"price_return_24h"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 volmanifest=volbase.SOURCE_DIR/"manifest.json";pricemanifest=PRICE_DIR/"manifest.json";passed=all(checks.values());core={"protocol_version":"cvbrpcr_12_source_support_v1","policy_id":"CVBRPCR-12","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifests":{"volatility":{"path":str(volmanifest),"sha256":sha(volmanifest)},"completed_price":{"path":str(pricemanifest),"sha256":sha(pricemanifest)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
