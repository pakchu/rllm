"""Build source-only support clocks for frozen DVBCR-12."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_daily_variance_backloading_continuation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

HISTORY=Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb12_complete.csv.gz");TAIL_DIR=Path("data/options_oi_chase_exhaustion_sources_2023_2026");TAIL=TAIL_DIR/"btc_completed_hour.csv.gz";CLOCK=Path("data/daily_variance_backloading_continuation_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/daily_variance_backloading_continuation_relay_controls_2023_2026");RESULT=Path("results/daily_variance_backloading_continuation_relay_support_2026-08-08.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_volatility_gate","no_backloading_gate","no_late_dominance_gate","first_six_hour_direction","direction_flip");ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","daily_realized_variation","realized_variation_rank","final_12h_variance_share","final_12h_return","final_6h_return","first_6h_return")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def strict_prior_midrank(values:pd.Series,lookback:int=180,minimum:int=126)->pd.Series:
 n=pd.to_numeric(values,errors="coerce").astype(float);out=pd.Series(np.nan,index=n.index,dtype=float);history=[]
 for i,x in n.items():
  prior=history[-lookback:]
  if math.isfinite(x) and len(prior)>=minimum:
   a=np.asarray(prior);out.at[i]=(np.sum(a<x)+.5*np.sum(a==x))/len(a)
  if math.isfinite(x):history.append(x)
 return out

def hourly_sources()->pd.DataFrame:
 h=pd.read_csv(HISTORY,compression="gzip",usecols=["date","open","close"]);h["date"]=pd.to_datetime(h.date,utc=True,format="mixed");h["open"]=pd.to_numeric(h.open,errors="coerce");h["close"]=pd.to_numeric(h.close,errors="coerce");h["hour_start"]=h.date.dt.floor("h")
 g=h.groupby("hour_start",as_index=False).agg(open=("open","first"),close=("close","last"),rows=("date","size"));g["source_valid"]=g.rows.eq(12)&np.isfinite(g[["open","close"]]).all(axis=1)&g[["open","close"]].gt(0).all(axis=1);g["decision_time"]=g.hour_start+pd.Timedelta(hours=1)
 t=pd.read_csv(TAIL,compression="gzip");t["decision_time"]=pd.to_datetime(t.decision_time,utc=True,format="mixed");t["open"]=pd.to_numeric(t.open,errors="coerce");t["close"]=pd.to_numeric(t.close,errors="coerce");t["source_valid"]=t.source_valid.astype(str).str.lower().eq("true")&np.isfinite(t[["open","close"]]).all(axis=1)&t[["open","close"]].gt(0).all(axis=1)
 first=t.decision_time.min();x=pd.concat([g[g.decision_time<first][["decision_time","open","close","source_valid"]],t[["decision_time","open","close","source_valid"]]],ignore_index=True).sort_values("decision_time").drop_duplicates("decision_time",keep="last").reset_index(drop=True)
 if x.decision_time.duplicated().any() or not x.decision_time.is_monotonic_increasing:raise RuntimeError("DVBCR hourly source time drift")
 return x

def features()->pd.DataFrame:
 h=hourly_sources();h["hour_return"]=np.log(h.close/h.open);consecutive=h.decision_time.diff().eq(pd.Timedelta(hours=1));valid=h.source_valid&np.isfinite(h.hour_return)
 sq=h.hour_return.pow(2);h["daily_var"]=sq.rolling(24,min_periods=24).sum();h["final_12h_var"]=sq.rolling(12,min_periods=12).sum();h["daily_realized_variation"]=np.sqrt(h.daily_var);h["final_12h_variance_share"]=h.final_12h_var/h.daily_var;h["final_12h_return"]=h.hour_return.rolling(12,min_periods=12).sum();h["final_6h_return"]=h.hour_return.rolling(6,min_periods=6).sum();h["first_6h_return"]=h.hour_return.shift(18).rolling(6,min_periods=6).sum();h["source_valid_day"]=valid.rolling(24,min_periods=24).sum().eq(24)&consecutive.rolling(23,min_periods=23).sum().eq(23)&np.isfinite(h[["daily_realized_variation","final_12h_variance_share","final_12h_return","final_6h_return","first_6h_return"]]).all(axis=1)&h.daily_var.gt(0)
 d=h[h.decision_time.dt.hour.eq(0)&h.decision_time.dt.minute.eq(0)][["decision_time","daily_realized_variation","final_12h_variance_share","final_12h_return","final_6h_return","first_6h_return","source_valid_day"]].copy().reset_index(drop=True);d["realized_variation_rank"]=strict_prior_midrank(d.daily_realized_variation.where(d.source_valid_day));return d

def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 volatility=pd.Series(True,index=f.index) if control=="no_volatility_gate" else f.realized_variation_rank.ge(.65);backload=pd.Series(True,index=f.index) if control=="no_backloading_gate" else f.final_12h_variance_share.ge(.65);dominance=pd.Series(True,index=f.index) if control=="no_late_dominance_gate" else f.final_6h_return.abs().ge(.5*f.final_12h_return.abs());direction=f.first_6h_return if control=="first_six_hour_direction" else f.final_6h_return;active=f.source_valid_day&f.realized_variation_rank.notna()&volatility&backload&dominance&direction.ne(0)&np.isfinite(direction);return active,np.sign(direction)
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  ss=int(side.at[i]);ss=-ss if control=="direction_flip" else ss;next_allowed=exit_;rows.append({"candidate":"DVBCR-12","control":control,"split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":ss,"daily_realized_variation":float(f.at[i,"daily_realized_variation"]),"realized_variation_rank":float(f.at[i,"realized_variation_rank"]),"final_12h_variance_share":float(f.at[i,"final_12h_variance_share"]),"final_12h_return":float(f.at[i,"final_12h_return"]),"final_6h_return":float(f.at[i,"final_6h_return"]),"first_6h_return":float(f.at[i,"first_6h_return"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict[str,int|float]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 st={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in st.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());tailmanifest=TAIL_DIR/"manifest.json";passed=all(checks.values());core={"protocol_version":"dvbcr_12_source_support_v1","policy_id":"DVBCR-12","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_bindings":{"historical":{"path":str(HISTORY),"sha256":sha(HISTORY)},"completed_hour_tail":{"path":str(TAIL),"sha256":sha(TAIL)},"tail_manifest":{"path":str(tailmanifest),"sha256":sha(tailmanifest)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":st,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
