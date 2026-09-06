"""Build source-only HVDVVR-12 clocks before Gross9 or economics."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_high_volatility_dvol_variation_risk_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market
PREREG_SHA="3bd6bafad4aebcb0b97dee8dc40844999d4f55a9bd37bdc0cde0c0fd223da47d";HELPER=Path("training/build_scheduled_trend_concordance_relay_support.py");HELPER_SHA="8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f";END=pd.Timestamp("2026-08-01T00:00:00Z")
STATE=Path("data/high_volatility_dvol_variation_risk_relay_sources_2023_2026/hourly_states.csv.gz");CLOCK=Path("data/high_volatility_dvol_variation_risk_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_dvol_variation_risk_relay_controls_2023_2026");RESULT=Path("results/high_volatility_dvol_variation_risk_relay_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_dvol_variation_gate","no_btc_variation_gate","dvol_direction","one_day_stale_dvol","direction_flip","same_clock_forced_long")
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","dvol_change","dvol_variation","dvol_variation_rank","btc_variation","btc_variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(values:pd.Series)->pd.Series:
 x=pd.to_numeric(values,errors="coerce");out=pd.Series(np.nan,index=x.index,dtype=float);history=[]
 for i,current in x.items():
  prior=np.asarray(history[-720:],float)
  if np.isfinite(current) and len(prior)>=336:out.at[i]=((prior<current).sum()+.5*(prior==current).sum())/len(prior)
  if np.isfinite(current):history.append(float(current))
 return out
def score_states(dvol:pd.DataFrame,market:pd.DataFrame)->pd.DataFrame:
 d=dvol.copy();d["date"]=pd.to_datetime(d.date,utc=True);d["close_time"]=pd.to_datetime(d.close_time,utc=True)
 if d.date.duplicated().any():raise RuntimeError("HVDVVR duplicate DVOL hour")
 for c in ("open","high","low","close"):d[c]=pd.to_numeric(d[c],errors="coerce")
 exact=d.date.diff().eq(pd.Timedelta(hours=1));coherent=np.isfinite(d[["open","high","low","close"]]).all(axis=1)&d[["open","high","low","close"]].gt(0).all(axis=1)&d.high.ge(d[["open","close"]].max(axis=1))&d.low.le(d[["open","close"]].min(axis=1))&d.high.ge(d.low)&d.close_time.eq(d.date+pd.Timedelta(hours=1))
 returns=np.log(d.close/d.close.shift(1));d["dvol_valid"]=coherent.rolling(25,min_periods=25).sum().eq(25)&exact.rolling(24,min_periods=24).sum().eq(24);d["dvol_change"]=np.log(d.close/d.close.shift(24)).where(d.dvol_valid);d["dvol_variation"]=np.sqrt(returns.pow(2).rolling(24,min_periods=24).sum()).where(d.dvol_valid)
 m=market.copy();m["date"]=pd.to_datetime(m.date,utc=True);m=m.sort_values("date").set_index("date");close=pd.to_numeric(m.close,errors="coerce");valid=np.isfinite(close)&close.gt(0);step=m.index.to_series().diff().eq(pd.Timedelta(minutes=5));btc_var=np.sqrt(np.log(close/close.shift(1)).pow(2).rolling(288,min_periods=288).sum());btc_ok=valid.rolling(289,min_periods=289).sum().eq(289)&step.rolling(288,min_periods=288).sum().eq(288)
 lookup=pd.DataFrame({"decision_time":m.index+pd.Timedelta(minutes=5),"btc_variation":btc_var.where(btc_ok).to_numpy()});d=d.merge(lookup,left_on="close_time",right_on="decision_time",how="left",validate="one_to_one")
 d["state_valid"]=d.dvol_valid&np.isfinite(d[["dvol_change","dvol_variation","btc_variation"]]).all(axis=1)&d.dvol_change.ne(0);d["dvol_variation_rank"]=rank(d.dvol_variation.where(d.state_valid));d["btc_variation_rank"]=rank(d.btc_variation.where(d.state_valid));return d
def build_clock(states:pd.DataFrame,control:str="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 daily=states[states.decision_time.dt.hour.eq(0)].copy().reset_index(drop=True);used=daily.shift(1) if control=="one_day_stale_dvol" else daily;valid=used.state_valid.eq(True)&used.dvol_change.ne(0)
 if control=="one_day_stale_dvol":valid&=daily.decision_time.sub(used.decision_time).eq(pd.Timedelta(days=1))
 dvol_gate=pd.Series(True,index=daily.index) if control=="no_dvol_variation_gate" else used.dvol_variation_rank.ge(.75);btc_gate=pd.Series(True,index=daily.index) if control=="no_btc_variation_gate" else daily.btc_variation_rank.ge(.65);active=valid&dvol_gate&btc_gate;side=-np.sign(used.dvol_change).fillna(0).astype(int)
 if control=="dvol_direction":side=-side
 elif control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=daily.index)
 rows=[]
 for i in daily.index[active]:
  decision=pd.Timestamp(daily.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12);split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  source=used.loc[i];rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"dvol_change":float(source.dvol_change),"dvol_variation":float(source.dvol_variation),"dvol_variation_rank":float(source.dvol_variation_rank),"btc_variation":float(daily.at[i,"btc_variation"]),"btc_variation_rank":float(daily.at[i,"btc_variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict[str,Any]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 bindings={prereg.DEFAULT_OUTPUT:PREREG_SHA,HELPER:HELPER_SHA,prereg.DVOL:prereg.DVOL_SHA,prereg.SOURCE_MANIFEST:prereg.SOURCE_MANIFEST_SHA,prereg.MARKET:prereg.MARKET_SHA}
 for p,h in bindings.items():
  if sha(p)!=h:raise RuntimeError(f"HVDVVR binding drift: {p}")
 dvol=pd.read_csv(prereg.DVOL);market,market_source=load_market();states=score_states(dvol,market);primary=build_clock(states);controls={n:build_clock(states,n) for n in CONTROLS};STATE.parent.mkdir(parents=True,exist_ok=True);CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(states,STATE);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=MINIMUM[n]),(f"{n}_side_balance",x["minority_side_share"]>=.2),(f"{n}_month_concentration",x["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvdvvr_12_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"bindings":{str(p):h for p,h in bindings.items()},"market_source":market_source,"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"source_state":{"path":str(STATE),"sha256":sha(STATE),"rows":len(states)},"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":prereg.canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
