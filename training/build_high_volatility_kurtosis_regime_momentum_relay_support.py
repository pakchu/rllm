"""Build source-only HVKRMR-24 clocks before Gross9 or economics."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_high_volatility_kurtosis_regime_momentum_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market
PREREG_SHA="4f1cdafce69f24cf785eaf8e1ce2e6e0ff2ada97654a3a4e426dea7c963e4d56";HELPER=Path("training/build_scheduled_trend_concordance_relay_support.py");HELPER_SHA="8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f";END=pd.Timestamp("2026-08-01T00:00:00Z")
STATE=Path("data/high_volatility_kurtosis_regime_momentum_relay_sources_2023_2026/daily_states.csv.gz");CLOCK=Path("data/high_volatility_kurtosis_regime_momentum_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_kurtosis_regime_momentum_relay_controls_2023_2026");RESULT=Path("results/high_volatility_kurtosis_regime_momentum_relay_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_kurtosis_tail","no_variation_gate","one_day_stale_moments","direction_flip","same_clock_forced_long");COLUMNS=("candidate","control","split","source_day","decision_time","feature_available_time","entry_time","exit_time","side","day_return","realized_variation","realized_kurtosis","kurtosis_rank","variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(v:pd.Series)->pd.Series:
 x=pd.to_numeric(v,errors="coerce");o=pd.Series(np.nan,index=x.index,dtype=float);h=[]
 for i,c in x.items():
  q=np.asarray(h[-270:],float)
  if np.isfinite(c) and len(q)>=180:o.at[i]=((q<c).sum()+.5*(q==c).sum())/len(q)
  if np.isfinite(c):h.append(float(c))
 return o
def score_states(market:pd.DataFrame)->pd.DataFrame:
 m=market.copy();m["date"]=pd.to_datetime(m.date,utc=True);m=m.sort_values("date").set_index("date");close=pd.to_numeric(m.close,errors="coerce");m["r"]=np.log(close/close.shift(1));rows=[]
 for day,w in m.groupby(m.index.floor("D"),sort=True):
  expected=pd.date_range(day,day+pd.Timedelta(days=1),freq="5min",inclusive="left");w=w.reindex(expected);ohlc=w[["open","high","low","close"]].apply(pd.to_numeric,errors="coerce");valid=len(w)==288 and np.isfinite(ohlc).all().all() and ohlc.gt(0).all().all() and np.isfinite(w.r).all() and w.index.equals(expected)
  r=pd.to_numeric(w.r,errors="coerce").to_numpy(float);rv=float(np.square(r).sum()) if valid else np.nan;rq=float(288*np.power(r,4).sum()/rv**2) if valid and rv>0 else np.nan;dr=float(np.log(float(ohlc.close.iloc[-1])/float(ohlc.open.iloc[0]))) if valid else np.nan;rows.append({"source_day":day,"decision_time":day+pd.Timedelta(days=1),"source_valid":bool(valid and np.isfinite([rv,rq,dr]).all() and dr!=0),"day_return":dr,"realized_variation":rv,"realized_kurtosis":rq})
 e=pd.DataFrame(rows);e["kurtosis_rank"]=rank(e.realized_kurtosis.where(e.source_valid));e["variation_rank"]=rank(np.sqrt(e.realized_variation).where(e.source_valid));return e
def build_clock(states:pd.DataFrame,control:str="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=states.shift(1) if control=="one_day_stale_moments" else states;valid=used.source_valid.eq(True)&used.day_return.ne(0)&np.isfinite(used[["day_return","realized_variation","realized_kurtosis","kurtosis_rank","variation_rank"]]).all(axis=1)
 if control=="one_day_stale_moments":valid&=states.source_day.sub(used.source_day).eq(pd.Timedelta(days=1))
 kg=pd.Series(True,index=states.index) if control=="no_kurtosis_tail" else used.kurtosis_rank.ge(.8);vg=pd.Series(True,index=states.index) if control=="no_variation_gate" else used.variation_rank.ge(.65);active=valid&kg&vg;side=np.sign(used.day_return).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=states.index)
 rows=[]
 for i in states.index[active]:
  decision=pd.Timestamp(states.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=24);split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  u=used.loc[i];rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"source_day":u.source_day,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"day_return":float(u.day_return),"realized_variation":float(u.realized_variation),"realized_kurtosis":float(u.realized_kurtosis),"kurtosis_rank":float(u.kurtosis_rank),"variation_rank":float(u.variation_rank)})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict[str,Any]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 bindings={prereg.DEFAULT_OUTPUT:PREREG_SHA,HELPER:HELPER_SHA,prereg.MARKET:prereg.MARKET_SHA}
 for p,h in bindings.items():
  if sha(p)!=h:raise RuntimeError(f"HVKRMR binding drift: {p}")
 market,market_source=load_market();states=score_states(market);primary=build_clock(states);controls={n:build_clock(states,n) for n in CONTROLS};STATE.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(states,STATE);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=MINIMUM[n]),(f"{n}_side_balance",x["minority_side_share"]>=.2),(f"{n}_month_concentration",x["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvkrmr_24_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"bindings":{str(p):h for p,h in bindings.items()},"market_source":market_source,"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"source_state":{"path":str(STATE),"sha256":sha(STATE),"rows":len(states)},"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":prereg.canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
