"""Build source-only HVEABRR-12 clocks before Gross9 or economics."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_high_volatility_equity_adjusted_btc_residual_reversal as prereg
from training import build_bitcoin_stock_correlation_break_relay_support as cross
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market
PREREG_SHA="9c28b1305045cb0bf461563f7428837e5f9965cec9eae7c98160506d46b03748";CROSS_HELPER=Path("training/build_bitcoin_stock_correlation_break_relay_support.py");CROSS_HELPER_SHA="a72b4504908748941adb60d11214e21ccdc6c39bc753805a14db84a991a13392";MARKET_HELPER=Path("training/build_scheduled_trend_concordance_relay_support.py");MARKET_HELPER_SHA="8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f";END=pd.Timestamp("2026-08-01T00:00:00Z")
STATE=Path("data/high_volatility_equity_adjusted_btc_residual_reversal_sources_2023_2026/session_states.csv.gz");CLOCK=Path("data/high_volatility_equity_adjusted_btc_residual_reversal_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_equity_adjusted_btc_residual_reversal_controls_2023_2026");RESULT=Path("results/high_volatility_equity_adjusted_btc_residual_reversal_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_variation_gate","no_residual_tail","raw_btc_return_reversal","one_session_stale_beta","direction_flip","same_clock_forced_long");COLUMNS=("candidate","control","split","session_date","cash_close_time","feature_available_time","entry_time","exit_time","side","spy_return","btc_return","beta","residual","residual_rank","variation","variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def beta(prior:pd.DataFrame)->float:
 if len(prior)<40 or not {"spy_return","btc_return"}.issubset(prior.columns):return np.nan
 x=prior.spy_return.to_numpy(float);y=prior.btc_return.to_numpy(float)
 if not np.isfinite(x).all() or not np.isfinite(y).all():return np.nan
 variance=float(np.mean(np.square(x-x.mean())))
 return float(np.mean((x-x.mean())*(y-y.mean()))/variance) if variance>0 else np.nan
def score_states(paired:pd.DataFrame,market:pd.DataFrame)->pd.DataFrame:
 f=market.copy();f["date"]=pd.to_datetime(f.date,utc=True);closes=f.set_index("date").sort_index().close.astype(float);valid_history=[];residual_history=[];variation_history=[];rows=[]
 for _,r in paired.iterrows():
  if not bool(r.pair_valid):continue
  close=pd.Timestamp(r.cash_close_time);idx=pd.date_range(close-pd.Timedelta(hours=24,minutes=5),close-pd.Timedelta(minutes=5),freq="5min");values=closes.reindex(idx).to_numpy(float)
  if len(values)!=289 or not np.isfinite(values).all() or not (values>0).all():continue
  variation=float(np.sqrt(np.square(np.diff(np.log(values))).sum()));vh=np.asarray(variation_history[-90:],float);vr=float(((vh<variation).sum()+.5*(vh==variation).sum())/len(vh)) if len(vh)>=60 else np.nan
  history=pd.DataFrame(valid_history[-60:]);b=beta(history);stale=beta(pd.DataFrame(valid_history[-61:-1])) if len(valid_history)>=41 else np.nan;residual=float(r.btc_return-b*r.spy_return) if np.isfinite(b) else np.nan;stale_residual=float(r.btc_return-stale*r.spy_return) if np.isfinite(stale) else np.nan;rh=np.asarray(residual_history[-90:],float);rr=float(((np.abs(rh)<abs(residual)).sum()+.5*(np.abs(rh)==abs(residual)).sum())/len(rh)) if np.isfinite(residual) and len(rh)>=60 else np.nan
  rows.append({"session_date":r.session_date,"cash_close_time":close,"spy_return":float(r.spy_return),"btc_return":float(r.btc_return),"beta":b,"stale_beta":stale,"residual":residual,"stale_residual":stale_residual,"residual_rank":rr,"variation":variation,"variation_rank":vr,"elapsed_gap_hours":float(r.elapsed_gap_hours)})
  valid_history.append({"spy_return":float(r.spy_return),"btc_return":float(r.btc_return)});variation_history.append(variation)
  if np.isfinite(residual):residual_history.append(residual)
 return pd.DataFrame(rows)
def build_clock(s:pd.DataFrame,control:str="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 rows=[]
 for _,r in s.iterrows():
  residual=float(r.stale_residual if control=="one_session_stale_beta" else r.residual);b=float(r.stale_beta if control=="one_session_stale_beta" else r.beta);tail=bool(np.isfinite(r.residual_rank) and r.residual_rank>=.75);volatile=bool(np.isfinite(r.variation_rank) and r.variation_rank>=.65)
  if not np.isfinite(residual) or residual==0 or (not tail and control!="no_residual_tail") or (not volatile and control!="no_variation_gate"):continue
  side=-int(np.sign(residual))
  if control=="raw_btc_return_reversal":side=-int(np.sign(r.btc_return)) if r.btc_return!=0 else 0
  elif control=="direction_flip":side=-side
  elif control=="same_clock_forced_long":side=1
  if side==0:continue
  close=pd.Timestamp(r.cash_close_time);feature=close+pd.Timedelta(minutes=5);entry=close+pd.Timedelta(minutes=10);exit_=entry+pd.Timedelta(hours=12);split=next((n for n,(a,z) in SPLITS.items() if entry>=a and exit_<=z),None)
  if split is None:continue
  rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"session_date":r.session_date,"cash_close_time":close,"feature_available_time":feature,"entry_time":entry,"exit_time":exit_,"side":side,"spy_return":float(r.spy_return),"btc_return":float(r.btc_return),"beta":b,"residual":residual,"residual_rank":float(r.residual_rank),"variation":float(r.variation),"variation_rank":float(r.variation_rank)})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict[str,Any]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 bindings={prereg.DEFAULT_OUTPUT:PREREG_SHA,CROSS_HELPER:CROSS_HELPER_SHA,MARKET_HELPER:MARKET_HELPER_SHA,prereg.SPY:prereg.SPY_SHA,prereg.SCHEDULE:prereg.SCHEDULE_SHA,prereg.BTC_HOURLY:prereg.BTC_HOURLY_SHA,prereg.MARKET:prereg.MARKET_SHA}
 for p,h in bindings.items():
  if sha(p)!=h:raise RuntimeError(f"HVEABRR binding drift: {p}")
 spy,_=cross.load_spy();schedule=cross.session_schedule(spy);frozen=pd.read_csv(prereg.SCHEDULE);frozen["session_date"]=pd.to_datetime(frozen.session_date,utc=True).dt.tz_localize(None);frozen["cash_close_time"]=pd.to_datetime(frozen.cash_close_time,utc=True)
 if not schedule[["session_date","cash_close_time","close_local_time","early_close"]].reset_index(drop=True).equals(frozen[["session_date","cash_close_time","close_local_time","early_close"]].reset_index(drop=True)):raise RuntimeError("HVEABRR session schedule parity drift")
 btc=pd.read_csv(prereg.BTC_HOURLY);btc["hour_start"]=pd.to_datetime(btc.hour_start,utc=True);btc["decision_time"]=pd.to_datetime(btc.decision_time,utc=True);paired=cross.paired_returns(spy,schedule,btc);market,market_source=load_market();states=score_states(paired,market);primary=build_clock(states);controls={n:build_clock(states,n) for n in CONTROLS};STATE.parent.mkdir(parents=True,exist_ok=True);CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(states,STATE);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=MINIMUM[n]),(f"{n}_side_balance",x["minority_side_share"]>=.2),(f"{n}_month_concentration",x["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hveabrr_12_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"bindings":{str(p):h for p,h in bindings.items()},"source":{"market":market_source,"paired_session_rows":len(paired),"corporate_action_adjusted_spy":True,"actual_nyse_close_schedule_verified":True},"source_state":{"path":str(STATE),"sha256":sha(STATE),"rows":len(states)},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":prereg.canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
