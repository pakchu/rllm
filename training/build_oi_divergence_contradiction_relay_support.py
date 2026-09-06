"""Build source-only OIDCR-8 clocks from the frozen pre-entry feature snapshot."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_oi_divergence_contradiction_relay as prereg
from training import backtest_all_alpha_month as month
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

CLOCK=Path("data/oi_divergence_contradiction_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/oi_divergence_contradiction_relay_controls_2023_2026");RESULT=Path("results/oi_divergence_contradiction_relay_support_2026-08-08.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MIN={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_range_vol_gate","no_rsi_gate","no_return_contradiction","one_bar_stale_features","direction_flip");ECONOMIC_OUTCOMES_AUTHORIZED=False
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def signals(frame:pd.DataFrame,control:str="primary")->tuple[np.ndarray,np.ndarray]:
 t=prereg.THRESHOLDS;f=frame.shift(1) if control=="one_bar_stale_features" else frame
 oi=pd.to_numeric(f.oi_minus_px_4h_z,errors="coerce").to_numpy(float);ret=pd.to_numeric(f.return_zscore_48,errors="coerce").to_numpy(float);rv=pd.to_numeric(f.range_vol,errors="coerce").to_numpy(float);rsi=pd.to_numeric(f.rsi_norm,errors="coerce").to_numpy(float)
 finite=np.isfinite(oi)&np.isfinite(ret)&np.isfinite(rv)&np.isfinite(rsi);vol=np.ones(len(f),bool) if control=="no_range_vol_gate" else rv>=t["range_vol"];rsi_long=np.ones(len(f),bool) if control=="no_rsi_gate" else rsi<=-t["rsi_abs"];rsi_short=np.ones(len(f),bool) if control=="no_rsi_gate" else rsi>=t["rsi_abs"]
 if control=="no_return_contradiction":ret_long=ret_short=np.ones(len(f),bool)
 else:ret_long=ret<=-t["return_z_abs"];ret_short=ret>=t["return_z_abs"]
 dates=pd.to_datetime(frame.date,utc=True).dt.tz_convert(None);slot=month._interval_slots(dates,6,month._research_offset(6))
 return finite&slot&vol&rsi_long&ret_long&(oi>=t["oi_abs"]),finite&slot&vol&rsi_short&ret_short&(oi<=-t["oi_abs"])
def clock(frame:pd.DataFrame,control:str="primary")->pd.DataFrame:
 lo,sh=signals(frame,control);dates=pd.to_datetime(frame.date,utc=True);rows=[];next_allowed=None
 for i in np.flatnonzero(lo|sh):
  if lo[i] and sh[i]:continue
  entry=dates.iloc[i]+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  next_allowed=exit_;side=1 if lo[i] else -1;rows.append({"candidate":"OIDCR-8","control":control,"split":split,"decision_time":entry,"feature_available_time":entry,"entry_time":entry,"exit_time":exit_,"side":-side if control=="direction_flip" else side,"state":"inventory_absorption" if side==1 else "unsupported_price_appreciation"})
 return pd.DataFrame(rows,columns=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","state"))
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);f=pd.read_csv(prereg.SOURCE,compression="gzip");f["date"]=pd.to_datetime(f.date,utc=True);primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MIN[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"oidcr_8_source_support_v1","policy_id":"OIDCR-8","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_snapshot":{"path":str(prereg.SOURCE),"sha256":sha(prereg.SOURCE),"rows":len(f)},"completed_preentry_sources_opened":True,"btc_postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
