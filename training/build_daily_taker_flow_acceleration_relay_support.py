"""Build source-only support clocks for frozen DTFAR-12."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_daily_taker_flow_acceleration_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
FLOW_PANEL=Path("data/binance_stablecoin_quote_flow_btc_2023_2026_aug/BTC_stablecoin_quote_flow_1h_2023-07-01_2026-07-31T23.csv.gz");FLOW_SHA="44374b9a2298ae4b64f0c1e7208665b1c08c8221045308694311123deae1c805";FLOW_MANIFEST=FLOW_PANEL.parent/"build_manifest.json";FLOW_MANIFEST_SHA="b9c64c3ce651934d9761a6d0731e814b2a92f5237b3040e11f794d7eb024a898";PRICE_DIR=Path("data/options_oi_chase_exhaustion_sources_2023_2026");CLOCK=Path("data/daily_taker_flow_acceleration_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/daily_taker_flow_acceleration_relay_controls_2023_2026");RESULT=Path("results/daily_taker_flow_acceleration_relay_support_2026-08-08.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_volatility_gate","no_flow_tail","flow_level","one_day_stale_acceleration","direction_flip");ECONOMIC_OUTCOMES_AUTHORIZED=False;COLUMNS=("candidate","control","split","source_day","decision_time","feature_available_time","entry_time","exit_time","side","normalized_flow","flow_change","flow_change_z","realized_variation","realized_variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def strict_prior_midrank(v:pd.Series,lookback:int=90,minimum:int=60)->pd.Series:
 n=pd.to_numeric(v,errors="coerce").astype(float);o=pd.Series(np.nan,index=n.index,dtype=float);h=[]
 for i,x in n.items():
  p=h[-lookback:]
  if math.isfinite(x) and len(p)>=minimum:
   a=np.asarray(p);o.at[i]=(np.sum(a<x)+.5*np.sum(a==x))/len(a)
  if math.isfinite(x):h.append(x)
 return o
def causal_z(v:pd.Series,lookback:int=90,minimum:int=60)->pd.Series:
 x=pd.to_numeric(v,errors="coerce").to_numpy(float);o=np.full(len(x),np.nan)
 for i,current in enumerate(x):
  p=x[max(0,i-lookback):i];p=p[np.isfinite(p)]
  if np.isfinite(current) and len(p)>=minimum:
   s=float(np.std(p,ddof=1));o[i]=(current-float(np.mean(p)))/s if s>0 else np.nan
 return pd.Series(o,index=v.index)
def features()->pd.DataFrame:
 if sha(FLOW_PANEL)!=FLOW_SHA or sha(FLOW_MANIFEST)!=FLOW_MANIFEST_SHA:raise RuntimeError("DTFAR flow source drift")
 x=pd.read_csv(FLOW_PANEL,compression="gzip");x["date"]=pd.to_datetime(x.date,utc=True,errors="raise");x=x[x.symbol.eq("BTCUSDT")].copy();x["base_volume_btc"]=pd.to_numeric(x.base_volume_btc,errors="coerce");x["signed_taker_flow_btc"]=pd.to_numeric(x.signed_taker_flow_btc,errors="coerce");x["source_day"]=x.date.dt.floor("D");g=x.groupby("source_day",as_index=False).agg(hours=("date","size"),complete=("source_complete","all"),volume=("base_volume_btc","sum"),signed=("signed_taker_flow_btc","sum"));g["flow_valid"]=g.hours.eq(24)&g.complete&np.isfinite(g[["volume","signed"]]).all(axis=1)&g.volume.gt(0);g["normalized_flow"]=g.signed/g.volume;g["flow_change"]=g.normalized_flow.diff();g["flow_change_z"]=causal_z(g.flow_change.where(g.flow_valid&g.flow_valid.shift(1,fill_value=False)&g.source_day.diff().eq(pd.Timedelta(days=1))));g["flow_level_z"]=causal_z(g.normalized_flow.where(g.flow_valid));g["decision_time"]=g.source_day+pd.Timedelta(days=1)
 p=pd.read_csv(PRICE_DIR/"btc_completed_hour.csv.gz",compression="gzip");p["decision_time"]=pd.to_datetime(p.decision_time,utc=True,format="mixed");p["open"]=pd.to_numeric(p.open,errors="coerce");p["close"]=pd.to_numeric(p.close,errors="coerce");p["valid"]=p.source_valid.astype(str).str.lower().eq("true")&np.isfinite(p[["open","close"]]).all(axis=1)&p[["open","close"]].gt(0).all(axis=1);p=p.sort_values("decision_time").reset_index(drop=True);consecutive=p.decision_time.diff().eq(pd.Timedelta(hours=1));p["hour_return"]=np.log(p.close/p.open);p["realized_variation"]=np.sqrt(p.hour_return.pow(2).rolling(24,min_periods=24).sum());p["price_valid_day"]=p.valid.rolling(24,min_periods=24).sum().eq(24)&consecutive.rolling(23,min_periods=23).sum().eq(23)&np.isfinite(p.realized_variation);d=p[p.decision_time.dt.hour.eq(0)&p.decision_time.dt.minute.eq(0)][["decision_time","realized_variation","price_valid_day"]].copy().reset_index(drop=True);d["realized_variation_rank"]=strict_prior_midrank(d.realized_variation.where(d.price_valid_day));f=g.merge(d,on="decision_time",how="inner",validate="one_to_one");f["source_valid"]=f.flow_valid&f.price_valid_day&np.isfinite(f[["flow_change","flow_change_z","realized_variation","realized_variation_rank"]]).all(axis=1)&f.flow_change.ne(0);return f
def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 change=f.normalized_flow if control=="flow_level" else f.flow_change;z=f.flow_level_z if control=="flow_level" else f.flow_change_z
 if control=="one_day_stale_acceleration":change=f.flow_change.shift(1);z=f.flow_change_z.shift(1)
 tail=pd.Series(True,index=f.index) if control=="no_flow_tail" else z.abs().ge(1.);vol=pd.Series(True,index=f.index) if control=="no_volatility_gate" else f.realized_variation_rank.ge(.65);active=f.source_valid&np.isfinite(change)&np.isfinite(z)&change.ne(0)&tail&vol;side=np.sign(change);side=-side if control=="direction_flip" else side;return active,side
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  next_allowed=exit_;rows.append({"candidate":"DTFAR-12","control":control,"split":split,"source_day":f.at[i,"source_day"],"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"normalized_flow":float(f.at[i,"normalized_flow"]),"flow_change":float(f.at[i,"flow_change"]),"flow_change_z":float(f.at[i,"flow_change_z"]),"realized_variation":float(f.at[i,"realized_variation"]),"realized_variation_rank":float(f.at[i,"realized_variation_rank"])})
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
 pm=PRICE_DIR/"manifest.json";passed=all(checks.values());core={"protocol_version":"dtfar_12_source_support_v1","policy_id":"DTFAR-12","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifests":{"flow":{"path":str(FLOW_MANIFEST),"sha256":sha(FLOW_MANIFEST)},"completed_price":{"path":str(pm),"sha256":sha(pm)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
