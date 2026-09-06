"""Build source-only support clocks for frozen DSQFCR-12."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd
from training import build_options_crowding_deleveraging_relay_support_v4 as volbase
from training import preregister_daily_stablecoin_quote_flow_consensus_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PANEL=Path("data/binance_stablecoin_quote_flow_btc_2023_2026_aug/BTC_stablecoin_quote_flow_1h_2023-07-01_2026-07-31T23.csv.gz")
PANEL_SHA="44374b9a2298ae4b64f0c1e7208665b1c08c8221045308694311123deae1c805";SOURCE_MANIFEST=PANEL.parent/"build_manifest.json";SOURCE_MANIFEST_SHA="b9c64c3ce651934d9761a6d0731e814b2a92f5237b3040e11f794d7eb024a898"
CLOCK=Path("data/daily_stablecoin_quote_flow_consensus_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/daily_stablecoin_quote_flow_consensus_relay_controls_2023_2026");RESULT=Path("results/daily_stablecoin_quote_flow_consensus_relay_support_2026-08-08.json")
SPLITS=volbase.SPLITS;MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_volatility_gate","usdt_only","usdc_only","no_participation_gate","direction_flip");ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","source_day","decision_time","feature_available_time","entry_time","exit_time","side","flow_usdt","flow_usdc","usdc_volume_share","bvol_close","prior_bvol_q60","dvol_close","prior_dvol_q60")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def features()->pd.DataFrame:
 if sha(PANEL)!=PANEL_SHA or sha(SOURCE_MANIFEST)!=SOURCE_MANIFEST_SHA:raise RuntimeError("DSQFCR spot-flow source drift")
 x=pd.read_csv(PANEL,compression="gzip");x["date"]=pd.to_datetime(x.date,utc=True,errors="raise");x=x[x.symbol.isin(["BTCUSDT","BTCUSDC"])].copy()
 if x[["date","symbol"]].duplicated().any():raise RuntimeError("DSQFCR duplicate source hour")
 for c in ("base_volume_btc","signed_taker_flow_btc"):x[c]=pd.to_numeric(x[c],errors="coerce")
 x["source_day"]=x.date.dt.floor("D")
 g=x.groupby(["source_day","symbol"],as_index=False).agg(hours=("date","size"),complete=("source_complete","all"),volume=("base_volume_btc","sum"),signed=("signed_taker_flow_btc","sum"))
 g["valid"]=g.hours.eq(24)&g.complete&np.isfinite(g[["volume","signed"]]).all(axis=1)&g.volume.gt(0)
 w=g.pivot(index="source_day",columns="symbol",values=["hours","valid","volume","signed"]);w.columns=[f"{a}_{b}" for a,b in w.columns];w=w.reset_index()
 required=[f"{a}_{s}" for a in ("valid","volume","signed") for s in ("BTCUSDT","BTCUSDC")]
 for c in required:
  if c not in w:w[c]=np.nan
 for c in ("volume_BTCUSDT","volume_BTCUSDC","signed_BTCUSDT","signed_BTCUSDC"):w[c]=pd.to_numeric(w[c],errors="coerce")
 w["flow_usdt"]=w.signed_BTCUSDT/w.volume_BTCUSDT;w["flow_usdc"]=w.signed_BTCUSDC/w.volume_BTCUSDC;w["usdc_volume_share"]=w.volume_BTCUSDC/(w.volume_BTCUSDT+w.volume_BTCUSDC);numeric=w[["flow_usdt","flow_usdc","usdc_volume_share"]].apply(pd.to_numeric,errors="coerce");w[["flow_usdt","flow_usdc","usdc_volume_share"]]=numeric;w["flow_valid"]=w.valid_BTCUSDT.astype(bool)&w.valid_BTCUSDC.astype(bool)&np.isfinite(numeric).all(axis=1)
 w["decision_time"]=w.source_day+pd.Timedelta(days=1)
 b,d,oi,funding=volbase.load_sources();v=volbase.joined_features(b,d,oi,funding)[["decision_time","bvol_close","dvol_close","bvol_valid"]].copy()
 for c in ("bvol_close","dvol_close"):
  p=c.split("_")[0];v[c]=pd.to_numeric(v[c],errors="coerce");valid=v[c].where(v.bvol_valid&np.isfinite(v[c])&v[c].gt(0));v[f"prior_{p}_q60"]=valid.shift(1).rolling(720,min_periods=672).quantile(.60)
 v=v[v.decision_time.dt.hour.eq(0)&v.decision_time.dt.minute.eq(0)].copy();v["vol_valid"]=v.bvol_valid&np.isfinite(v[["bvol_close","dvol_close","prior_bvol_q60","prior_dvol_q60"]]).all(axis=1)&v[["bvol_close","dvol_close"]].gt(0).all(axis=1)
 return w.merge(v[["decision_time","bvol_close","dvol_close","prior_bvol_q60","prior_dvol_q60","vol_valid"]],on="decision_time",how="inner",validate="one_to_one").sort_values("decision_time").reset_index(drop=True)

def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 usdt=f.flow_usdt;usdc=f.flow_usdc
 if control=="usdt_only":consensus=usdt.abs().ge(.05)&usdt.ne(0);side=np.sign(usdt)
 elif control=="usdc_only":consensus=usdc.abs().ge(.05)&usdc.ne(0);side=np.sign(usdc)
 else:consensus=usdt.abs().ge(.05)&usdc.abs().ge(.05)&usdt.ne(0)&usdc.ne(0)&np.sign(usdt).eq(np.sign(usdc));side=np.sign(usdt)
 participation=pd.Series(True,index=f.index) if control=="no_participation_gate" else f.usdc_volume_share.ge(.01)
 volatile=pd.Series(True,index=f.index) if control=="no_volatility_gate" else f.bvol_close.ge(f.prior_bvol_q60)&f.dvol_close.ge(f.prior_dvol_q60)
 return f.flow_valid&f.vol_valid&consensus&participation&volatile,side

def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  ss=int(side.at[i]);ss=-ss if control=="direction_flip" else ss;next_allowed=exit_;rows.append({"candidate":"DSQFCR-12","control":control,"split":split,"source_day":f.at[i,"source_day"],"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":ss,"flow_usdt":float(f.at[i,"flow_usdt"]),"flow_usdc":float(f.at[i,"flow_usdc"]),"usdc_volume_share":float(f.at[i,"usdc_volume_share"]),"bvol_close":float(f.at[i,"bvol_close"]),"prior_bvol_q60":float(f.at[i,"prior_bvol_q60"]),"dvol_close":float(f.at[i,"dvol_close"]),"prior_dvol_q60":float(f.at[i,"prior_dvol_q60"])})
 return pd.DataFrame(rows,columns=COLUMNS)

def stats(c:pd.DataFrame,n:str)->dict[str,int|float]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.0,"max_month_share":0.0}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}

def run()->dict[str,Any]:
 f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 st={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in st.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());volmanifest=volbase.SOURCE_DIR/"manifest.json";passed=all(checks.values());core={"protocol_version":"dsqfcr_12_source_support_v1","policy_id":"DSQFCR-12","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifests":{"spot_flow":{"path":str(SOURCE_MANIFEST),"sha256":sha(SOURCE_MANIFEST)},"volatility":{"path":str(volmanifest),"sha256":sha(volmanifest)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":st,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
