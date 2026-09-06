"""Build source-support clocks for DQDIR-6 without post-entry outcomes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_options_crowding_deleveraging_relay_support_v4 as volbase
from training import build_stablecoin_quote_flow_diffusion_support as flowbase
from training import preregister_dominant_quote_deleveraging_ignition_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PANEL=Path("data/binance_stablecoin_quote_flow_btc_2023_2026_aug/BTC_stablecoin_quote_flow_1h_2023-07-01_2026-07-31T23.csv.gz");PANEL_SHA="44374b9a2298ae4b64f0c1e7208665b1c08c8221045308694311123deae1c805";FLOW_MANIFEST=PANEL.parent/"build_manifest.json";FLOW_MANIFEST_SHA="b9c64c3ce651934d9761a6d0731e814b2a92f5237b3040e11f794d7eb024a898";VOL_DIR=Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026")
CLOCK=Path("data/dominant_quote_deleveraging_ignition_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/dominant_quote_deleveraging_ignition_relay_controls_2023_2026");RESULT=Path("results/dominant_quote_deleveraging_ignition_relay_support_2026-08-08.json")
SPLITS=volbase.SPLITS;MIN={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_joint_expansion","no_oi_deleveraging","no_alternative_quiet","one_hour_stale_flow","direction_flip");ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","source_hour_start","decision_time","feature_available_time","entry_time","exit_time","side","z_usdt","z_usdc","z_fdusd","oi_current_time","oi_prior_time","oi_change","bvol_body","dvol_body")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def load_flow()->pd.DataFrame:
 if sha(PANEL)!=PANEL_SHA or sha(FLOW_MANIFEST)!=FLOW_MANIFEST_SHA:raise RuntimeError("DQDIR flow source drift")
 s=pd.read_csv(PANEL,compression="gzip");s["date"]=pd.to_datetime(s.date,utc=True,errors="raise")
 if tuple(s.columns)!=flowbase.SOURCE_COLUMNS or s[["date","symbol"]].duplicated().any() or not s.source_complete.all():raise RuntimeError("DQDIR flow schema invalid")
 return flowbase.derive_state(s)
def features()->pd.DataFrame:
 b,d,o,f=volbase.load_sources(VOL_DIR);v=volbase.joined_features(b,d,o,f);flow=load_flow();j=flow.merge(v,on="decision_time",how="inner",validate="one_to_one").sort_values("decision_time").reset_index(drop=True)
 current_age=j.decision_time-j.oi_current_time;prior_target=j.decision_time-pd.Timedelta(hours=1);prior_age=prior_target-j.oi_prior_time
 oi_valid=np.isfinite(j[["oi_current","oi_prior","oi_change"]]).all(axis=1)&j[["oi_current","oi_prior"]].gt(0).all(axis=1)&current_age.between(pd.Timedelta(0),pd.Timedelta(minutes=5))&prior_age.between(pd.Timedelta(0),pd.Timedelta(minutes=5))
 vol_valid=j.bvol_valid&np.isfinite(j[["bvol_open","bvol_close","dvol_open","dvol_close","bvol_body","dvol_body"]]).all(axis=1)&j[["bvol_open","bvol_close","dvol_open","dvol_close"]].gt(0).all(axis=1)
 j["signal_valid"]=j.source_valid&oi_valid&vol_valid;return j
def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series,pd.DataFrame]:
 flow=f.shift(1) if control=="one_hour_stale_flow" else f;impulse=flow.z_usdt.ne(0)&flow.z_usdt.abs().ge(.75);quiet=pd.Series(True,index=f.index) if control=="no_alternative_quiet" else flow.z_usdc.abs().lt(.5)&flow.z_fdusd.abs().lt(.5);oi=pd.Series(True,index=f.index) if control=="no_oi_deleveraging" else f.oi_change.lt(0);vol=pd.Series(True,index=f.index) if control=="no_joint_expansion" else f.bvol_body.gt(0)&f.dvol_body.gt(0)
 previous=f.shift(1);valid=f.signal_valid&previous.signal_valid&f.decision_time.diff().eq(pd.Timedelta(hours=1));active=valid&impulse&quiet&oi&vol;return active&~active.shift(1,fill_value=False),np.sign(flow.z_usdt),flow
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 on,side,flow=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[on]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  signal_side=int(side.at[i]);signal_side=-signal_side if control=="direction_flip" else signal_side;next_allowed=exit_;x=flow.loc[i];rows.append({"candidate":"DQDIR-6","control":control,"split":split,"source_hour_start":x.source_hour_start,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":signal_side,"z_usdt":float(x.z_usdt),"z_usdc":float(x.z_usdc),"z_fdusd":float(x.z_fdusd),"oi_current_time":f.at[i,"oi_current_time"],"oi_prior_time":f.at[i,"oi_prior_time"],"oi_change":float(f.at[i,"oi_change"]),"bvol_body":float(f.at[i,"bvol_body"]),"dvol_body":float(f.at[i,"dvol_body"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MIN[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());vm=VOL_DIR/"manifest.json";passed=all(checks.values());core={"protocol_version":"dqdir_6_source_support_v1","policy_id":"DQDIR-6","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifests":{"flow":{"path":str(FLOW_MANIFEST),"sha256":sha(FLOW_MANIFEST)},"volatility_oi":{"path":str(vm),"sha256":sha(vm)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
