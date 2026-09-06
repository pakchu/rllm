"""Build source-support clocks for VGSFR-6 without post-entry outcomes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import build_options_crowding_deleveraging_relay_support_v4 as volbase
from training import build_stablecoin_quote_flow_diffusion_support as flowbase
from training import preregister_volatility_gated_stablecoin_sequential_follower_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PANEL=Path("data/binance_stablecoin_quote_flow_btc_2023_2026_aug/BTC_stablecoin_quote_flow_1h_2023-07-01_2026-07-31T23.csv.gz");PANEL_SHA="44374b9a2298ae4b64f0c1e7208665b1c08c8221045308694311123deae1c805"
SOURCE_MANIFEST=PANEL.parent/"build_manifest.json";SOURCE_MANIFEST_SHA="b9c64c3ce651934d9761a6d0731e814b2a92f5237b3040e11f794d7eb024a898"
CLOCK=Path("data/volatility_gated_stablecoin_sequential_follower_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/volatility_gated_stablecoin_sequential_follower_relay_controls_2023_2026");RESULT=Path("results/volatility_gated_stablecoin_sequential_follower_relay_support_2026-08-08.json")
SPLITS=volbase.SPLITS;MIN={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_sequential_order","no_volatility_gate","no_usdt_lag","no_participation","direction_flip");ECONOMIC_OUTCOMES_AUTHORIZED=False
COLUMNS=("candidate","control","split","prior_source_hour_start","current_source_hour_start","decision_time","feature_available_time","entry_time","exit_time","side","leader","prior_z_usdt","prior_z_usdc","prior_z_fdusd","current_z_usdt","current_z_usdc","current_z_fdusd","alt_share","prior_alt_share_q50","bvol_close","prior_bvol_q60","dvol_close","prior_dvol_q60")
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def chash(value:Any)->str:return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def load_flow()->pd.DataFrame:
 if sha(PANEL)!=PANEL_SHA or sha(SOURCE_MANIFEST)!=SOURCE_MANIFEST_SHA:raise RuntimeError("VGSFR spot-flow source drift")
 source=pd.read_csv(PANEL,compression="gzip");source["date"]=pd.to_datetime(source.date,utc=True,errors="raise")
 if tuple(source.columns)!=flowbase.SOURCE_COLUMNS or source[["date","symbol"]].duplicated().any() or not source.source_complete.all():raise RuntimeError("VGSFR spot-flow schema invalid")
 return flowbase.derive_state(source)

def features()->pd.DataFrame:
 state=load_flow();bvol,dvol,oi,funding=volbase.load_sources();vol=volbase.joined_features(bvol,dvol,oi,funding)[["decision_time","bvol_close","dvol_close","bvol_valid"]].copy()
 for c in ("bvol_close","dvol_close"):vol[c]=pd.to_numeric(vol[c],errors="coerce");vol[f"prior_{c.split('_')[0]}_q60"]=vol[c].where(vol.bvol_valid&np.isfinite(vol[c])&vol[c].gt(0)).shift(1).rolling(720,min_periods=672).quantile(.60)
 frame=state.merge(vol,on="decision_time",how="inner",validate="one_to_one").sort_values("decision_time").reset_index(drop=True)
 frame["vol_valid"]=frame.bvol_valid&np.isfinite(frame[["bvol_close","dvol_close"]]).all(axis=1)&frame[["bvol_close","dvol_close"]].gt(0).all(axis=1)
 return frame

def conditions(frame:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series,pd.Series]:
 p=frame.shift(1);same=np.sign(p.z_usdc).eq(np.sign(p.z_fdusd))&p.z_usdc.ne(0)&p.z_fdusd.ne(0)
 lead_usdc=same&p.z_usdc.abs().ge(1)&p.z_fdusd.abs().lt(.5);lead_fdusd=same&p.z_fdusd.abs().ge(1)&p.z_usdc.abs().lt(.5);ordered=lead_usdc|lead_fdusd
 side=np.sign(p.z_usdc).where(lead_usdc,np.sign(p.z_fdusd));current_same=np.sign(frame.z_usdc).eq(side)&np.sign(frame.z_fdusd).eq(side)
 follow_usdc=lead_fdusd&frame.z_usdc.abs().ge(.75)&frame.z_fdusd.abs().ge(.5);follow_fdusd=lead_usdc&frame.z_fdusd.abs().ge(.75)&frame.z_usdc.abs().ge(.5)
 sequential=ordered&current_same&(follow_usdc|follow_fdusd)
 if control=="no_sequential_order":
  side=np.sign(frame.z_usdc+frame.z_fdusd);sequential=np.sign(frame.z_usdc).eq(np.sign(frame.z_fdusd))&side.ne(0)&frame.z_usdc.abs().ge(.75)&frame.z_fdusd.abs().ge(.75)
 usdt=pd.Series(True,index=frame.index) if control=="no_usdt_lag" else (side*p.z_usdt).lt(.25)&(side*frame.z_usdt).lt(.5)
 participation=pd.Series(True,index=frame.index) if control=="no_participation" else frame.alt_share.ge(frame.prior_alt_share_q50)
 volatile=pd.Series(True,index=frame.index) if control=="no_volatility_gate" else frame.bvol_close.ge(frame.prior_bvol_q60)&frame.dvol_close.ge(frame.prior_dvol_q60)
 consecutive=frame.decision_time.diff().eq(pd.Timedelta(hours=1));valid=frame.source_valid&p.source_valid&frame.vol_valid&consecutive&frame.prior_bvol_q60.notna()&frame.prior_dvol_q60.notna()
 return valid&sequential&usdt&participation&volatile,side,lead_usdc

def clock(frame:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side,lead_usdc=conditions(frame,control);on=active&~active.shift(1,fill_value=False);rows=[];next_allowed=None
 for i in frame.index[on]:
  decision=pd.Timestamp(frame.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  s=int(side.at[i]);s=-s if control=="direction_flip" else s;next_allowed=exit_;leader="BTCUSDC" if bool(lead_usdc.at[i]) else "BTCFDUSD"
  rows.append({"candidate":"VGSFR-6","control":control,"split":split,"prior_source_hour_start":frame.at[i-1,"source_hour_start"],"current_source_hour_start":frame.at[i,"source_hour_start"],"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":s,"leader":leader,"prior_z_usdt":float(frame.at[i-1,"z_usdt"]),"prior_z_usdc":float(frame.at[i-1,"z_usdc"]),"prior_z_fdusd":float(frame.at[i-1,"z_fdusd"]),"current_z_usdt":float(frame.at[i,"z_usdt"]),"current_z_usdc":float(frame.at[i,"z_usdc"]),"current_z_fdusd":float(frame.at[i,"z_fdusd"]),"alt_share":float(frame.at[i,"alt_share"]),"prior_alt_share_q50":float(frame.at[i,"prior_alt_share_q50"]),"bvol_close":float(frame.at[i,"bvol_close"]),"prior_bvol_q60":float(frame.at[i,"prior_bvol_q60"]),"dvol_close":float(frame.at[i,"dvol_close"]),"prior_dvol_q60":float(frame.at[i,"prior_dvol_q60"])})
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
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());volmanifest=volbase.SOURCE_DIR/"manifest.json";passed=all(checks.values());core={"protocol_version":"vgsfr_6_source_support_v1","policy_id":"VGSFR-6","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifests":{"spot_flow":{"path":str(SOURCE_MANIFEST),"sha256":sha(SOURCE_MANIFEST)},"volatility":{"path":str(volmanifest),"sha256":sha(volmanifest)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x)} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":ECONOMIC_OUTCOMES_AUTHORIZED,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return result
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
