"""Materialize source-only SPVTA-8 support clocks."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_spot_perpetual_variance_transfer_asymmetry as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
PREREG_SHA="4de6c667cdd58ec5d38b8786309bbdd0af480572eb179db79d10bca2968a4f4d"
SPOT=Path("data/spot_led_volatility_catchup_sources_2023_2026/spot_hourly.csv.gz");SPOT_SHA="94d398e948f1283321e3784abc2a2d5dfff7a2331ff88c9720754a5720d2c6fc";SPOT_MANIFEST=SPOT.parent/"manifest.json";SPOT_MANIFEST_SHA="3bbd05162f66e486d3c67c217f48d3886de02124005c57c30fd0c42cc5ba2365"
PERP=Path("data/options_oi_chase_exhaustion_sources_2023_2026/btc_completed_hour.csv.gz");PERP_SHA="f075a882b80fc1d050aacd9abd417d4be6b6511c4307e39c98ef25f08822c496";PERP_MANIFEST=PERP.parent/"manifest.json";PERP_MANIFEST_SHA="3e350d16da72da7b60d9e91fbfb1ff4c2e13e5cb954b52b19ceaddf8c4f0e66d"
SOURCE_DIR=Path("data/spot_perpetual_variance_transfer_asymmetry_sources_2023_2026");FEATURES=SOURCE_DIR/"preentry_features.csv.gz";SOURCE_MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/spot_perpetual_variance_transfer_asymmetry_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/spot_perpetual_variance_transfer_asymmetry_controls_2023_2026");RESULT=Path("results/spot_perpetual_variance_transfer_asymmetry_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_volatility_gate","no_relocation_tail","no_opposite_half_geometry","spot_direction_only","one_block_stale_relocation","direction_flip")
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","spot_first_variation","spot_second_variation","perp_first_variation","perp_second_variation","spot_second_share","perp_second_share","relocation","relocation_rank","spot_final2_return","perp_final2_return","btc_realized_variation","btc_realized_variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def strict_prior_midrank(v:pd.Series,lookback:int=90,minimum:int=60)->pd.Series:
 n=pd.to_numeric(v,errors="coerce").astype(float);o=pd.Series(np.nan,index=n.index,dtype=float);h=[]
 for i,c in n.items():
  q=h[-lookback:]
  if math.isfinite(c) and len(q)>=minimum:
   a=np.asarray(q);o.at[i]=(np.sum(a<c)+.5*np.sum(a==c))/len(a)
  if math.isfinite(c):h.append(c)
 return o
def build_features()->pd.DataFrame:
 if sha(SPOT)!=SPOT_SHA or sha(SPOT_MANIFEST)!=SPOT_MANIFEST_SHA or sha(PERP)!=PERP_SHA or sha(PERP_MANIFEST)!=PERP_MANIFEST_SHA:raise RuntimeError("SPVTA source drift")
 s=pd.read_csv(SPOT,compression="gzip");s["hour_start"]=pd.to_datetime(s.hour_start,utc=True);s["spot_open"]=pd.to_numeric(s.hour_open,errors="coerce");s["spot_close"]=pd.to_numeric(s.hour_close,errors="coerce");s["spot_valid"]=s.spot_valid.astype(str).str.lower().eq("true")&np.isfinite(s[["spot_open","spot_close"]]).all(axis=1)&s[["spot_open","spot_close"]].gt(0).all(axis=1);s["spot_return"]=np.log(s.spot_close/s.spot_open)
 p=pd.read_csv(PERP,compression="gzip");p["hour_start"]=pd.to_datetime(p.hour_start,utc=True);p["perp_open"]=pd.to_numeric(p.open,errors="coerce");p["perp_close"]=pd.to_numeric(p.close,errors="coerce");p["perp_valid"]=p.source_valid.astype(str).str.lower().eq("true")&np.isfinite(p[["perp_open","perp_close"]]).all(axis=1)&p[["perp_open","perp_close"]].gt(0).all(axis=1);p["perp_return"]=np.log(p.perp_close/p.perp_open);p=p.sort_values("hour_start").reset_index(drop=True);consecutive=p.hour_start.diff().eq(pd.Timedelta(hours=1));p["btc_realized_variation"]=np.sqrt(p.perp_return.pow(2).rolling(24,min_periods=24).sum());p["btc_valid"]=p.perp_valid.rolling(24,min_periods=24).sum().eq(24)&consecutive.rolling(23,min_periods=23).sum().eq(23)&np.isfinite(p.btc_realized_variation)
 h=s[["hour_start","spot_return","spot_valid"]].merge(p[["hour_start","perp_return","perp_valid","btc_realized_variation","btc_valid"]],on="hour_start",how="inner",validate="one_to_one").sort_values("hour_start");h["decision_time"]=h.hour_start.dt.floor("8h")+pd.Timedelta(hours=8);rows=[]
 for decision,g in h.groupby("decision_time",sort=True):
  g=g.sort_values("hour_start");expected=pd.date_range(pd.Timestamp(decision)-pd.Timedelta(hours=8),pd.Timestamp(decision),freq="1h",inclusive="left");valid=len(g)==8 and g.hour_start.reset_index(drop=True).equals(pd.Series(expected,name="hour_start")) and bool(g.spot_valid.all() and g.perp_valid.all())
  if not valid:continue
  sr=g.spot_return.to_numpy(float);pr=g.perp_return.to_numpy(float);sf=float(np.square(sr[:4]).sum());ss=float(np.square(sr[4:]).sum());pf=float(np.square(pr[:4]).sum());ps=float(np.square(pr[4:]).sum());st=sf+ss;pt=pf+ps
  rows.append({"decision_time":decision,"block_valid":st>0 and pt>0,"spot_first_variation":sf,"spot_second_variation":ss,"perp_first_variation":pf,"perp_second_variation":ps,"spot_second_share":ss/st if st>0 else np.nan,"perp_second_share":ps/pt if pt>0 else np.nan,"spot_final2_return":float(sr[-2:].sum()),"perp_final2_return":float(pr[-2:].sum()),"btc_realized_variation":float(g.btc_realized_variation.iloc[-1]),"btc_valid":bool(g.btc_valid.iloc[-1])})
 f=pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True);f["relocation"]=f.spot_second_share-f.perp_second_share;f["relocation_rank"]=strict_prior_midrank(f.relocation.where(f.block_valid));f["btc_realized_variation_rank"]=strict_prior_midrank(f.btc_realized_variation.where(f.btc_valid));f["signal_valid"]=f.block_valid&f.btc_valid&np.isfinite(f[["relocation","relocation_rank","spot_final2_return","perp_final2_return","btc_realized_variation","btc_realized_variation_rank"]]).all(axis=1);return f
def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 relocation=f.relocation;rank=f.relocation_rank;spot_share=f.spot_second_share;perp_share=f.perp_second_share
 if control=="one_block_stale_relocation":relocation=relocation.shift(1);rank=rank.shift(1);spot_share=spot_share.shift(1);perp_share=perp_share.shift(1)
 positive=relocation.gt(0);tail=pd.Series(True,index=f.index) if control=="no_relocation_tail" else rank.ge(.65);geometry=pd.Series(True,index=f.index) if control=="no_opposite_half_geometry" else spot_share.gt(.5)&perp_share.le(.5);direction=f.spot_final2_return.ne(0)
 if control!="spot_direction_only":direction&=f.perp_final2_return.ne(0)&np.sign(f.spot_final2_return).eq(np.sign(f.perp_final2_return))
 volatile=pd.Series(True,index=f.index) if control=="no_volatility_gate" else f.btc_realized_variation_rank.ge(.65);active=f.signal_valid&np.isfinite(relocation)&positive&tail&geometry&direction&volatile;side=np.sign(f.spot_final2_return);side=-side if control=="direction_flip" else side;return active,side
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  next_allowed=exit_;row={"candidate":"SPVTA-8","control":control,"split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i])};row.update({c:float(f.at[i,c]) for c in COLUMNS[8:]});rows.append(row)
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("SPVTA preregistration hash drift")
 f=build_features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);CLOCK.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(f,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"spvta_8_preentry_sources_v1","spot":{"path":str(SPOT),"sha256":SPOT_SHA,"manifest_path":str(SPOT_MANIFEST),"manifest_sha256":SPOT_MANIFEST_SHA},"perpetual":{"path":str(PERP),"sha256":PERP_SHA,"manifest_path":str(PERP_MANIFEST),"manifest_sha256":PERP_MANIFEST_SHA},"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(f)},"candidate_incidence_opened":False,"postentry_outcomes_opened":False,"no_imputation":True};sm={**sc,"manifest_hash":chash(sc)};SOURCE_MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());passed=all(checks.values());core={"protocol_version":"spvta_8_source_support_v1","policy_id":"SPVTA-8","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(SOURCE_MANIFEST),"sha256":sha(SOURCE_MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
