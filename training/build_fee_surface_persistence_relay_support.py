"""Build source-only FSPR-24 clocks before Gross9 or economic outcomes."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_fee_surface_persistence_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
PREREG_SHA="1c2d84b5a3951eccc90fcb86646cdc8f5d9ddae569415e522327f3369d6451d1";FEES=Path("data/mempool_block_feerates_3y_2026-07-20.csv.gz");FEES_SHA="007d13ba756fd29faae1ae87caa11554438b54bb5028f24b2f0c21ddf3a0e55d";PRICE=Path("data/options_oi_chase_exhaustion_sources_2023_2026/btc_completed_hour.csv.gz");PRICE_SHA="f075a882b80fc1d050aacd9abd417d4be6b6511c4307e39c98ef25f08822c496"
SOURCE_DIR=Path("data/fee_surface_persistence_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"preentry_features.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/fee_surface_persistence_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/fee_surface_persistence_relay_controls_2023_2026");RESULT=Path("results/fee_surface_persistence_relay_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("one_transition_only","three_of_five_breadth","no_volatility_gate","one_bucket_stale_surface","direction_flip");FEE_COLS=["fee_p10","fee_p25","fee_p50","fee_p75","fee_p90"]
COLUMNS=("candidate","control","split","bucket_start_utc","decision_time","feature_available_time","entry_time","exit_time","side","broad_sign","magnitude","magnitude_rank","btc_realized_variation","btc_variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def rank(v:pd.Series,lookback:int=180,minimum:int=90)->pd.Series:
 x=pd.to_numeric(v,errors="coerce").to_numpy(float);o=np.full(len(x),np.nan);h=[]
 for i,c in enumerate(x):
  q=h[-lookback:]
  if np.isfinite(c) and len(q)>=minimum:
   a=np.asarray(q);o[i]=(np.sum(a<c)+.5*np.sum(a==c))/len(a)
  if np.isfinite(c):h.append(c)
 return pd.Series(o,index=v.index)
def broad(values:np.ndarray,minimum:int=4)->float:
 s=np.sign(values[np.isfinite(values)]);s=s[s!=0]
 if (s==1).sum()>=minimum:return 1.
 if (s==-1).sum()>=minimum:return -1.
 return np.nan
def features()->pd.DataFrame:
 if sha(FEES)!=FEES_SHA or sha(PRICE)!=PRICE_SHA:raise RuntimeError("FSPR source drift")
 f=pd.read_csv(FEES,compression="gzip");
 for c in ("bucket_start_utc","bucket_end_utc","available_at_utc"):f[c]=pd.to_datetime(f[c],utc=True)
 if f.bucket_start_utc.duplicated().any() or not f.bucket_start_utc.diff().dropna().eq(pd.Timedelta(hours=12)).all() or not f.bucket_end_utc.eq(f.bucket_start_utc+pd.Timedelta(hours=12)).all() or not f.available_at_utc.eq(f.bucket_end_utc+pd.Timedelta(hours=48)).all():raise RuntimeError("FSPR fee clock drift")
 x=np.log1p(f[FEE_COLS].astype(float));d=x.diff();f["broad_sign_4"]=d.apply(lambda r:broad(r.to_numpy(),4),axis=1);f["broad_sign_3"]=d.apply(lambda r:broad(r.to_numpy(),3),axis=1);f["magnitude"]=d.median(axis=1).abs();f["magnitude_rank"]=rank(f.magnitude)
 p=pd.read_csv(PRICE,compression="gzip");p["hour_start"]=pd.to_datetime(p.hour_start,utc=True);p["decision_time"]=pd.to_datetime(p.decision_time,utc=True);p["open"]=pd.to_numeric(p.open,errors="coerce");p["close"]=pd.to_numeric(p.close,errors="coerce");p["valid"]=p.source_valid.astype(str).str.lower().eq("true")&np.isfinite(p[["open","close"]]).all(axis=1)&p[["open","close"]].gt(0).all(axis=1);p["r2"]=np.log(p.close/p.open).pow(2);p=p.sort_values("decision_time");p["btc_realized_variation"]=np.sqrt(p.r2.rolling(24,min_periods=24).sum());p["btc_valid"]=p.valid.rolling(24,min_periods=24).sum().eq(24)&p.hour_start.diff().eq(pd.Timedelta(hours=1)).rolling(23,min_periods=23).sum().eq(23);v=p[p.decision_time.dt.hour.isin([0,12])][["decision_time","btc_realized_variation","btc_valid"]].copy();v["btc_variation_rank"]=rank(v.btc_realized_variation.where(v.btc_valid));f=f.merge(v,left_on="available_at_utc",right_on="decision_time",how="left",validate="one_to_one");f["persistent_4"]=f.broad_sign_4.notna()&f.broad_sign_4.shift(1).eq(f.broad_sign_4);f["persistent_3"]=f.broad_sign_3.notna()&f.broad_sign_3.shift(1).eq(f.broad_sign_3);return f
def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 sign=f.broad_sign_3 if control=="three_of_five_breadth" else f.broad_sign_4;persist=f.persistent_3 if control=="three_of_five_breadth" else f.persistent_4;mag=f.magnitude_rank
 if control=="one_transition_only":persist=sign.notna()
 if control=="one_bucket_stale_surface":sign=sign.shift(1);persist=persist.shift(1,fill_value=False);mag=mag.shift(1)
 vol=pd.Series(True,index=f.index) if control=="no_volatility_gate" else f.btc_variation_rank.ge(.65);active=persist&sign.notna()&mag.ge(.70)&f.btc_valid.fillna(False)&vol;side=sign.copy();side=-side if control=="direction_flip" else side;return active,side
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"available_at_utc"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=24)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  next_allowed=exit_;rows.append({"candidate":"FSPR-24","control":control,"split":split,"bucket_start_utc":f.at[i,"bucket_start_utc"],"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"broad_sign":float(f.at[i,"broad_sign_4"]),"magnitude":float(f.at[i,"magnitude"]),"magnitude_rank":float(f.at[i,"magnitude_rank"]),"btc_realized_variation":float(f.at[i,"btc_realized_variation"]),"btc_variation_rank":float(f.at[i,"btc_variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict[str,Any]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("FSPR prereg drift")
 f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(f,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 mc={"protocol_version":"fspr_24_preentry_source_v1","fees":{"path":str(FEES),"sha256":FEES_SHA},"completed_btc":{"path":str(PRICE),"sha256":PRICE_SHA},"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(f)},"postentry_outcomes_opened":False,"gross9_rows_opened":False};m={**mc,"manifest_hash":chash(mc)};MANIFEST.write_text(json.dumps(m,indent=2)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=MINIMUM[n]),(f"{n}_side_balance",x["minority_side_share"]>=.2),(f"{n}_month_concentration",x["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"fspr_24_source_support_v1","policy_id":"FSPR-24","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":m["manifest_hash"]},"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
