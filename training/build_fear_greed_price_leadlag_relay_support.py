"""Materialize source-only FGPLR-24 support clocks."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_fear_greed_price_leadlag_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
PREREG_SHA="eb729ad2f5e8d1e5153bb7db1fb528ac582211e4e724a25db6d99e96f0025073"
SENTIMENT=Path("data/fear_greed_extremity_reversal_sources_2023_2026/fear_greed_daily.csv.gz");SENTIMENT_SHA="a50769db6ca15b9cbb538b4f03fd71956a42a3ca418a7628d8ba0c63d0b8f1dd";SENTIMENT_MANIFEST=SENTIMENT.parent/"manifest.json";SENTIMENT_MANIFEST_SHA="afa0f674838270d1945b83278478d2111928f181de07ae8d020ed1c4bc406302"
PRICE=Path("data/options_oi_chase_exhaustion_sources_2023_2026/btc_completed_hour.csv.gz");PRICE_SHA="f075a882b80fc1d050aacd9abd417d4be6b6511c4307e39c98ef25f08822c496";PRICE_MANIFEST=PRICE.parent/"manifest.json";PRICE_MANIFEST_SHA="3e350d16da72da7b60d9e91fbfb1ff4c2e13e5cb954b52b19ceaddf8c4f0e66d"
SOURCE_DIR=Path("data/fear_greed_price_leadlag_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"preentry_features.csv.gz";SOURCE_MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/fear_greed_price_leadlag_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/fear_greed_price_leadlag_relay_controls_2023_2026");RESULT=Path("results/fear_greed_price_leadlag_relay_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_volatility_gate","no_sentiment_change_tail","no_direction_disagreement","sentiment_direction","one_day_stale_sentiment_change","direction_flip")
COLUMNS=("candidate","control","split","sentiment_date","decision_time","feature_available_time","entry_time","exit_time","side","fear_greed_value","sentiment_change","sentiment_change_rank","btc_day_return","btc_realized_variation","btc_variation_rank")
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
def features()->pd.DataFrame:
 if sha(SENTIMENT)!=SENTIMENT_SHA or sha(SENTIMENT_MANIFEST)!=SENTIMENT_MANIFEST_SHA or sha(PRICE)!=PRICE_SHA or sha(PRICE_MANIFEST)!=PRICE_MANIFEST_SHA:raise RuntimeError("FGPLR source drift")
 s=pd.read_csv(SENTIMENT,compression="gzip");s["sentiment_date"]=pd.to_datetime(s.sentiment_date,utc=True);s["fear_greed_value"]=pd.to_numeric(s.fear_greed_value,errors="coerce");s=s.sort_values("sentiment_date").reset_index(drop=True);consecutive=s.sentiment_date.diff().eq(pd.Timedelta(days=1));s["sentiment_change"]=(s.fear_greed_value-s.fear_greed_value.shift(1)).where(consecutive);s["sentiment_change_rank"]=strict_prior_midrank(s.sentiment_change.abs());s["decision_time"]=s.sentiment_date+pd.Timedelta(days=1)
 p=pd.read_csv(PRICE,compression="gzip");p["hour_start"]=pd.to_datetime(p.hour_start,utc=True);p["open"]=pd.to_numeric(p.open,errors="coerce");p["close"]=pd.to_numeric(p.close,errors="coerce");p["valid"]=p.source_valid.astype(str).str.lower().eq("true")&np.isfinite(p[["open","close"]]).all(axis=1)&p[["open","close"]].gt(0).all(axis=1);p["hour_return"]=np.log(p.close/p.open);p["sentiment_date"]=p.hour_start.dt.floor("D");g=p.groupby("sentiment_date",as_index=False).agg(hours=("hour_start","size"),first=("hour_start","min"),last=("hour_start","max"),valid=("valid","all"),btc_day_return=("hour_return","sum"),squared=("hour_return",lambda x:float(np.square(x).sum())));g["btc_valid"]=g.hours.eq(24)&g.first.eq(g.sentiment_date)&g.last.eq(g.sentiment_date+pd.Timedelta(hours=23))&g.valid&np.isfinite(g[["btc_day_return","squared"]]).all(axis=1);g["btc_realized_variation"]=np.sqrt(g.squared);g["decision_time"]=g.sentiment_date+pd.Timedelta(days=1);g["btc_variation_rank"]=strict_prior_midrank(g.btc_realized_variation.where(g.btc_valid))
 f=s.merge(g[["sentiment_date","decision_time","btc_valid","btc_day_return","btc_realized_variation","btc_variation_rank"]],on=["sentiment_date","decision_time"],how="inner",validate="one_to_one");f["signal_valid"]=f.btc_valid&np.isfinite(f[["sentiment_change","sentiment_change_rank","btc_day_return","btc_realized_variation","btc_variation_rank"]]).all(axis=1)&f.sentiment_change.ne(0)&f.btc_day_return.ne(0);return f
def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 change=f.sentiment_change;rank=f.sentiment_change_rank
 if control=="one_day_stale_sentiment_change":change=change.shift(1);rank=rank.shift(1)
 tail=pd.Series(True,index=f.index) if control=="no_sentiment_change_tail" else rank.ge(.60);disagree=pd.Series(True,index=f.index) if control=="no_direction_disagreement" else np.sign(change).eq(-np.sign(f.btc_day_return));volatile=pd.Series(True,index=f.index) if control=="no_volatility_gate" else f.btc_variation_rank.ge(.65);active=f.signal_valid&np.isfinite(change)&change.ne(0)&tail&disagree&volatile;side=np.sign(change) if control=="sentiment_direction" else np.sign(f.btc_day_return);side=-side if control=="direction_flip" else side;return active,side
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=24)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  next_allowed=exit_;rows.append({"candidate":"FGPLR-24","control":control,"split":split,"sentiment_date":f.at[i,"sentiment_date"],"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"fear_greed_value":float(f.at[i,"fear_greed_value"]),"sentiment_change":float(f.at[i,"sentiment_change"]),"sentiment_change_rank":float(f.at[i,"sentiment_change_rank"]),"btc_day_return":float(f.at[i,"btc_day_return"]),"btc_realized_variation":float(f.at[i,"btc_realized_variation"]),"btc_variation_rank":float(f.at[i,"btc_variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("FGPLR preregistration hash drift")
 f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);CLOCK.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(f,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"fgplr_24_preentry_sources_v1","sentiment":{"path":str(SENTIMENT),"sha256":SENTIMENT_SHA,"manifest_path":str(SENTIMENT_MANIFEST),"manifest_sha256":SENTIMENT_MANIFEST_SHA},"completed_btc":{"path":str(PRICE),"sha256":PRICE_SHA,"manifest_path":str(PRICE_MANIFEST),"manifest_sha256":PRICE_MANIFEST_SHA},"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(f)},"candidate_incidence_opened":False,"postentry_outcomes_opened":False,"no_imputation":True};sm={**sc,"manifest_hash":chash(sc)};SOURCE_MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());passed=all(checks.values());core={"protocol_version":"fgplr_24_source_support_v1","policy_id":"FGPLR-24","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(SOURCE_MANIFEST),"sha256":sha(SOURCE_MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
