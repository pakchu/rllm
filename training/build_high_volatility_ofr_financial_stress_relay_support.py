"""Materialize outcome-blind source support for frozen HVOFSR-24."""
from __future__ import annotations
import argparse,hashlib,io,json
from pathlib import Path
from typing import Any
from urllib.request import Request,urlopen
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from training import preregister_high_volatility_ofr_financial_stress_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE="/home/pakchu/rllm/.env";BUILDER=Path("training/build_high_volatility_ofr_financial_stress_relay_support.py");PREREG_SHA="9991b178a2e5c35e4e2e8a582c8930d3f9d26270ea8db91efd11fc0aa9491763"
ROOT=Path("data/high_volatility_ofr_financial_stress_relay_sources_2021_2026");RAW=ROOT/"ofr_fsi.csv";PANEL=ROOT/"ofr_fsi_panel.csv.gz";BTC_SOURCE=ROOT/"btc_1m_ts_open_close.csv.gz";FEATURES=ROOT/"hvofsr_preentry_features.csv.gz";MANIFEST=ROOT/"manifest.json"
CLOCK=Path("data/high_volatility_ofr_financial_stress_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_ofr_financial_stress_relay_controls_2023_2026");RESULT=Path("results/high_volatility_ofr_financial_stress_relay_support_2026-08-12.json")
SOURCE_START=pd.Timestamp("2021-01-01");SOURCE_END=pd.Timestamp("2026-08-01");BTC_START=pd.Timestamp("2021-01-01T22:00Z");BTC_END=pd.Timestamp("2026-08-01T22:01Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),pd.Timestamp("2026-08-01T00:00Z"))};MINIMUM={"train":8,"test":12,"eval":12,"final":8}
CONTROLS=("no_btc_volatility_gate","no_stress_change_tail","one_observation_stale_change","fsi_level_sign","direction_flip","same_clock_forced_long")
COLUMNS=("candidate","control","split","source_day","publication_proxy_day","decision_time","entry_time","exit_time","side","fsi_level","fsi_change","stress_change_rank","btc_variation","btc_variation_rank")
QUERY="SELECT ts,open,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
EXPECTED_COLUMNS=("Date","OFR FSI","Credit","Equity valuation","Safe assets","Funding","Volatility","United States","Other advanced economies","Emerging markets")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def prior_rank(v:pd.Series,lookback:int,minimum:int)->pd.Series:
 x=pd.to_numeric(v,errors="coerce");o=pd.Series(np.nan,index=x.index,dtype=float);h=[]
 for i,c in x.items():
  q=np.asarray(h[-lookback:],float)
  if np.isfinite(c) and len(q)>=minimum:o.at[i]=((q<c).sum()+.5*(q==c).sum())/len(q)
  if np.isfinite(c):h.append(float(c))
 return o
def normalize_ofr(payload:bytes)->pd.DataFrame:
 try:x=pd.read_csv(io.BytesIO(payload))
 except Exception as e:raise RuntimeError("HVOFSR OFR CSV parse failure") from e
 if tuple(x.columns)!=EXPECTED_COLUMNS:raise RuntimeError(f"HVOFSR OFR schema drift: {tuple(x.columns)}")
 x=x.copy();x["source_day"]=pd.to_datetime(x["Date"],format="%Y-%m-%d",errors="raise");x["fsi_level"]=pd.to_numeric(x["OFR FSI"],errors="coerce")
 x=x.loc[x.source_day.ge(SOURCE_START)&x.source_day.lt(SOURCE_END)].sort_values("source_day").reset_index(drop=True)
 if x.empty or x.source_day.duplicated().any() or not x.source_day.is_monotonic_increasing:raise RuntimeError("HVOFSR OFR dates invalid")
 if x.source_day.dt.dayofweek.ge(5).any() or x.source_day.diff().dt.days.dropna().gt(14).any():raise RuntimeError("HVOFSR OFR business-date sequence drift")
 if not np.isfinite(x.fsi_level.to_numpy(float)).all():raise RuntimeError("HVOFSR OFR total index incomplete")
 return x[[*EXPECTED_COLUMNS,"source_day","fsi_level"]]
def download_ofr()->tuple[bytes,pd.DataFrame]:
 with urlopen(Request(prereg.SOURCE_URL,headers={"User-Agent":"Mozilla/5.0"}),timeout=60) as r:payload=r.read()
 return payload,normalize_ofr(payload)
def normalize_btc(raw:pd.DataFrame)->pd.DataFrame:
 if raw.columns.tolist()!=["ts","open","close"]:raise RuntimeError("HVOFSR BTC schema drift")
 x=raw.copy();x.ts=pd.to_datetime(x.ts,utc=True,errors="raise");x=x.sort_values("ts").reset_index(drop=True);expected=pd.date_range(BTC_START,BTC_END,freq="1min",inclusive="left")
 if len(x)!=len(expected) or not x.ts.equals(pd.Series(expected,name="ts")):raise RuntimeError("HVOFSR BTC source is not the exact requested 1m grid")
 for c in ("open","close"):x[c]=pd.to_numeric(x[c],errors="coerce")
 if not np.isfinite(x[["open","close"]].to_numpy(float)).all() or not x[["open","close"]].gt(0).all(axis=None):raise RuntimeError("HVOFSR BTC prices invalid")
 return x
def load_btc(env_file:str=ENV_FILE)->pd.DataFrame:
 from sqlalchemy import create_engine,text
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(env_file);e=create_engine(postgres_url_from_env(env_file),connect_args={"connect_timeout":10})
 try:x=pd.read_sql_query(text(QUERY),e,params={"start":BTC_START.to_pydatetime(),"end":BTC_END.to_pydatetime()})
 finally:e.dispose()
 return normalize_btc(x)
def build_features(ofr:pd.DataFrame,bars:pd.DataFrame)->pd.DataFrame:
 x=ofr[["source_day","fsi_level"]].copy();x["fsi_change"]=x.fsi_level.diff();x["stress_change_rank"]=prior_rank(x.fsi_change.abs(),252,126);x["publication_proxy_day"]=x.source_day.shift(-2);x["decision_time"]=pd.to_datetime(x.publication_proxy_day,utc=True)+pd.Timedelta(hours=22)
 b=bars.set_index("ts");vals=[]
 for decision in x.decision_time:
  if pd.isna(decision) or decision<BTC_START+pd.Timedelta(hours=24) or decision>=BTC_END:vals.append(np.nan);continue
  w=b.loc[(b.index>=decision-pd.Timedelta(hours=24))&(b.index<decision)];vals.append(float(np.sqrt(np.square(np.log(w.close.to_numpy(float)/w.open.to_numpy(float))).sum())) if len(w)==1440 else np.nan)
 x["btc_variation"]=vals;x["state_valid"]=x.fsi_change.notna()&x.fsi_change.ne(0)&np.isfinite(x[["fsi_change","stress_change_rank","btc_variation"]]).all(axis=1);x["btc_variation_rank"]=prior_rank(x.btc_variation.where(x.state_valid),270,180);return x
def signal(x:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=x.shift(1) if control=="one_observation_stale_change" else x;score=used.fsi_change.copy();valid=used.state_valid.eq(True)&score.ne(0)
 stress=pd.Series(True,index=x.index) if control=="no_stress_change_tail" else used.stress_change_rank.ge(.70);btc=pd.Series(True,index=x.index) if control=="no_btc_volatility_gate" else x.btc_variation_rank.ge(.65);active=valid&stress&btc
 if control=="fsi_level_sign":score=used.fsi_level;active=used.fsi_level.ne(0)&used.fsi_level.notna()&used.stress_change_rank.ge(.70)&btc
 side=-np.sign(score).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=x.index)
 active&=side.ne(0);return active.fillna(False),side
def build_clock(x:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=signal(x,control);rows=[];next_allowed=None
 for i in x.index[active]:
  decision=pd.Timestamp(x.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=24)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  next_allowed=exit_;used=x.loc[i-1] if control=="one_observation_stale_change" else x.loc[i]
  rows.append({"candidate":"HVOFSR-24","control":control,"split":split,"source_day":used.source_day,"publication_proxy_day":x.at[i,"publication_proxy_day"],"decision_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"fsi_level":float(used.fsi_level),"fsi_change":float(used.fsi_change),"stress_change_rank":float(used.stress_change_rank),"btc_variation":float(x.at[i,"btc_variation"]),"btc_variation_rank":float(x.at[i,"btc_variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict[str,Any]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run(env_file:str=ENV_FILE)->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVOFSR preregistration hash drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);payload,ofr=download_ofr();bars=load_btc(env_file);features=build_features(ofr,bars);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};ROOT.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);RAW.write_bytes(payload);_write_gzip_csv(ofr,PANEL);_write_gzip_csv(bars,BTC_SOURCE);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 source_core={"protocol_version":"hvofsr_24_sources_v1","preregistration_sha256":PREREG_SHA,"official_url":prereg.SOURCE_URL,"raw":{"path":str(RAW),"sha256":sha(RAW)},"ofr_panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(ofr)},"btc":{"path":str(BTC_SOURCE),"sha256":sha(BTC_SOURCE),"query":QUERY,"rows":len(bars)},"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features)},"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"outcomes_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};MANIFEST.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
 support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,z in support.items() for k,v in ((f"{n}_minimum_events",z["events"]>=MINIMUM[n]),(f"{n}_side_balance",z["minority_side_share"]>=.2),(f"{n}_month_concentration",z["max_month_share"]<=.45))};passed=all(checks.values());core={"protocol_version":"hvofsr_24_source_support_v1","policy_id":"HVOFSR-24","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--env-file",default=ENV_FILE);a=p.parse_args();r=run(a.env_file);print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
