"""Materialize source-only KPAR-12 support clocks."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_kimchi_premium_acceleration_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");SOURCE_DIR=Path("data/kimchi_premium_acceleration_relay_sources_2023_2026");PANEL=SOURCE_DIR/"hourly_cross_venue_panel.csv.gz";SOURCE_MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/kimchi_premium_acceleration_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/kimchi_premium_acceleration_relay_controls_2023_2026");RESULT=Path("results/kimchi_premium_acceleration_relay_support_2026-08-08.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),pd.Timestamp("2026-08-01T00:00:00Z"))};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_volatility_gate","no_premium_tail","premium_level","one_day_stale_acceleration","direction_flip");COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","kimchi_premium","premium_change","premium_change_z","btc_realized_variation","btc_realized_variation_rank")
def query(table:str)->str:return f"SELECT date_bin('1 hour',ts,TIMESTAMPTZ '1970-01-01') AS hour_start,(array_agg(open ORDER BY ts))[1] AS hour_open,(array_agg(close ORDER BY ts DESC))[1] AS hour_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_timestamps,min(ts) AS first_ts,max(ts) AS last_ts FROM {table} WHERE symbol=:symbol AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def causal_z(v:pd.Series,lookback:int=90,minimum:int=60)->pd.Series:
 x=pd.to_numeric(v,errors="coerce").to_numpy(float);o=np.full(len(x),np.nan)
 for i,current in enumerate(x):
  p=x[max(0,i-lookback):i];p=p[np.isfinite(p)]
  if np.isfinite(current) and len(p)>=minimum:
   s=float(np.std(p,ddof=1));o[i]=(current-float(np.mean(p)))/s if s>0 else np.nan
 return pd.Series(o,index=v.index)
def strict_prior_midrank(v:pd.Series,lookback:int=90,minimum:int=60)->pd.Series:
 n=pd.to_numeric(v,errors="coerce").astype(float);o=pd.Series(np.nan,index=n.index,dtype=float);h=[]
 for i,x in n.items():
  p=h[-lookback:]
  if math.isfinite(x) and len(p)>=minimum:
   a=np.asarray(p);o.at[i]=(np.sum(a<x)+.5*np.sum(a==x))/len(a)
  if math.isfinite(x):h.append(x)
 return o
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_hourly(connection,table,symbol,prefix,minimum,first_minutes,last_minute):
 from sqlalchemy import text
 f=pd.read_sql_query(text(query(table)),connection,params={"symbol":symbol,"start":START.to_pydatetime(),"end":END.to_pydatetime()});f["hour_start"]=pd.to_datetime(f.hour_start,utc=True);f["first_ts"]=pd.to_datetime(f.first_ts,utc=True);f["last_ts"]=pd.to_datetime(f.last_ts,utc=True)
 for c in ("hour_open","hour_close"):f[c]=pd.to_numeric(f[c],errors="coerce")
 f[f"{prefix}_valid"]=f.distinct_timestamps.ge(minimum)&f.first_ts.le(f.hour_start+pd.Timedelta(minutes=first_minutes))&f.last_ts.ge(f.hour_start+pd.Timedelta(minutes=last_minute))&np.isfinite(f[["hour_open","hour_close"]]).all(axis=1)&f[["hour_open","hour_close"]].gt(0).all(axis=1)
 return f.rename(columns={"hour_open":f"{prefix}_open","hour_close":f"{prefix}_close"})[["hour_start",f"{prefix}_open",f"{prefix}_close",f"{prefix}_valid"]]
def materialize()->dict:
 db=postgres_engine()
 with db.connect() as c:
  u=load_hourly(c,"bars_upbit","KRW-BTC","upbit",30,5,55);b=load_hourly(c,"bars_binance","BTCUSDT","binance",55,1,59);x=load_hourly(c,"bars_polygon","USDKRW","fx",40,10,55)
 db.dispose();idx=pd.date_range(START,END-pd.Timedelta(hours=1),freq="1h");f=u.merge(b,on="hour_start",how="outer").merge(x,on="hour_start",how="outer").set_index("hour_start").reindex(idx).rename_axis("hour_start").reset_index();f["decision_time"]=f.hour_start+pd.Timedelta(hours=1)
 for c in ("upbit_valid","binance_valid","fx_valid"):f[c]=f[c].fillna(False).astype(bool)
 f["premium_valid"]=f.upbit_valid&f.binance_valid&f.fx_valid;f["kimchi_premium"]=(f.upbit_close/(f.binance_close*f.fx_close)-1).where(f.premium_valid);f["btc_hour_return"]=np.log(f.binance_close/f.binance_open).where(f.binance_valid);f["btc_realized_variation"]=np.sqrt(f.btc_hour_return.pow(2).rolling(6,min_periods=6).sum());f["btc_rv_valid"]=f.binance_valid.rolling(6,min_periods=6).sum().eq(6)&np.isfinite(f.btc_realized_variation)
 SOURCE_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(f,PANEL);core={"protocol_version":"kpar_12_cross_venue_source_v1","queries":{"upbit":query("bars_upbit"),"binance":query("bars_binance"),"fx":query("bars_polygon")},"symbols":{"upbit":"KRW-BTC","binance":"BTCUSDT","fx":"USDKRW"},"window":[START.isoformat(),END.isoformat()],"outcomes_opened":False,"candidate_incidence_opened":False,"no_imputation":True,"output":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(f),"valid_premium_hours":int(f.premium_valid.sum())}};p={**core,"manifest_hash":chash(core)};SOURCE_MANIFEST.write_text(json.dumps(p,indent=2,allow_nan=False)+"\n");return p
def features()->pd.DataFrame:
 f=pd.read_csv(PANEL,compression="gzip");f["decision_time"]=pd.to_datetime(f.decision_time,utc=True);f["premium_valid"]=f.premium_valid.astype(str).str.lower().eq("true");f["btc_rv_valid"]=f.btc_rv_valid.astype(str).str.lower().eq("true")
 for c in ("kimchi_premium","btc_realized_variation"):f[c]=pd.to_numeric(f[c],errors="coerce")
 f["premium_change"]=f.kimchi_premium-f.kimchi_premium.shift(6);f["change_valid"]=f.premium_valid&f.premium_valid.shift(6,fill_value=False)
 d=f[f.decision_time.dt.hour.eq(0)&f.decision_time.dt.dayofweek.lt(5)].copy().reset_index(drop=True);d["premium_change_z"]=causal_z(d.premium_change.where(d.change_valid));d["premium_level_z"]=causal_z(d.kimchi_premium.where(d.premium_valid));d["btc_realized_variation_rank"]=strict_prior_midrank(d.btc_realized_variation.where(d.btc_rv_valid));d["signal_valid"]=d.change_valid&d.btc_rv_valid&np.isfinite(d[["kimchi_premium","premium_change","premium_change_z","btc_realized_variation","btc_realized_variation_rank"]]).all(axis=1)&d.premium_change.ne(0);return d
def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 change=f.kimchi_premium if control=="premium_level" else f.premium_change;z=f.premium_level_z if control=="premium_level" else f.premium_change_z
 if control=="one_day_stale_acceleration":change=change.shift(1);z=z.shift(1)
 tail=pd.Series(True,index=f.index) if control=="no_premium_tail" else z.abs().ge(1.);vol=pd.Series(True,index=f.index) if control=="no_volatility_gate" else f.btc_realized_variation_rank.ge(.65);active=f.signal_valid&np.isfinite(change)&np.isfinite(z)&change.ne(0)&tail&vol;side=np.sign(change);side=-side if control=="direction_flip" else side;return active,side
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(f,control);rows=[]
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12);split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  rows.append({"candidate":"KPAR-12","control":control,"split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"kimchi_premium":float(f.at[i,"kimchi_premium"]),"premium_change":float(f.at[i,"premium_change"]),"premium_change_z":float(f.at[i,"premium_change_z"]),"btc_realized_variation":float(f.at[i,"btc_realized_variation"]),"btc_realized_variation_rank":float(f.at[i,"btc_realized_variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict:
 sm=materialize();f=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());passed=all(checks.values());core={"protocol_version":"kpar_12_source_support_v1","policy_id":"KPAR-12","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(SOURCE_MANIFEST),"sha256":sha(SOURCE_MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};p={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(p,indent=2,allow_nan=False)+"\n");return p
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
