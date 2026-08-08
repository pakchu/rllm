"""Materialize hourly participation and build outcome-blind SPVIR-6 support."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from preprocessing.live_db_features import postgres_url_from_env
from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import preregister_spot_participation_volatility_ignition_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
START=pd.Timestamp("2023-06-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");ENV_FILE="/home/pakchu/rllm/.env"
SOURCE_DIR=Path("data/spot_participation_volatility_ignition_sources_2023_2026");PANEL=SOURCE_DIR/"hourly_panel.csv.gz";SOURCE_MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/spot_participation_volatility_ignition_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/spot_participation_volatility_ignition_relay_controls_2023_2026");RESULT=Path("results/spot_participation_volatility_ignition_relay_support_2026-08-08.json")
SPLITS=base.SPLITS;MINIMUM_EVENTS={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_joint_expansion","no_participation_tail","no_direction_agreement","one_hour_stale_participation","direction_flip")
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","spot_return","perpetual_return","spot_participation","prior_participation_q75","bvol_body","dvol_body")
def query(table:str)->str:return f"SELECT date_bin('1 hour',ts,TIMESTAMPTZ '1970-01-01') AS hour_start,(array_agg(open ORDER BY ts))[1] AS hour_open,(array_agg(close ORDER BY ts DESC))[1] AS hour_close,sum(volume) AS base_volume,count(*) AS source_rows,count(DISTINCT ts) AS distinct_timestamps,min(ts) AS first_ts,max(ts) AS last_ts FROM {table} WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def normalize(f:pd.DataFrame,prefix:str)->pd.DataFrame:
 f=f.copy()
 for c in ("hour_start","first_ts","last_ts"):f[c]=pd.to_datetime(f[c],utc=True,format="mixed")
 for c in ("hour_open","hour_close","base_volume"):f[c]=pd.to_numeric(f[c],errors="coerce")
 if f.hour_start.duplicated().any() or not f.hour_start.is_monotonic_increasing:raise RuntimeError(f"SPVIR {prefix} clock invalid")
 grid=pd.DataFrame({"hour_start":pd.date_range(START,END,freq="1h",inclusive="left")});f=grid.merge(f,on="hour_start",how="left",validate="one_to_one");f[["source_rows","distinct_timestamps"]]=f[["source_rows","distinct_timestamps"]].fillna(0).astype(int)
 f[f"{prefix}_valid"]=f.source_rows.eq(60)&f.distinct_timestamps.eq(60)&f.first_ts.eq(f.hour_start)&f.last_ts.eq(f.hour_start+pd.Timedelta(minutes=59))&np.isfinite(f[["hour_open","hour_close","base_volume"]]).all(axis=1)&f[["hour_open","hour_close","base_volume"]].gt(0).all(axis=1)
 return f.rename(columns={"hour_open":f"{prefix}_open","hour_close":f"{prefix}_close","base_volume":f"{prefix}_volume"})[["hour_start",f"{prefix}_open",f"{prefix}_close",f"{prefix}_volume",f"{prefix}_valid"]]
def query_panel(env_file:str=ENV_FILE)->pd.DataFrame:
 from sqlalchemy import create_engine,text
 e=create_engine(postgres_url_from_env(env_file),connect_args={"connect_timeout":10})
 try:
  with e.connect() as c:
   spot=pd.read_sql_query(text(query("bars_binance_spot")),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()});perp=pd.read_sql_query(text(query("bars_binance")),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:e.dispose()
 f=normalize(spot,"spot").merge(normalize(perp,"perpetual"),on="hour_start",validate="one_to_one");f["decision_time"]=f.hour_start+pd.Timedelta(hours=1);f["spot_return"]=f.spot_close/f.spot_open-1;f["perpetual_return"]=f.perpetual_close/f.perpetual_open-1;den=f.spot_volume+f.perpetual_volume;f["participation_valid"]=f.spot_valid&f.perpetual_valid&den.gt(0)&np.isfinite(den);f["spot_participation"]=f.spot_volume/den;f["prior_participation_q75"]=f.spot_participation.where(f.participation_valid).shift(1).rolling(720,min_periods=672).quantile(.75);return f
def volatility()->pd.DataFrame:
 b=pd.read_csv(base.SOURCE_DIR/"bvol_hourly.csv.gz",compression="gzip");d=pd.read_csv(base.SOURCE_DIR/"dvol_hourly.csv.gz",compression="gzip");x=pd.DataFrame({"decision_time":pd.to_datetime(b["feature_available_time_utc"],utc=True,format="mixed"),"bvol_open":pd.to_numeric(b["open"],errors="coerce"),"bvol_close":pd.to_numeric(b["close"],errors="coerce"),"bvol_valid":b["feature_valid"].astype(str).str.lower().eq("true")});y=pd.DataFrame({"decision_time":pd.to_datetime(d["close_time"],utc=True,format="mixed"),"dvol_open":pd.to_numeric(d["open"],errors="coerce"),"dvol_close":pd.to_numeric(d["close"],errors="coerce")});x=x.merge(y,on="decision_time",validate="one_to_one");n=["bvol_open","bvol_close","dvol_open","dvol_close"];x["vol_valid"]=x.bvol_valid&np.isfinite(x[n]).all(axis=1)&x[n].gt(0).all(axis=1);x["bvol_body"]=x.bvol_close/x.bvol_open-1;x["dvol_body"]=x.dvol_close/x.dvol_open-1;return x
def features(panel:pd.DataFrame)->pd.DataFrame:
 f=panel.merge(volatility(),on="decision_time",validate="one_to_one");f["signal_valid"]=f.participation_valid&f.vol_valid&np.isfinite(f[["spot_return","perpetual_return","spot_participation"]]).all(axis=1)&f.spot_return.ne(0)&f.perpetual_return.ne(0);return f.sort_values("decision_time").reset_index(drop=True)
def conditions(f:pd.DataFrame,control:str)->pd.Series:
 vol=pd.Series(True,index=f.index) if control=="no_joint_expansion" else f.bvol_body.gt(0)&f.dvol_body.gt(0);agree=f.spot_return.ne(0)&f.perpetual_return.ne(0)
 if control!="no_direction_agreement":agree&=np.sign(f.spot_return).eq(np.sign(f.perpetual_return))
 if control=="no_participation_tail":part=pd.Series(True,index=f.index)
 elif control=="one_hour_stale_participation":part=f.spot_participation.shift(1).ge(f.prior_participation_q75.shift(1))
 else:part=f.prior_participation_q75.notna()&f.spot_participation.ge(f.prior_participation_q75)
 stale=f.signal_valid.shift(1,fill_value=False) if control=="one_hour_stale_participation" else pd.Series(True,index=f.index);active=f.signal_valid&stale&vol&agree&part;return active&~active.shift(1,fill_value=False)&f.signal_valid.shift(1,fill_value=False)&f.decision_time.diff().eq(pd.Timedelta(hours=1))
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 on=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[on]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  side=int(np.sign(f.at[i,"spot_return"]));side=-side if control=="direction_flip" else side;next_allowed=exit_;rows.append({"candidate":"SPVIR-6","control":control,"split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":side,"spot_return":float(f.at[i,"spot_return"]),"perpetual_return":float(f.at[i,"perpetual_return"]),"spot_participation":float(f.at[i,"spot_participation"]),"prior_participation_q75":float(f.at[i,"prior_participation_q75"]),"bvol_body":float(f.at[i,"bvol_body"]),"dvol_body":float(f.at[i,"dvol_body"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def materialize(f:pd.DataFrame)->dict:
 SOURCE_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(f,PANEL);core={"protocol_version":"spvir_6_hourly_source_v1","queries":{"spot":query("bars_binance_spot"),"perpetual":query("bars_binance")},"window":[START.isoformat(),END.isoformat()],"outcomes_opened":False,"candidate_incidence_opened":False,"output":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(f),"valid_rows":int(f.participation_valid.sum())}};r={**core,"manifest_hash":chash(core)};SOURCE_MANIFEST.write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n");return r
def run(env_file:str=ENV_FILE)->dict:
 panel=query_panel(env_file);sm=materialize(panel);f=features(panel);primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM_EVENTS[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());vm=base.SOURCE_DIR/"manifest.json";passed=all(checks.values());core={"protocol_version":"spvir_6_source_support_v1","policy_id":"SPVIR-6","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifests":{"hourly_participation":{"path":str(SOURCE_MANIFEST),"sha256":sha(SOURCE_MANIFEST),"manifest_hash":sm["manifest_hash"]},"volatility":{"path":str(vm),"sha256":sha(vm)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":p=argparse.ArgumentParser();p.add_argument("--env-file",default=ENV_FILE);a=p.parse_args();r=run(a.env_file);print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
