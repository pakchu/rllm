"""Materialize spot hours and build outcome-blind support for SLVCR-6."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from preprocessing.live_db_features import postgres_url_from_env
from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import build_options_led_intrahour_absorption_support as intrahour
from training import preregister_spot_led_volatility_catchup_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

START=pd.Timestamp("2023-06-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");ENV_FILE="/home/pakchu/rllm/.env"
SOURCE_DIR=Path("data/spot_led_volatility_catchup_sources_2023_2026");SPOT=SOURCE_DIR/"spot_hourly.csv.gz";SOURCE_MANIFEST=SOURCE_DIR/"manifest.json"
CLOCK=Path("data/spot_led_volatility_catchup_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/spot_led_volatility_catchup_relay_controls_2023_2026");RESULT=Path("results/spot_led_volatility_catchup_relay_support_2026-08-08.json")
SPLITS=base.SPLITS;MINIMUM_EVENTS={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_joint_expansion","no_spot_tail","no_direction_agreement","no_partial_transmission","direction_flip")
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","spot_return","perpetual_return","perpetual_to_spot_ratio","prior_abs_spot_q60","bvol_body","dvol_body")
QUERY="""SELECT date_bin('1 hour',ts,TIMESTAMPTZ '1970-01-01') AS hour_start,(array_agg(open ORDER BY ts))[1] AS hour_open,(array_agg(close ORDER BY ts DESC))[1] AS hour_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_timestamps,min(ts) AS first_ts,max(ts) AS last_ts FROM bars_binance_spot WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def query_spot(env_file:str=ENV_FILE)->pd.DataFrame:
 from sqlalchemy import create_engine,text
 e=create_engine(postgres_url_from_env(env_file),connect_args={"connect_timeout":10})
 try:
  with e.connect() as c:f=pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:e.dispose()
 return normalize_spot(f)
def normalize_spot(f:pd.DataFrame)->pd.DataFrame:
 f=f.copy()
 for c in ("hour_start","first_ts","last_ts"):f[c]=pd.to_datetime(f[c],utc=True,format="mixed")
 if f.hour_start.duplicated().any() or not f.hour_start.is_monotonic_increasing:raise RuntimeError("SLVCR spot hourly clock invalid")
 for c in ("hour_open","hour_close"):f[c]=pd.to_numeric(f[c],errors="coerce")
 grid=pd.DataFrame({"hour_start":pd.date_range(START,END,freq="1h",inclusive="left")});f=grid.merge(f,on="hour_start",how="left",validate="one_to_one")
 for c in ("source_rows","distinct_timestamps"):f[c]=f[c].fillna(0).astype(int)
 f["spot_valid"]=f.source_rows.eq(60)&f.distinct_timestamps.eq(60)&f.first_ts.eq(f.hour_start)&f.last_ts.eq(f.hour_start+pd.Timedelta(minutes=59))&np.isfinite(f[["hour_open","hour_close"]]).all(axis=1)&f[["hour_open","hour_close"]].gt(0).all(axis=1)
 f.loc[~f.spot_valid,["hour_open","hour_close"]]=np.nan;f["decision_time"]=f.hour_start+pd.Timedelta(hours=1);f["spot_return"]=f.hour_close/f.hour_open-1
 f["prior_abs_spot_q60"]=f.spot_return.abs().where(f.spot_valid).shift(1).rolling(720,min_periods=672).quantile(.60);return f
def volatility()->pd.DataFrame:
 b=pd.read_csv(base.SOURCE_DIR/"bvol_hourly.csv.gz",compression="gzip");d=pd.read_csv(base.SOURCE_DIR/"dvol_hourly.csv.gz",compression="gzip")
 x=pd.DataFrame({"decision_time":pd.to_datetime(b.feature_available_time_utc,utc=True,format="mixed"),"bvol_open":pd.to_numeric(b.open,errors="coerce"),"bvol_close":pd.to_numeric(b.close,errors="coerce"),"bvol_valid":b.feature_valid.astype(str).str.lower().eq("true")})
 y=pd.DataFrame({"decision_time":pd.to_datetime(d.close_time,utc=True,format="mixed"),"dvol_open":pd.to_numeric(d.open,errors="coerce"),"dvol_close":pd.to_numeric(d.close,errors="coerce")});x=x.merge(y,on="decision_time",validate="one_to_one")
 n=["bvol_open","bvol_close","dvol_open","dvol_close"];x["vol_valid"]=x.bvol_valid&np.isfinite(x[n]).all(axis=1)&x[n].gt(0).all(axis=1);x["bvol_body"]=x.bvol_close/x.bvol_open-1;x["dvol_body"]=x.dvol_close/x.dvol_open-1;return x
def features(spot:pd.DataFrame)->pd.DataFrame:
 p=pd.read_csv(intrahour.PRICE_DIR/"btc_intrahour_path.csv.gz",compression="gzip");p["decision_time"]=pd.to_datetime(p.decision_time,utc=True,format="mixed");p["perpetual_open"]=pd.to_numeric(p.hour_open,errors="coerce");p["perpetual_close"]=pd.to_numeric(p.hour_close,errors="coerce");p["perpetual_valid"]=p.source_valid.astype(str).str.lower().eq("true");p["perpetual_return"]=p.perpetual_close/p.perpetual_open-1
 f=spot.merge(p[["decision_time","perpetual_valid","perpetual_return"]],on="decision_time",validate="one_to_one").merge(volatility(),on="decision_time",validate="one_to_one")
 f["signal_valid"]=f.spot_valid&f.perpetual_valid&f.vol_valid&np.isfinite(f[["spot_return","perpetual_return"]]).all(axis=1)&f.spot_return.ne(0)&f.perpetual_return.ne(0);return f.sort_values("decision_time").reset_index(drop=True)
def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 vol=pd.Series(True,index=f.index) if control=="no_joint_expansion" else f.bvol_body.gt(0)&f.dvol_body.gt(0)
 tail=f.spot_return.ne(0)
 if control!="no_spot_tail":tail&=f.prior_abs_spot_q60.notna()&f.spot_return.abs().ge(f.prior_abs_spot_q60)
 agree=f.perpetual_return.ne(0)
 if control!="no_direction_agreement":agree&=np.sign(f.spot_return).eq(np.sign(f.perpetual_return))
 ratio=f.perpetual_return.abs()/f.spot_return.abs();partial=pd.Series(True,index=f.index) if control=="no_partial_transmission" else ratio.le(.50)
 active=f.signal_valid&vol&tail&agree&partial;on=active&~active.shift(1,fill_value=False)&f.signal_valid.shift(1,fill_value=False)&f.decision_time.diff().eq(pd.Timedelta(hours=1));return on,ratio
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 on,ratio=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[on]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(s,e) in SPLITS.items() if entry>=s and exit_<=e),None)
  if split is None:continue
  side=int(np.sign(f.at[i,"spot_return"]));side=-side if control=="direction_flip" else side;next_allowed=exit_;rows.append({"candidate":"SLVCR-6","control":control,"split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":side,"spot_return":float(f.at[i,"spot_return"]),"perpetual_return":float(f.at[i,"perpetual_return"]),"perpetual_to_spot_ratio":float(ratio.at[i]),"prior_abs_spot_q60":float(f.at[i,"prior_abs_spot_q60"]),"bvol_body":float(f.at[i,"bvol_body"]),"dvol_body":float(f.at[i,"dvol_body"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def materialize(f:pd.DataFrame)->dict:
 SOURCE_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(f,SPOT);core={"protocol_version":"slvcr_6_spot_source_v1","query":QUERY,"table":"bars_binance_spot","symbol":"BTCUSDT","interval":"1m","window":[START.isoformat(),END.isoformat()],"outcomes_opened":False,"candidate_incidence_opened":False,"output":{"path":str(SPOT),"sha256":sha(SPOT),"rows":len(f),"valid_rows":int(f.spot_valid.sum())}};r={**core,"manifest_hash":chash(core)};SOURCE_MANIFEST.write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n");return r
def run(env_file:str=ENV_FILE)->dict:
 s=query_spot(env_file);sm=materialize(s);f=features(s);primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,x in support.items():checks[f"{n}_minimum_events"]=x["events"]>=MINIMUM_EVENTS[n];checks[f"{n}_side_balance"]=x["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=x["max_month_share"]<=.45
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());vm=base.SOURCE_DIR/"manifest.json";pm=intrahour.PRICE_DIR/"manifest.json";passed=all(checks.values());core={"protocol_version":"slvcr_6_source_support_v1","policy_id":"SLVCR-6","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":sha(prereg.DEFAULT_OUTPUT),"manifest_hash":reg["manifest_hash"]},"source_manifests":{"spot":{"path":str(SOURCE_MANIFEST),"sha256":sha(SOURCE_MANIFEST),"manifest_hash":sm["manifest_hash"]},"volatility":{"path":str(vm),"sha256":sha(vm)},"perpetual_completed_hour":{"path":str(pm),"sha256":sha(pm)}},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":p=argparse.ArgumentParser();p.add_argument("--env-file",default=ENV_FILE);a=p.parse_args();r=run(a.env_file);print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
