"""Build outcome-blind source support for frozen HVEMFX-12."""
from __future__ import annotations
import gzip, hashlib, io, json, math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_em_fx_dollar_stress_breadth_relay as prereg

ENV_FILE="/home/pakchu/rllm/.env";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");PREREG_SHA="09be9ec220761d2a0956e81d48ec5740808ce3895d34ea13077d02e7e2c60cc8";REG=prereg.build();P=REG["policy"];SPLITS={k:tuple(map(pd.Timestamp,v)) for k,v in REG["stages"].items()};GATES=REG["source_support_gates"];SYMBOLS=tuple(REG["source_plan"]["fx"]["symbols"]);CONTROLS=tuple(REG["diagnostic_controls"]["names"])
FX_QUERY="""SELECT date_trunc('day',ts) AS source_day,(array_agg(open ORDER BY ts))[1] AS session_open,max(high) AS session_high,min(low) AS session_low,(array_agg(close ORDER BY ts DESC))[1] AS session_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_polygon WHERE symbol=:symbol AND interval='1m' AND ts>=:start AND ts<:end AND extract(isodow from ts) BETWEEN 1 AND 5 AND ts::time<TIME '22:30' GROUP BY 1 ORDER BY 1"""
BTC_QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,(array_agg(close ORDER BY ts DESC))[1] AS close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""
ROOT=Path("data/high_volatility_em_fx_dollar_stress_breadth_relay_sources_2023_2026");PANEL=ROOT/"scheduled_em_fx_states.csv.gz";MANIFEST=ROOT/"manifest.json";CLOCK=Path("data/high_volatility_em_fx_dollar_stress_breadth_relay_clocks_2023_2026.csv.gz");SPLIT_DIR=Path("data/high_volatility_em_fx_dollar_stress_breadth_relay_split_clocks_2023_2026");CONTROL_DIR=Path("data/high_volatility_em_fx_dollar_stress_breadth_relay_controls_2023_2026");RESULT=Path("results/high_volatility_em_fx_dollar_stress_breadth_relay_support_2026-08-13.json");BUILDER=Path(__file__).relative_to(Path.cwd())
CLOCK_COLS=("candidate","control","split","source_day","decision_time","feature_available_time","entry_time","exit_time","side","common_dollar_direction","agreeing_pairs","median_absolute_pair_z","btc_realized_variation","variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def causal_z(v:pd.Series,lookback:int,minimum:int)->pd.Series:
 a=pd.to_numeric(v,errors="coerce").to_numpy(float);out=np.full(len(a),np.nan);history=[]
 for i,current in enumerate(a):
  prior=np.asarray(history[-lookback:],float)
  if math.isfinite(current) and len(prior)>=minimum:
   std=float(np.std(prior,ddof=1));out[i]=(current-float(np.mean(prior)))/std if std>0 else np.nan
  if math.isfinite(current):history.append(current)
 return pd.Series(out,index=v.index)
def midrank(v:pd.Series,lookback:int,minimum:int)->pd.Series:
 a=pd.to_numeric(v,errors="coerce").to_numpy(float);out=np.full(len(a),np.nan);history=[]
 for i,current in enumerate(a):
  prior=np.asarray(history[-lookback:],float)
  if math.isfinite(current) and len(prior)>=minimum:out[i]=(np.sum(prior<current)+.5*np.sum(prior==current))/len(prior)
  if math.isfinite(current):history.append(current)
 return pd.Series(out,index=v.index)
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_sources():
 from sqlalchemy import text
 db=postgres_engine();fx={}
 try:
  with db.connect() as c:
   for symbol in SYMBOLS:fx[symbol]=pd.read_sql_query(text(FX_QUERY),c,params={"symbol":symbol,"start":START.to_pydatetime(),"end":END.to_pydatetime()})
   btc=pd.read_sql_query(text(BTC_QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:db.dispose()
 return fx,btc
def prepare_fx(raw:pd.DataFrame,symbol:str)->pd.DataFrame:
 expected=["source_day","session_open","session_high","session_low","session_close","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if raw.columns.tolist()!=expected:raise RuntimeError(f"HVEMFX schema drift: {symbol}")
 x=raw.copy();x.source_day=pd.to_datetime(x.source_day,utc=True);x.first_ts=pd.to_datetime(x.first_ts,utc=True);x.last_ts=pd.to_datetime(x.last_ts,utc=True)
 for c in ("session_open","session_high","session_low","session_close","source_rows","distinct_rows"):x[c]=pd.to_numeric(x[c],errors="coerce")
 z=x[["session_open","session_high","session_low","session_close"]];valid=x.distinct_rows.ge(240)&x.first_ts.le(x.source_day+pd.Timedelta("5h"))&x.last_ts.ge(x.source_day+pd.Timedelta("9h30m"))&x.coherent.eq(True)&np.isfinite(z).all(axis=1)&z.gt(0).all(axis=1)&x.session_high.ge(z[["session_open","session_close"]].max(axis=1))&x.session_low.le(z[["session_open","session_close"]].min(axis=1));ret=np.log(x.session_close/x.session_open).where(valid);return pd.DataFrame({"source_day":x.source_day,f"{symbol}_valid":valid,f"{symbol}_return":ret,f"{symbol}_z":causal_z(ret,P["fx_prior_sessions"],P["fx_prior_min_sessions"])})
def prepare_btc(raw:pd.DataFrame)->pd.DataFrame:
 x=raw.copy();x.date=pd.to_datetime(x.date,utc=True);x.first_ts=pd.to_datetime(x.first_ts,utc=True);x.last_ts=pd.to_datetime(x.last_ts,utc=True)
 for c in ("open","high","low","close","source_rows","distinct_rows"):x[c]=pd.to_numeric(x[c],errors="coerce")
 z=x[["open","high","low","close"]];x["valid"]=x.source_rows.eq(5)&x.distinct_rows.eq(5)&x.first_ts.eq(x.date)&x.last_ts.eq(x.date+pd.Timedelta("4m"))&x.coherent.eq(True)&np.isfinite(z).all(axis=1)&z.gt(0).all(axis=1);x["return"]=np.log(x.close/x.open);grid=pd.date_range(START,END,freq="5min",inclusive="left");x=x.set_index("date").reindex(grid);x["variation"]=np.sqrt(x["return"].pow(2).where(x.valid.eq(True)).rolling(288,min_periods=288).sum());x["window_valid"]=x.valid.eq(True).rolling(288,min_periods=288).sum().eq(288);return x
def build_panel(fx:dict[str,pd.DataFrame],btc_raw:pd.DataFrame)->pd.DataFrame:
 parts=[prepare_fx(fx[s],s) for s in SYMBOLS];f=parts[0]
 for piece in parts[1:]:f=f.merge(piece,on="source_day",how="outer",validate="one_to_one")
 f=f.sort_values("source_day").reset_index(drop=True);f["decision_time"]=f.source_day+pd.Timedelta("22h30m");btc=prepare_btc(btc_raw);ends=f.decision_time-pd.Timedelta("5m");f["btc_realized_variation"]=[float(btc.at[t,"variation"]) if t in btc.index and math.isfinite(btc.at[t,"variation"]) else np.nan for t in ends];f["btc_valid"]=[bool(btc.at[t,"window_valid"]) if t in btc.index else False for t in ends];valid_cols=[f"{s}_valid" for s in SYMBOLS];zcols=[f"{s}_z" for s in SYMBOLS];f["source_valid"]=f[valid_cols].fillna(False).all(axis=1)&f.btc_valid&np.isfinite(f[zcols]).all(axis=1)&np.isfinite(f.btc_realized_variation);z=f[zcols].astype(float);positive=z.gt(0).sum(axis=1);negative=z.lt(0).sum(axis=1);weighted=np.sign(z.sum(axis=1));f["common_dollar_direction"]=np.where(positive>negative,1,np.where(negative>positive,-1,weighted)).astype(int);f["agreeing_pairs"]=np.maximum(positive,negative);f["median_absolute_pair_z"]=z.abs().median(axis=1);f["variation_rank"]=midrank(f.btc_realized_variation.where(f.source_valid),P["realized_variation_prior_sessions"],P["realized_variation_min_sessions"]);return f
def conditions(f:pd.DataFrame,control:str):
 direction=f.common_dollar_direction;breadth=f.agreeing_pairs;shock=f.median_absolute_pair_z
 if control=="one_session_stale_stress":direction=direction.shift(1);breadth=breadth.shift(1);shock=shock.shift(1)
 minimum=2 if control=="two_pair_breadth" else P["minimum_agreeing_pairs"];shock_gate=pd.Series(True,index=f.index) if control=="no_em_fx_shock_gate" else shock.ge(P["median_absolute_pair_z_min"]);vol=pd.Series(True,index=f.index) if control=="no_volatility_gate" else f.variation_rank.ge(P["realized_variation_rank_min"]);active=f.source_valid&direction.abs().eq(1)&breadth.ge(minimum)&shock_gate&vol;side=-direction
 if control=="direction_flip":side=-side
 if control=="forced_long":side=pd.Series(1,index=f.index)
 return active,side
def clock(f:pd.DataFrame,control="primary"):
 active,side=conditions(f,control);rows=[];reserved=None
 for i in f.index[active]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta("5m");exit_=entry+pd.Timedelta("12h")
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split:reserved=exit_;rows.append({"candidate":"HVEMFX-12","control":control,"split":split,"source_day":f.at[i,"source_day"],"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),**{c:float(f.at[i,c]) for c in CLOCK_COLS[9:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLS)
def stats(c,n):
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":s,"minority_side_share":min(l,s)/len(x),"max_month_share":int(m.max())/len(x)}
def gz(x):
 b=io.BytesIO();raw=x.to_csv(index=False,float_format="%.12g",lineterminator="\n").encode()
 with gzip.GzipFile(fileobj=b,mode="wb",compresslevel=6,mtime=0) as f:f.write(raw)
 return b.getvalue()
def immutable(p,b):p.parent.mkdir(parents=True,exist_ok=True);(p.exists() and p.read_bytes()!=b) and (_ for _ in ()).throw(RuntimeError(f"refusing overwrite {p}"));p.write_bytes(b)
def jb(x):return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=str)+"\n").encode()
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVEMFX prereg drift")
 fx,btc=load_sources();panel=build_panel(fx,btc);primary=clock(panel);controls={n:clock(panel,n) for n in CONTROLS};immutable(PANEL,gz(panel));immutable(CLOCK,gz(primary))
 for n,x in controls.items():immutable(CONTROL_DIR/f"{n}.csv.gz",gz(x))
 for n in SPLITS:immutable(SPLIT_DIR/f"{n}.csv.gz",gz(primary[primary.split.eq(n)]))
 source_core={"protocol_version":"hvemfx_12_source_v1","fx_query":FX_QUERY,"btc_query":BTC_QUERY,"query_sha256":chash({"fx":FX_QUERY,"btc":BTC_QUERY}),"window":[START.isoformat(),END.isoformat()],"builder":{"path":str(BUILDER),"sha256":sha(BUILDER)},"panel":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(panel),"valid_rows":int(panel.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};manifest={**source_core,"manifest_hash":chash(source_core)};immutable(MANIFEST,jb(manifest));support={n:stats(primary,n) for n in SPLITS};checks={k:o for n,x in support.items() for k,o in ((f"{n}_minimum_events",x["events"]>=GATES["minimum_events"][n]),(f"{n}_side_balance",x["minority_side_share"]>=GATES["minority_side_share_min"]),(f"{n}_month_concentration",x["max_month_share"]<=GATES["max_month_share"]))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvemfx_12_source_support_v1","policy_id":"HVEMFX-12","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":manifest["manifest_hash"]},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"split_artifacts":{n:{"path":str(SPLIT_DIR/f"{n}.csv.gz"),"sha256":sha(SPLIT_DIR/f"{n}.csv.gz"),"rows":int(primary.split.eq(n).sum())} for n in SPLITS},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":chash(core)};immutable(RESULT,jb(result));return result
if __name__=="__main__":print(json.dumps({"passed":run()["support_passed"],"result":str(RESULT)}))
