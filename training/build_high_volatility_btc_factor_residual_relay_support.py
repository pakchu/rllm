"""Build source-only support for frozen HVBFRR-12."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_high_volatility_btc_factor_residual_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="de97c861daa3ea78241e6f996345b18f5fc8fb7869acfaa7e2550df02c254683";SYMBOLS=("BTCUSDT",*prereg.ALTS);START=pd.Timestamp("2023-01-01T00:00Z");END=pd.Timestamp("2026-08-01T00:00Z");RAW_START=START-pd.Timedelta(hours=6)
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_residual_tail","no_variation_gate","raw_btc_return_side","fixed_beta_one","one_day_stale_fit","direction_flip","same_clock_forced_long")
QUERY="""SELECT symbol,date_trunc('day',ts+INTERVAL '6 hours')+INTERVAL '2 hours' AS decision_time,(array_agg(open ORDER BY ts))[1] AS block_open,(array_agg(close ORDER BY ts DESC))[1] AS block_close,sqrt(sum(power(ln(close/open),2))) AS realized_variation,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM bars_binance WHERE interval='1m' AND symbol=ANY(:symbols) AND ts>=:start AND ts<:end GROUP BY symbol,decision_time ORDER BY decision_time,symbol"""
SOURCE_DIR=Path("data/high_volatility_btc_factor_residual_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"daily_factor_residual.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/high_volatility_btc_factor_residual_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_btc_factor_residual_relay_controls_2023_2026");RESULT=Path("results/high_volatility_btc_factor_residual_relay_support_2026-08-09.json")
FEATURE_COLUMNS=("decision_time","feature_available_time","source_valid","btc_return","alt_factor","btc_variation","fit_alpha","fit_beta","residual","residual_scale","standardized_residual","fixed_beta_one_z","variation_rank")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side",*FEATURE_COLUMNS[3:])
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return prereg.canonical_hash(x)
def prior_midrank(v:pd.Series,lookback:int=270,minimum:int=180)->pd.Series:
 out=pd.Series(np.nan,index=v.index,dtype=float);h=[]
 for i,x in pd.to_numeric(v,errors="coerce").items():
  p=h[-lookback:]
  if math.isfinite(x) and len(p)>=minimum:
   a=np.asarray(p);out.at[i]=(np.sum(a<x)+.5*np.sum(a==x))/len(a)
  if math.isfinite(x):h.append(float(x))
 return out
def causal_residuals(x:pd.Series,y:pd.Series,lookback:int=270,minimum:int=180)->pd.DataFrame:
 out=pd.DataFrame(index=x.index,columns=["fit_alpha","fit_beta","residual","residual_scale","standardized_residual","fixed_beta_one_z"],dtype=float);history=[]
 for i,(xi,yi) in enumerate(zip(pd.to_numeric(x,errors="coerce"),pd.to_numeric(y,errors="coerce"))):
  prior=history[-lookback:]
  if math.isfinite(xi) and math.isfinite(yi) and len(prior)>=minimum:
   a=np.asarray(prior,float);design=np.column_stack([np.ones(len(a)),a[:,0]]);coef=np.linalg.lstsq(design,a[:,1],rcond=None)[0];fitted=design@coef;scale=float(np.std(a[:,1]-fitted,ddof=1));fixed_scale=float(np.std(a[:,1]-a[:,0],ddof=1));res=float(yi-(coef[0]+coef[1]*xi));out.iloc[i]=[coef[0],coef[1],res,scale,res/scale if scale>0 else np.nan,(yi-xi)/fixed_scale if fixed_scale>0 else np.nan]
  if math.isfinite(xi) and math.isfinite(yi):history.append((float(xi),float(yi)))
 return out
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_blocks()->pd.DataFrame:
 from sqlalchemy import text
 db=engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"symbols":list(SYMBOLS),"start":RAW_START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:db.dispose()
def build_features(blocks:pd.DataFrame)->pd.DataFrame:
 req={"symbol","decision_time","block_open","block_close","realized_variation","source_rows","distinct_rows","first_ts","last_ts","coherent"}
 if not req.issubset(blocks):raise ValueError("HVBFRR source schema drift")
 b=blocks.copy();b["decision_time"]=pd.to_datetime(b.decision_time,utc=True);b["first_ts"]=pd.to_datetime(b.first_ts,utc=True);b["last_ts"]=pd.to_datetime(b.last_ts,utc=True)
 for c in ("block_open","block_close","realized_variation","source_rows","distinct_rows"):b[c]=pd.to_numeric(b[c],errors="coerce")
 expected_first=b.decision_time-pd.Timedelta(hours=8);expected_last=b.decision_time-pd.Timedelta(minutes=1);b["valid"]=(b.source_rows.eq(480)&b.distinct_rows.eq(480)&b.first_ts.eq(expected_first)&b.last_ts.eq(expected_last)&b.coherent.eq(True)&np.isfinite(b[["block_open","block_close","realized_variation"]]).all(axis=1)&b.block_open.gt(0)&b.block_close.gt(0)&b.realized_variation.gt(0));b["return"]=np.log(b.block_close/b.block_open).where(b.valid)
 decisions=pd.date_range(pd.Timestamp("2023-01-01T02:00Z"),END,freq="1D",inclusive="left");ret=b.pivot(index="decision_time",columns="symbol",values="return").reindex(decisions);valid=b.pivot(index="decision_time",columns="symbol",values="valid").reindex(decisions).fillna(False).all(axis=1);variation=b[b.symbol.eq("BTCUSDT")].set_index("decision_time").realized_variation.reindex(decisions)
 frame=pd.DataFrame({"decision_time":decisions,"feature_available_time":decisions,"source_valid":valid.to_numpy(bool),"btc_return":ret["BTCUSDT"].to_numpy(float),"alt_factor":ret[list(prereg.ALTS)].mean(axis=1).to_numpy(float),"btc_variation":variation.to_numpy(float)})
 fit=causal_residuals(frame.alt_factor.where(frame.source_valid),frame.btc_return.where(frame.source_valid));frame=pd.concat([frame,fit.reset_index(drop=True)],axis=1);frame["variation_rank"]=prior_midrank(frame.btc_variation.where(frame.source_valid));return frame[list(FEATURE_COLUMNS)]
def conditions(f:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 u=f.shift(1) if control=="one_day_stale_fit" else f;valid=u.source_valid.eq(True)&np.isfinite(u.standardized_residual);rg=pd.Series(True,index=f.index) if control=="no_residual_tail" else u.standardized_residual.abs().ge(1.);vg=pd.Series(True,index=f.index) if control=="no_variation_gate" else u.variation_rank.ge(.65);active=valid&rg&vg;basis=u.btc_return if control=="raw_btc_return_side" else u.fixed_beta_one_z if control=="fixed_beta_one" else u.standardized_residual;side=np.sign(basis).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=f.index)
 return active&side.ne(0),side,u
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,sides,u=conditions(f,control);rows=[]
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12);split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  source=u.loc[i];rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(sides.at[i]),**{c:source[c] for c in FEATURE_COLUMNS[3:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(c:pd.DataFrame,s:str)->dict:
 x=c[c.split.eq(s)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());q=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVBFRR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);blocks=load_blocks();features=build_features(blocks);primary=clock(features);controls={n:clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"hvbfr_12_source_v1","query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"table":"bars_binance","symbols":list(SYMBOLS),"window":[RAW_START.isoformat(),END.isoformat()],"physical_rows_1m":int(pd.to_numeric(blocks.source_rows).sum()),"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features),"valid_rows":int(features.source_valid.sum())},"outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True};sm={**sc,"manifest_hash":chash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"hvbfr_12_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};out={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");return out
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
