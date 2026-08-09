"""Build source-only TECC-12 clocks."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_ticket_expansion_concordance_continuation as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="35b3912969893bf77886ffc687961a242c4234901232c25c991366a3df19d526";START=pd.Timestamp("2023-04-01T00:00Z");END=pd.Timestamp("2026-08-01T00:00Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_ticket_tail","no_concordance","early_ticket_dominance","direction_flip")
QUERY="""SELECT date_bin('2 hours',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS block_start,(array_agg(open ORDER BY ts))[1] AS block_open,max(high) AS block_high,min(low) AS block_low,(array_agg(close ORDER BY ts DESC))[1] AS block_close,sum(quote_asset_volume) AS quote_volume,sum(number_of_trades) AS trade_count,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close) AND low<=least(open,close) AND high>=low AND quote_asset_volume>=0 AND number_of_trades>=0 AND number_of_trades=floor(number_of_trades)) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY block_start ORDER BY block_start"""
SOURCE_DIR=Path("data/ticket_expansion_concordance_continuation_sources_2023_2026");FEATURES=SOURCE_DIR/"four_hour_ticket_expansion.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/ticket_expansion_concordance_continuation_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/ticket_expansion_concordance_continuation_controls_2023_2026");RESULT=Path("results/ticket_expansion_concordance_continuation_support_2026-08-09.json")
FEATURE_COLUMNS=("decision_time","feature_available_time","source_valid","early_return","late_return","early_quote_volume","late_quote_volume","early_trade_count","late_trade_count","early_ticket","late_ticket","ticket_expansion","ticket_expansion_rank","early_dominance_rank","concordant","primary_state")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","early_return","late_return","early_ticket","late_ticket","ticket_expansion","ticket_expansion_rank","early_dominance_rank","concordant")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def strict_prior_midrank(values:pd.Series,lookback:int=540,minimum:int=360)->pd.Series:
 out=pd.Series(np.nan,index=values.index,dtype=float);history=[]
 for i,current in pd.to_numeric(values,errors="coerce").items():
  prior=history[-lookback:]
  if math.isfinite(current) and len(prior)>=minimum:
   a=np.asarray(prior);out.at[i]=(np.sum(a<current)+.5*np.sum(a==current))/len(a)
  if math.isfinite(current):history.append(float(current))
 return out
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_source()->pd.DataFrame:
 from sqlalchemy import text
 db=engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:db.dispose()
def build_features(raw:pd.DataFrame)->pd.DataFrame:
 req=["block_start","block_open","block_high","block_low","block_close","quote_volume","trade_count","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if not set(req).issubset(raw.columns):raise ValueError("TECC schema drift")
 f=raw[req].copy();f.block_start=pd.to_datetime(f.block_start,utc=True,errors="coerce");f.first_ts=pd.to_datetime(f.first_ts,utc=True,errors="coerce");f.last_ts=pd.to_datetime(f.last_ts,utc=True,errors="coerce")
 for c in ("block_open","block_high","block_low","block_close","quote_volume","trade_count","source_rows","distinct_rows"):f[c]=pd.to_numeric(f[c],errors="coerce")
 f=f.sort_values("block_start",kind="mergesort").set_index("block_start");rows=[]
 for d in pd.date_range(START+pd.Timedelta(hours=8),END,freq="4h",inclusive="left"):
  expected=pd.date_range(d-pd.Timedelta(hours=6),d,freq="2h",inclusive="left");w=f.reindex(expected);p=w[["block_open","block_high","block_low","block_close"]]
  ok=bool(len(w)==3 and np.isfinite(w[["block_open","block_high","block_low","block_close","quote_volume","trade_count","source_rows","distinct_rows"]]).all(axis=1).all() and p.gt(0).all(axis=1).all() and w.source_rows.eq(120).all() and w.distinct_rows.eq(120).all() and w.coherent.fillna(False).astype(bool).all() and w.first_ts.equals(pd.Series(expected,index=expected)) and w.last_ts.equals(pd.Series(expected+pd.Timedelta(minutes=119),index=expected)))
  early=w.iloc[:2];late=w.iloc[2:];eq=float(early.quote_volume.sum()) if ok else np.nan;lq=float(late.quote_volume.sum()) if ok else np.nan;en=float(early.trade_count.sum()) if ok else np.nan;ln=float(late.trade_count.sum()) if ok else np.nan;ok=bool(ok and eq>0 and lq>0 and en>0 and ln>0)
  er=float(np.log(early.block_close.iloc[-1]/early.block_open.iloc[0])) if ok else np.nan;lr=float(np.log(late.block_close.iloc[-1]/late.block_open.iloc[0])) if ok else np.nan;et=eq/en if ok else np.nan;lt=lq/ln if ok else np.nan;exp=float(np.log(lt/et)) if ok else np.nan;con=bool(ok and er!=0 and lr!=0 and np.sign(er)==np.sign(lr))
  rows.append({"decision_time":d,"feature_available_time":d,"source_valid":ok,"early_return":er,"late_return":lr,"early_quote_volume":eq,"late_quote_volume":lq,"early_trade_count":en,"late_trade_count":ln,"early_ticket":et,"late_ticket":lt,"ticket_expansion":exp,"concordant":con})
 out=pd.DataFrame(rows);out["ticket_expansion_rank"]=strict_prior_midrank(out.ticket_expansion.where(out.source_valid));out["early_dominance_rank"]=strict_prior_midrank((-out.ticket_expansion).where(out.source_valid));out["primary_state"]=out.source_valid&out.concordant&out.ticket_expansion.gt(0)&out.ticket_expansion_rank.ge(.8);return out[list(FEATURE_COLUMNS)]
def states_and_side(f:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 if control=="no_ticket_tail":state=f.source_valid&f.concordant&f.ticket_expansion.gt(0);side=np.sign(f.late_return)
 elif control=="no_concordance":state=f.source_valid&f.ticket_expansion.gt(0)&f.ticket_expansion_rank.ge(.8)&f.late_return.ne(0);side=np.sign(f.late_return)
 elif control=="early_ticket_dominance":state=f.source_valid&f.concordant&f.ticket_expansion.lt(0)&f.early_dominance_rank.ge(.8);side=np.sign(f.late_return)
 else:state=f.primary_state;side=np.sign(f.late_return)*(-1 if control=="direction_flip" else 1)
 onset=state&~state.shift(1,fill_value=False);return onset,pd.Series(side,index=f.index).astype("Int64").fillna(0).astype(int)
def build_clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,sides=states_and_side(f,control);rows=[];reserved=None
 for i in f.index[active&sides.ne(0)]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":d,"feature_available_time":f.at[i,"feature_available_time"],"entry_time":entry,"exit_time":exit_,"side":int(sides.at[i]),**{c:f.at[i,c] for c in CLOCK_COLUMNS[8:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(c:pd.DataFrame,s:str)->dict[str,float|int]:
 x=c[c.split.eq(s)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());q=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":l,"shorts":q,"minority_side_share":min(l,q)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("TECC prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);raw=load_source();features=build_features(raw);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"tecc_12_source_v1","query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":int(raw.source_rows.sum()),"aggregate_rows":len(raw),"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features)},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False};sm={**sc,"manifest_hash":canonical_hash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"tecc_12_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};out={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");return out
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
