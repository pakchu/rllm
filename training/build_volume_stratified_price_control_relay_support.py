"""Build source-only VSPCR-8 clocks."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_volume_stratified_price_control_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="e52a9aa8cc33aecb42c6e8496d95b0bd7d62e527fcbb0e38aaee02f036769a17";START=pd.Timestamp("2023-04-01T00:00Z");END=pd.Timestamp("2026-08-01T00:00Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("unweighted_6h_return","high_volume_without_disagreement","single_dominant_volume_bar","temporal_half_partition","one_decision_stale_strata","direction_flip")
QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS bar_time,(array_agg(open ORDER BY ts))[1] AS bar_open,(array_agg(close ORDER BY ts DESC))[1] AS bar_close,sum(quote_asset_volume) AS quote_volume,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND quote_asset_volume>=0 AND high>=greatest(open,close) AND low<=least(open,close) AND high>=low) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY bar_time ORDER BY bar_time"""
SOURCE_DIR=Path("data/volume_stratified_price_control_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"four_hour_strata.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/volume_stratified_price_control_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/volume_stratified_price_control_relay_controls_2023_2026");RESULT=Path("results/volume_stratified_price_control_relay_support_2026-08-09.json")
FEATURE_COLUMNS=("decision_time","feature_available_time","source_valid","high_volume_return","low_volume_return","normalizer","stratified_disagreement","high_volume_dominance","unweighted_return","dominant_bar_return","final_half_return","early_half_return","temporal_disagreement","temporal_dominance","stratified_rank","unweighted_rank","dominant_bar_rank","temporal_rank")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","high_volume_return","low_volume_return","normalizer","stratified_disagreement","high_volume_dominance","unweighted_return","dominant_bar_return","final_half_return","early_half_return","temporal_disagreement","temporal_dominance","stratified_rank","unweighted_rank","dominant_bar_rank","temporal_rank")
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
 req=["bar_time","bar_open","bar_close","quote_volume","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if not set(req).issubset(raw.columns):raise ValueError("VSPCR schema drift")
 f=raw[req].copy()
 for c in ("bar_time","first_ts","last_ts"):f[c]=pd.to_datetime(f[c],utc=True,errors="coerce")
 for c in ("bar_open","bar_close","quote_volume","source_rows","distinct_rows"):f[c]=pd.to_numeric(f[c],errors="coerce")
 f=f.sort_values("bar_time",kind="mergesort").set_index("bar_time");rows=[]
 for d in pd.date_range(START+pd.Timedelta(hours=8),END,freq="4h",inclusive="left"):
  expected=pd.date_range(d-pd.Timedelta(hours=6),d,freq="5min",inclusive="left");w=f.reindex(expected);p=w[["bar_open","bar_close"]]
  ok=bool(np.isfinite(w[["bar_open","bar_close","quote_volume","source_rows","distinct_rows"]]).all(axis=1).all() and p.gt(0).all(axis=1).all() and w.quote_volume.ge(0).all() and w.source_rows.eq(5).all() and w.distinct_rows.eq(5).all() and w.coherent.fillna(False).astype(bool).all() and w.first_ts.equals(pd.Series(expected,index=expected)) and w.last_ts.equals(pd.Series(expected+pd.Timedelta(minutes=4),index=expected)))
  if ok:
   r=np.log(w.bar_close.to_numpy(float)/w.bar_open.to_numpy(float));q=w.quote_volume.to_numpy(float);order=np.lexsort((expected.asi8,q));low=order[:18];high=order[-18:]
   rh=float(r[high].sum());rl=float(r[low].sum());v=float(np.sqrt(np.square(r).sum()));den=abs(rh)+abs(rl);s=abs(rh-rl)/v if v>0 else np.nan;dh=abs(rh)/den if den>0 else np.nan;u=float(r.sum());m=float(r[order[-1]]);re=float(r[:36].sum());rf=float(r[36:].sum());tden=abs(rf)+abs(re);st=abs(rf-re)/v if v>0 else np.nan;dt=abs(rf)/tden if tden>0 else np.nan;ok=bool(np.isfinite([rh,rl,v,s,dh,u,m,re,rf,st,dt]).all() and v>0)
  else:rh=rl=v=s=dh=u=m=re=rf=st=dt=np.nan
  rows.append({"decision_time":d,"feature_available_time":d,"source_valid":ok,"high_volume_return":rh,"low_volume_return":rl,"normalizer":v,"stratified_disagreement":s,"high_volume_dominance":dh,"unweighted_return":u,"dominant_bar_return":m,"final_half_return":rf,"early_half_return":re,"temporal_disagreement":st,"temporal_dominance":dt})
 out=pd.DataFrame(rows);out["stratified_rank"]=strict_prior_midrank(out.stratified_disagreement.where(out.source_valid));out["unweighted_rank"]=strict_prior_midrank(out.unweighted_return.abs().where(out.source_valid));out["dominant_bar_rank"]=strict_prior_midrank(out.dominant_bar_return.abs().where(out.source_valid));out["temporal_rank"]=strict_prior_midrank(out.temporal_disagreement.where(out.source_valid));return out[list(FEATURE_COLUMNS)]
def onset(active:pd.Series)->pd.Series:return active.fillna(False)&~active.shift(1,fill_value=False)
def active_and_side(f:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 primary=f.source_valid&f.high_volume_return.mul(f.low_volume_return).lt(0)&f.high_volume_dominance.ge(2/3)&f.stratified_rank.ge(.80);imp=f.high_volume_return
 if control=="unweighted_6h_return":state=f.source_valid&f.unweighted_rank.ge(.80);imp=f.unweighted_return
 elif control=="high_volume_without_disagreement":state=f.source_valid&f.high_volume_return.ne(0)&f.high_volume_dominance.ge(2/3)&f.stratified_rank.ge(.80)
 elif control=="single_dominant_volume_bar":state=f.source_valid&f.dominant_bar_return.ne(0)&f.dominant_bar_rank.ge(.80);imp=f.dominant_bar_return
 elif control=="temporal_half_partition":state=f.source_valid&f.final_half_return.mul(f.early_half_return).lt(0)&f.temporal_dominance.ge(2/3)&f.temporal_rank.ge(.80);imp=f.final_half_return
 elif control=="one_decision_stale_strata":state=primary.shift(1,fill_value=False);imp=f.high_volume_return.shift(1)
 else:state=primary
 active=onset(state);side=np.sign(imp)
 if control=="direction_flip":side=-side
 return active&pd.Series(imp,index=f.index).ne(0),pd.Series(side,index=f.index).astype("Int64").fillna(0).astype(int)
def build_clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,sides=active_and_side(f,control);rows=[];reserved=None
 for i in f.index[active&sides.ne(0)]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8)
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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("VSPCR prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);raw=load_source();features=build_features(raw);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"vspcr_8_source_v1","query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"physical_rows":int(raw.source_rows.sum()),"aggregate_rows":len(raw),"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features)},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False};sm={**sc,"manifest_hash":canonical_hash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"vspcr_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};out={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");return out
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
