"""Build source-only CATPJI-6 clocks."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_cross_asset_traded_price_jensen_isolation as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="7ee34adfca61a01a46151e5932ea65643a8b4c1336bdc73104e7bd044833f440";START=pd.Timestamp("2023-01-01T00:00Z");END=pd.Timestamp("2026-08-01T00:00Z");SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("btc_jensen_only","cross_asset_isolation_only","close_to_hour_vwap","one_hour_stale_jensen","direction_flip")
QUERY="""WITH five AS (SELECT symbol,date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS bar_time,(array_agg(open ORDER BY ts))[1] AS bar_open,(array_agg(close ORDER BY ts DESC))[1] AS bar_close,sum(volume) AS base_volume,sum(quote_asset_volume) AS quote_volume,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND volume>=0 AND quote_asset_volume>=0 AND high>=greatest(open,close) AND low<=least(open,close) AND high>=low) AS coherent FROM bars_binance WHERE symbol=ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end GROUP BY symbol,bar_time), hourly AS (SELECT symbol,date_bin('1 hour',bar_time,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS hour_start,(array_agg(bar_open ORDER BY bar_time))[1] AS hour_open,(array_agg(bar_close ORDER BY bar_time DESC))[1] AS hour_close,sum(base_volume) AS total_base,sum(quote_volume) AS total_quote,sum(base_volume*ln(quote_volume/base_volume)) AS weighted_log_price,sum(source_rows) AS physical_rows,count(*) AS five_rows,min(bar_time) AS first_bar,max(bar_time) AS last_bar,bool_and(source_rows=5 AND distinct_rows=5 AND first_ts=bar_time AND last_ts=bar_time+INTERVAL '4 minutes' AND coherent AND base_volume>0 AND quote_volume>0) AS coherent FROM five GROUP BY symbol,hour_start) SELECT *,ln(total_quote/total_base)-weighted_log_price/total_base AS jensen_gap,ln(hour_close/(total_quote/total_base)) AS close_vwap_displacement FROM hourly ORDER BY hour_start,symbol"""
SOURCE_DIR=Path("data/cross_asset_traded_price_jensen_isolation_sources_2023_2026");FEATURES=SOURCE_DIR/"hourly_jensen.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/cross_asset_traded_price_jensen_isolation_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/cross_asset_traded_price_jensen_isolation_controls_2023_2026");RESULT=Path("results/cross_asset_traded_price_jensen_isolation_support_2026-08-09.json")
FEATURE_COLUMNS=("decision_time","feature_available_time","source_valid","btc_return","btc_jensen_gap","btc_jensen_rank","alt_median_jensen","isolation","isolation_rank","close_vwap_displacement","close_vwap_abs_rank")
CLOCK_COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side",*FEATURE_COLUMNS[3:])
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def prior_midrank(v:pd.Series,lookback:int=2160,minimum:int=1440)->pd.Series:
 out=pd.Series(np.nan,index=v.index,dtype=float);h=[]
 for i,x in pd.to_numeric(v,errors="coerce").items():
  p=h[-lookback:]
  if math.isfinite(x) and len(p)>=minimum:
   a=np.asarray(p);out.at[i]=(np.sum(a<x)+.5*np.sum(a==x))/len(a)
  if math.isfinite(x):h.append(float(x))
 return out
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_source()->pd.DataFrame:
 from sqlalchemy import text
 db=engine()
 try:
  with db.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"symbols":list(SYMBOLS),"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:db.dispose()
def build_features(raw:pd.DataFrame)->pd.DataFrame:
 req={"symbol","hour_start","hour_open","hour_close","total_base","total_quote","physical_rows","five_rows","first_bar","last_bar","coherent","jensen_gap","close_vwap_displacement"}
 if not req.issubset(raw):raise ValueError("CATPJI schema drift")
 f=raw.copy()
 for c in ("hour_start","first_bar","last_bar"):f[c]=pd.to_datetime(f[c],utc=True,errors="coerce")
 for c in ("hour_open","hour_close","total_base","total_quote","physical_rows","five_rows","jensen_gap","close_vwap_displacement"):f[c]=pd.to_numeric(f[c],errors="coerce")
 grid=pd.date_range(START,END,freq="1h",inclusive="left");gaps={};returns={};valids={};displacements={}
 for symbol in SYMBOLS:
  x=f[f.symbol.eq(symbol)].sort_values("hour_start",kind="mergesort").drop_duplicates("hour_start",keep=False).set_index("hour_start").reindex(grid);valid=np.isfinite(x[["hour_open","hour_close","total_base","total_quote","physical_rows","five_rows","jensen_gap","close_vwap_displacement"]]).all(axis=1)&x.hour_open.gt(0)&x.hour_close.gt(0)&x.total_base.gt(0)&x.total_quote.gt(0)&x.physical_rows.eq(60)&x.five_rows.eq(12)&x.coherent.fillna(False).astype(bool)&x.jensen_gap.ge(-1e-15)&x.first_bar.eq(pd.Series(grid,index=grid))&x.last_bar.eq(pd.Series(grid+pd.Timedelta(minutes=55),index=grid));gaps[symbol]=x.jensen_gap.clip(lower=0).where(valid);returns[symbol]=np.log(x.hour_close/x.hour_open).where(valid);valids[symbol]=valid;displacements[symbol]=x.close_vwap_displacement.where(valid)
 panel=pd.DataFrame(gaps,index=grid);source_valid=panel.notna().all(axis=1)&pd.DataFrame(valids,index=grid).all(axis=1);btc=panel.BTCUSDT.where(source_valid);alt=panel[list(SYMBOLS[1:])].median(axis=1).where(source_valid);iso=(btc-alt).where(source_valid);ret=returns["BTCUSDT"].where(source_valid);disp=displacements["BTCUSDT"].where(source_valid)
 out=pd.DataFrame({"decision_time":grid+pd.Timedelta(hours=1),"feature_available_time":grid+pd.Timedelta(hours=1),"source_valid":source_valid.to_numpy(),"btc_return":ret.to_numpy(),"btc_jensen_gap":btc.to_numpy(),"alt_median_jensen":alt.to_numpy(),"isolation":iso.to_numpy(),"close_vwap_displacement":disp.to_numpy()});out["btc_jensen_rank"]=prior_midrank(out.btc_jensen_gap.where(out.source_valid));out["isolation_rank"]=prior_midrank(out.isolation.where(out.source_valid));out["close_vwap_abs_rank"]=prior_midrank(out.close_vwap_displacement.abs().where(out.source_valid));return out[list(FEATURE_COLUMNS)]
def onset(x:pd.Series)->pd.Series:return x.fillna(False)&~x.shift(1,fill_value=False)
def active_and_side(f:pd.DataFrame,control:str="primary"):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 primary=f.source_valid&f.btc_return.ne(0)&f.btc_jensen_rank.ge(.9)&f.isolation_rank.ge(.9);sidev=-f.btc_return
 if control=="btc_jensen_only":state=f.source_valid&f.btc_return.ne(0)&f.btc_jensen_rank.ge(.9);active=state&f.btc_jensen_rank.shift(1).lt(.9)
 elif control=="cross_asset_isolation_only":state=f.source_valid&f.btc_return.ne(0)&f.isolation_rank.ge(.9);active=state&f.isolation_rank.shift(1).lt(.9)
 elif control=="close_to_hour_vwap":state=f.source_valid&f.close_vwap_displacement.ne(0)&f.close_vwap_abs_rank.ge(.9);active=state&f.close_vwap_abs_rank.shift(1).lt(.9);sidev=-f.close_vwap_displacement
 elif control=="one_hour_stale_jensen":state=primary.shift(1,fill_value=False);sidev=(-f.btc_return).shift(1);active=onset(state)
 else:active=primary&f.btc_jensen_rank.shift(1).lt(.9)&f.isolation_rank.shift(1).lt(.9)
 side=np.sign(sidev)
 if control=="direction_flip":side=-side
 return active&pd.Series(sidev,index=f.index).ne(0),pd.Series(side,index=f.index).astype("Int64").fillna(0).astype(int)
def build_clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,sides=active_and_side(f,control);rows=[];reserved=None
 for i in f.index[active&sides.ne(0)]:
  d=pd.Timestamp(f.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=6)
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
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("CATPJI prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);raw=load_source();features=build_features(raw);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"catpji_6_source_v1","query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"symbols":list(SYMBOLS),"physical_rows":int(raw.physical_rows.sum()),"aggregate_rows":len(raw),"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features)},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False};sm={**sc,"manifest_hash":chash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"catpji_6_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};out={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");return out
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
