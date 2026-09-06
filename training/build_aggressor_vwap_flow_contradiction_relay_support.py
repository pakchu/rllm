"""Build source-only AVFCR-12 clocks from completed BTC one-minute bars."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_aggressor_vwap_flow_contradiction_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="4f9d8035c2582ec444aea52dfe188bae80169155aaa5278cca90c337c1f84b33"
START=pd.Timestamp("2023-06-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00Z"),pd.Timestamp("2024-01-01T00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00Z"),pd.Timestamp("2025-01-01T00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00Z"),pd.Timestamp("2026-01-01T00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00Z"),END)}
MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_flow_contradiction","flow_side","window_return_side","direction_flip")
QUERY="""SELECT date_trunc('day',ts) AS source_day,(array_agg(open ORDER BY ts))[1] AS first_open,(array_agg(close ORDER BY ts DESC))[1] AS last_close,sum(volume) AS total_base,sum(quote_asset_volume) AS total_quote,sum(taker_buy_base) AS buy_base,sum(taker_buy_quote) AS buy_quote,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close) AND low<=least(open,close) AND high>=low AND volume>=0 AND quote_asset_volume>=0 AND taker_buy_base>=0 AND taker_buy_base<=volume AND taker_buy_quote>=0 AND taker_buy_quote<=quote_asset_volume) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end AND EXTRACT(hour FROM ts)>=18 GROUP BY 1 ORDER BY 1"""
SOURCE_DIR=Path("data/aggressor_vwap_flow_contradiction_relay_sources_2023_2026");FEATURES=SOURCE_DIR/"daily_aggressor_vwap.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/aggressor_vwap_flow_contradiction_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/aggressor_vwap_flow_contradiction_relay_controls_2023_2026");RESULT=Path("results/aggressor_vwap_flow_contradiction_relay_support_2026-08-09.json")
FEATURE_COLUMNS=("source_day","decision_time","feature_available_time","source_valid","buy_base","buy_quote","sell_base","sell_quote","buy_vwap","sell_vwap","vwap_separation","signed_taker_flow","window_return","flow_contradiction")
CLOCK_COLUMNS=("candidate","control","split","source_day","decision_time","feature_available_time","entry_time","exit_time","side","buy_vwap","sell_vwap","vwap_separation","signed_taker_flow","window_return","flow_contradiction")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
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
 required=["source_day","first_open","last_close","total_base","total_quote","buy_base","buy_quote","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if not set(required).issubset(raw.columns):raise ValueError("AVFCR source schema drift")
 f=raw[required].copy()
 for c in ("source_day","first_ts","last_ts"):f[c]=pd.to_datetime(f[c],utc=True,errors="coerce")
 for c in ("first_open","last_close","total_base","total_quote","buy_base","buy_quote"):f[c]=pd.to_numeric(f[c],errors="coerce")
 rows=[]
 for r in f.itertuples(index=False):
  sell_base=float(r.total_base-r.buy_base);sell_quote=float(r.total_quote-r.buy_quote)
  valid=bool(pd.notna(r.source_day) and r.source_rows==360 and r.distinct_rows==360 and r.first_ts==r.source_day+pd.Timedelta(hours=18) and r.last_ts==r.source_day+pd.Timedelta(hours=23,minutes=59) and r.coherent and np.isfinite([r.first_open,r.last_close,r.total_base,r.total_quote,r.buy_base,r.buy_quote,sell_base,sell_quote]).all() and min(r.first_open,r.last_close,r.total_base,r.total_quote,r.buy_base,r.buy_quote,sell_base,sell_quote)>0)
  bv=float(r.buy_quote/r.buy_base) if valid else np.nan;sv=float(sell_quote/sell_base) if valid else np.nan
  sep=float(np.log(bv/sv)) if valid else np.nan;flow=float((2*r.buy_quote-r.total_quote)/r.total_quote) if valid else np.nan;ret=float(np.log(r.last_close/r.first_open)) if valid else np.nan
  contradiction=bool(valid and sep!=0 and flow!=0 and np.sign(sep)==-np.sign(flow))
  rows.append({"source_day":r.source_day,"decision_time":r.source_day+pd.Timedelta(days=1),"feature_available_time":r.source_day+pd.Timedelta(days=1),"source_valid":valid,"buy_base":float(r.buy_base),"buy_quote":float(r.buy_quote),"sell_base":sell_base,"sell_quote":sell_quote,"buy_vwap":bv,"sell_vwap":sv,"vwap_separation":sep,"signed_taker_flow":flow,"window_return":ret,"flow_contradiction":contradiction})
 return pd.DataFrame(rows,columns=FEATURE_COLUMNS)
def signal(f:pd.DataFrame,control:str="primary")->pd.Series:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 primary=f.source_valid&f.flow_contradiction&f.vwap_separation.ne(0)
 if control=="no_flow_contradiction":eligible=f.source_valid&f.vwap_separation.ne(0);stat=f.vwap_separation
 elif control=="flow_side":eligible=primary&f.signed_taker_flow.ne(0);stat=f.signed_taker_flow
 elif control=="window_return_side":eligible=primary&f.window_return.ne(0);stat=f.window_return
 else:eligible=primary;stat=f.vwap_separation
 side=np.sign(stat).astype("Int64").fillna(0).astype(int)
 if control=="direction_flip":side=-side
 return side.where(eligible,0)
def build_clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 sides=signal(f,control);rows=[];reserved=None
 for i in f.index[sides.ne(0)]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"source_day":f.at[i,"source_day"],"decision_time":decision,"feature_available_time":f.at[i,"feature_available_time"],"entry_time":entry,"exit_time":exit_,"side":int(sides.at[i]),**{c:f.at[i,c] for c in CLOCK_COLUMNS[9:]}})
 return pd.DataFrame(rows,columns=CLOCK_COLUMNS)
def stats(clock:pd.DataFrame,split:str)->dict[str,float|int]:
 x=clock[clock.split.eq(split)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 l=int(x.side.eq(1).sum());s=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts()
 return {"events":len(x),"longs":l,"shorts":s,"minority_side_share":min(l,s)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("AVFCR preregistration drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);raw=load_source();features=build_features(raw);primary=build_clock(features);controls={n:build_clock(features,n) for n in CONTROLS}
 SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(features,FEATURES);_write_gzip_csv(primary,CLOCK)
 for n,v in controls.items():_write_gzip_csv(v,CONTROL_DIR/f"{n}.csv.gz")
 sc={"protocol_version":"avfcr_12_source_v1","query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"window":[START.isoformat(),END.isoformat()],"aggregate_rows":len(raw),"features":{"path":str(FEATURES),"sha256":sha(FEATURES),"rows":len(features)},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False};sm={**sc,"manifest_hash":canonical_hash(sc)};MANIFEST.write_text(json.dumps(sm,indent=2,allow_nan=False)+"\n")
 support={n:stats(primary,n) for n in SPLITS};checks={}
 for n,v in support.items():checks[f"{n}_minimum_events"]=v["events"]>=MINIMUM[n];checks[f"{n}_side_balance"]=v["minority_side_share"]>=.2;checks[f"{n}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values());core={"protocol_version":"avfcr_12_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"execution_prices_opened":False,"postentry_return_or_pnl_opened":False,"gross9_rows_opened":False,"rv20_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(v),"promotion_authorized":False} for n,v in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};out={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n");return out
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
