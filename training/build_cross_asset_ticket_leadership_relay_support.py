"""Materialize source-only CATLR-12 support clocks from read-only PostgreSQL."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_cross_asset_ticket_leadership_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
PREREG_SHA="5a578cf1190c2edfc54836b4b6ca7c952d829983cae8d9ce09c46d1b9142ecde";ENV=Path("/home/pakchu/rllm/.env");SYMBOLS=("BTCUSDT","ADAUSDT","BNBUSDT","DOGEUSDT","ETHUSDT","SOLUSDT","XRPUSDT");ALTS=SYMBOLS[1:];START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR=Path("data/cross_asset_ticket_leadership_relay_sources_2023_2026");SNAPSHOT=SOURCE_DIR/"ticket_leadership_features.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/cross_asset_ticket_leadership_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/cross_asset_ticket_leadership_relay_controls_2023_2026");RESULT=Path("results/cross_asset_ticket_leadership_relay_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("btc_ticket_only","alt_ticket_suppression_only","no_return_tail","no_volatility_gate","one_block_stale_leadership","direction_flip")
QUERY="""SELECT symbol,date_trunc('day',ts)+(floor(extract(hour from ts)/8)*interval '8 hours') AS block_start,(array_agg(open ORDER BY ts))[1] AS block_open,(array_agg(close ORDER BY ts DESC))[1] AS block_close,max(high) AS block_high,min(low) AS block_low,sum(quote_asset_volume) AS quote_volume,sum(number_of_trades) AS trade_count,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts FROM bars_binance WHERE symbol = ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end GROUP BY symbol,block_start ORDER BY block_start,symbol"""
COLUMNS=("candidate","control","split","block_start","decision_time","feature_available_time","entry_time","exit_time","side","ticket_leadership","leadership_rank","btc_return","return_rank","range_vol","volatility_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def rank(v:pd.Series,lookback:int=90,minimum:int=60)->pd.Series:
 x=pd.to_numeric(v,errors="coerce").to_numpy(float);o=np.full(len(x),np.nan);h=[]
 for i,c in enumerate(x):
  q=h[-lookback:]
  if np.isfinite(c) and len(q)>=minimum:
   a=np.asarray(q);o[i]=(np.sum(a<c)+.5*np.sum(a==c))/len(a)
  if np.isfinite(c):h.append(c)
 return pd.Series(o,index=v.index)
def causal_log_median_ratio(v:pd.Series)->pd.Series:
 x=pd.to_numeric(v,errors="coerce").to_numpy(float);o=np.full(len(x),np.nan);h=[]
 for i,c in enumerate(x):
  q=h[-90:]
  if np.isfinite(c) and c>0 and len(q)>=60:
   m=float(np.median(q));o[i]=math.log(c/m) if m>0 else np.nan
  if np.isfinite(c) and c>0:h.append(c)
 return pd.Series(o,index=v.index)
def engine():
 from preprocessing.live_db_features import sqlalchemy_engine_from_env
 return sqlalchemy_engine_from_env(ENV)
def features()->tuple[pd.DataFrame,dict[str,Any]]:
 from sqlalchemy import text
 e=engine()
 try:
  with e.connect() as c:r=pd.read_sql_query(text(QUERY),c,params={"symbols":list(SYMBOLS),"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:e.dispose()
 for c in ("block_start","first_ts","last_ts"):r[c]=pd.to_datetime(r[c],utc=True)
 for c in ("block_open","block_close","block_high","block_low","quote_volume","trade_count"):r[c]=pd.to_numeric(r[c],errors="coerce")
 r["valid"]=r.source_rows.eq(480)&r.distinct_rows.eq(480)&r.first_ts.eq(r.block_start)&r.last_ts.eq(r.block_start+pd.Timedelta(hours=7,minutes=59))&np.isfinite(r[["block_open","block_close","block_high","block_low","quote_volume","trade_count"]]).all(axis=1)&r[["block_open","block_close","block_high","block_low"]].gt(0).all(axis=1)&r.quote_volume.gt(0)&r.trade_count.gt(0)
 r["average_ticket"]=(r.quote_volume/r.trade_count).where(r.valid)
 r["normalized_ticket"]=r.groupby("symbol",group_keys=False).average_ticket.apply(causal_log_median_ratio)
 wide=r.pivot(index="block_start",columns="symbol",values="normalized_ticket").reindex(columns=SYMBOLS);valid=r.pivot(index="block_start",columns="symbol",values="valid").reindex(columns=SYMBOLS).fillna(False);btc=r[r.symbol.eq("BTCUSDT")].set_index("block_start").sort_index();f=pd.DataFrame(index=wide.index);f["source_valid"]=valid.all(axis=1);f["btc_normalized_ticket"]=wide.BTCUSDT;f["alt_median_normalized_ticket"]=wide[list(ALTS)].median(axis=1);f["ticket_leadership"]=f.btc_normalized_ticket-f.alt_median_normalized_ticket;f["leadership_rank"]=rank(f.ticket_leadership);f["btc_ticket_rank"]=rank(f.btc_normalized_ticket);f["alt_suppression_rank"]=rank(-f.alt_median_normalized_ticket);f["btc_return"]=np.log(btc.block_close/btc.block_open).reindex(f.index);f["return_rank"]=rank(f.btc_return.abs());f["range_vol"]=(btc.block_high.rolling(2,min_periods=2).max()/btc.block_low.rolling(2,min_periods=2).min()-1).reindex(f.index);f["volatility_rank"]=rank(f.range_vol);f=f.reset_index();f["decision_time"]=f.block_start+pd.Timedelta(hours=8);return f,{"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"rows_read":len(r),"valid_symbol_blocks":int(r.valid.sum()),"first":str(r.block_start.min()),"last":str(r.block_start.max())}
def conditions(f:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 lead=f.ticket_leadership;lr=f.leadership_rank
 if control=="btc_ticket_only":lead=f.btc_normalized_ticket;lr=f.btc_ticket_rank
 if control=="alt_ticket_suppression_only":lead=-f.alt_median_normalized_ticket;lr=f.alt_suppression_rank
 if control=="one_block_stale_leadership":lead=lead.shift(1);lr=lr.shift(1)
 ret=pd.Series(True,index=f.index) if control=="no_return_tail" else f.return_rank.ge(.60);vol=pd.Series(True,index=f.index) if control=="no_volatility_gate" else f.volatility_rank.ge(.65);active=f.source_valid&np.isfinite(lead)&lead.gt(0)&lr.ge(.75)&f.btc_return.ne(0)&ret&vol;side=np.sign(f.btc_return);side=-side if control=="direction_flip" else side;return active,side
def clock(f:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(f,control);rows=[];next_allowed=None
 for i in f.index[active]:
  decision=pd.Timestamp(f.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=12)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  next_allowed=exit_;rows.append({"candidate":"CATLR-12","control":control,"split":split,"block_start":f.at[i,"block_start"],"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"ticket_leadership":float(f.at[i,"ticket_leadership"]),"leadership_rank":float(f.at[i,"leadership_rank"]),"btc_return":float(f.at[i,"btc_return"]),"return_rank":float(f.at[i,"return_rank"]),"range_vol":float(f.at[i,"range_vol"]),"volatility_rank":float(f.at[i,"volatility_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict[str,Any]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("CATLR prereg drift")
 f,source=features();primary=clock(f);controls={n:clock(f,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(f,SNAPSHOT);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 mc={"protocol_version":"catlr_12_preentry_source_v1","database":{"env_file":str(ENV),"table":"bars_binance","read_only":True,**source},"features":{"path":str(SNAPSHOT),"sha256":sha(SNAPSHOT),"rows":len(f)},"postentry_outcomes_opened":False,"gross9_rows_opened":False};m={**mc,"manifest_hash":chash(mc)};MANIFEST.write_text(json.dumps(m,indent=2)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=MINIMUM[n]),(f"{n}_side_balance",x["minority_side_share"]>=.2),(f"{n}_month_concentration",x["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"catlr_12_source_support_v1","policy_id":"CATLR-12","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":m["manifest_hash"]},"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
