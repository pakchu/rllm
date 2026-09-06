"""Build source-only HVMASR-8 clocks before Gross9 or economics."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_high_volatility_meme_attention_saturation_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV=Path("/home/pakchu/rllm/.env");PREREG_SHA="8237a9843254807a1e11410d8cc516c79208f7e0ec161c904dcd4d32dfa82198";WRITER=Path("training/build_binance_aggtrade_microstructure.py");WRITER_SHA="dc09b9b2d8838f6b1e64ef73636e39aacd7f843720a20ab01e566660d7d2c47a"
START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");STATE=Path("data/high_volatility_meme_attention_saturation_reversal_sources_2023_2026/four_hour_states.csv.gz");CLOCK=Path("data/high_volatility_meme_attention_saturation_reversal_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_meme_attention_saturation_reversal_controls_2023_2026");RESULT=Path("results/high_volatility_meme_attention_saturation_reversal_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_doge_return_tail","no_doge_turnover_tail","no_btc_variation_gate","one_block_stale_doge","direction_flip","same_clock_forced_long")
QUERY="""
WITH minute AS (
 SELECT symbol,ts,open,close,quote_asset_volume,
        lag(close) OVER (PARTITION BY symbol ORDER BY ts) AS prev_close
 FROM bars_binance
 WHERE interval='1m' AND symbol IN ('BTCUSDT','DOGEUSDT')
   AND ts>=:lag_start AND ts<:end
), kept AS (
 SELECT *,to_timestamp(floor(extract(epoch FROM ts)/14400)*14400) AS block_start
 FROM minute WHERE ts>=:start
)
SELECT symbol,block_start,count(*) AS rows,min(ts) AS first_ts,max(ts) AS last_ts,
       (array_agg(open ORDER BY ts))[1] AS first_open,
       (array_agg(close ORDER BY ts DESC))[1] AS last_close,
       sum(quote_asset_volume) AS quote_turnover,
       sqrt(sum(power(ln(close/prev_close),2))) AS variation
FROM kept GROUP BY symbol,block_start ORDER BY block_start,symbol
"""
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","doge_return","doge_abs_return_rank","doge_turnover","doge_turnover_rank","btc_variation","btc_variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(v:pd.Series)->pd.Series:
 x=pd.to_numeric(v,errors="coerce");o=pd.Series(np.nan,index=x.index,dtype=float);h=[]
 for i,c in x.items():
  q=np.asarray(h[-270:],float)
  if np.isfinite(c) and len(q)>=180:o.at[i]=((q<c).sum()+.5*(q==c).sum())/len(q)
  if np.isfinite(c):h.append(float(c))
 return o
def load_source()->pd.DataFrame:
 from preprocessing.live_db_features import sqlalchemy_engine_from_env
 from sqlalchemy import text
 engine=sqlalchemy_engine_from_env(ENV)
 try:
  with engine.connect() as c:raw=pd.read_sql_query(text(QUERY),c,params={"lag_start":(START-pd.Timedelta(minutes=1)).to_pydatetime(),"start":START.to_pydatetime(),"end":END.to_pydatetime()})
 finally:engine.dispose()
 return raw
def score_states(raw:pd.DataFrame)->pd.DataFrame:
 x=raw.copy();x["block_start"]=pd.to_datetime(x.block_start,utc=True);x["block_end"]=x.block_start+pd.Timedelta(hours=4);x["valid"]=x.rows.eq(240)&pd.to_datetime(x.first_ts,utc=True).eq(x.block_start)&pd.to_datetime(x.last_ts,utc=True).eq(x.block_end-pd.Timedelta(minutes=1))&np.isfinite(x[["first_open","last_close","quote_turnover","variation"]].astype(float)).all(axis=1)&x[["first_open","last_close","quote_turnover"]].astype(float).gt(0).all(axis=1)
 p=x.pivot(index="block_start",columns="symbol",values=["valid","first_open","last_close","quote_turnover","variation","rows"]).sort_index();e=pd.DataFrame(index=p.index);e["decision_time"]=e.index+pd.Timedelta(hours=4);e["source_valid"]=p["valid"]["BTCUSDT"].eq(True)&p["valid"]["DOGEUSDT"].eq(True);e["doge_return"]=np.log(pd.to_numeric(p["last_close"]["DOGEUSDT"])/pd.to_numeric(p["first_open"]["DOGEUSDT"])).where(e.source_valid);e["doge_turnover"]=pd.to_numeric(p["quote_turnover"]["DOGEUSDT"]).where(e.source_valid);e["btc_variation"]=pd.to_numeric(p["variation"]["BTCUSDT"]).where(e.source_valid);e["doge_abs_return_rank"]=rank(e.doge_return.abs());e["doge_turnover_rank"]=rank(e.doge_turnover);e["btc_variation_rank"]=rank(e.btc_variation);return e.reset_index()
def build_clock(states:pd.DataFrame,control:str="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=states.shift(1) if control=="one_block_stale_doge" else states;valid=used.source_valid.eq(True)&used.doge_return.ne(0)&np.isfinite(used[["doge_return","doge_abs_return_rank","doge_turnover","doge_turnover_rank"]]).all(axis=1)&np.isfinite(states[["btc_variation","btc_variation_rank"]]).all(axis=1)
 if control=="one_block_stale_doge":valid&=states.block_start.sub(used.block_start).eq(pd.Timedelta(hours=4))
 rg=pd.Series(True,index=states.index) if control=="no_doge_return_tail" else used.doge_abs_return_rank.ge(.75);tg=pd.Series(True,index=states.index) if control=="no_doge_turnover_tail" else used.doge_turnover_rank.ge(.85);vg=pd.Series(True,index=states.index) if control=="no_btc_variation_gate" else states.btc_variation_rank.ge(.65);active=valid&rg&tg&vg;side=-np.sign(used.doge_return).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=states.index)
 rows=[];reserved=None
 for i in states.index[active]:
  decision=pd.Timestamp(states.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8)
  if reserved is not None and entry<reserved:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  reserved=exit_;u=used.loc[i];rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"doge_return":float(u.doge_return),"doge_abs_return_rank":float(u.doge_abs_return_rank),"doge_turnover":float(u.doge_turnover),"doge_turnover_rank":float(u.doge_turnover_rank),"btc_variation":float(states.at[i,"btc_variation"]),"btc_variation_rank":float(states.at[i,"btc_variation_rank"])})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict[str,Any]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 bindings={prereg.DEFAULT_OUTPUT:PREREG_SHA,WRITER:WRITER_SHA,prereg.MARKET:prereg.MARKET_SHA}
 for p,h in bindings.items():
  if sha(p)!=h:raise RuntimeError(f"HVMASR binding drift: {p}")
 raw=load_source();states=score_states(raw);primary=build_clock(states);controls={n:build_clock(states,n) for n in CONTROLS};STATE.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(states,STATE);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=MINIMUM[n]),(f"{n}_side_balance",x["minority_side_share"]>=.2),(f"{n}_month_concentration",x["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvmasr_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"bindings":{str(p):h for p,h in bindings.items()},"source_query":{"sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"aggregate_rows":len(raw),"source_only":True},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"source_state":{"path":str(STATE),"sha256":sha(STATE),"rows":len(states)},"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":prereg.canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
