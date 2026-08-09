"""Build source-only HVPIAR-8 clocks before Gross9 or economics."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
if __package__ in (None,""):
 import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import preregister_high_volatility_price_impact_asymmetry_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV=Path("/home/pakchu/rllm/.env");PREREG_SHA="a3cb689a6938708b76740e819a65db8d7e2c412bfaaaec37eda03f06bff3a55f";WRITER=Path("training/build_binance_aggtrade_microstructure.py");WRITER_SHA="dc09b9b2d8838f6b1e64ef73636e39aacd7f843720a20ab01e566660d7d2c47a";START=pd.Timestamp("2023-01-01T00:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z")
STATE=Path("data/high_volatility_price_impact_asymmetry_relay_sources_2023_2026/eight_hour_states.csv.gz");CLOCK=Path("data/high_volatility_price_impact_asymmetry_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_price_impact_asymmetry_relay_controls_2023_2026");RESULT=Path("results/high_volatility_price_impact_asymmetry_relay_support_2026-08-09.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_asymmetry_tail","no_variation_gate","one_block_stale_asymmetry","direction_flip","same_clock_forced_long")
QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,(array_agg(open ORDER BY ts))[1] AS open,(array_agg(close ORDER BY ts DESC))[1] AS close,sum(quote_asset_volume) AS quote_turnover,count(*) AS source_rows FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","impact_asymmetry","asymmetry_rank","variation","variation_rank","positive_bars","negative_bars","upside_impact","downside_impact")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(v:pd.Series)->pd.Series:
 x=pd.to_numeric(v,errors="coerce");o=pd.Series(np.nan,index=x.index,dtype=float);h=[]
 for i,c in x.items():
  q=np.asarray(h[-270:],float)
  if np.isfinite(c) and len(q)>=252:o.at[i]=((q<c).sum()+.5*(q==c).sum())/len(q)
  if np.isfinite(c):h.append(float(c))
 return o
def load_source()->pd.DataFrame:
 from preprocessing.live_db_features import sqlalchemy_engine_from_env
 from sqlalchemy import text
 engine=sqlalchemy_engine_from_env(ENV)
 try:
  with engine.connect() as c:return pd.read_sql_query(text(QUERY),c,params={"start":(START-pd.Timedelta(minutes=5)).to_pydatetime(),"end":END.to_pydatetime()})
 finally:engine.dispose()
def score_states(raw:pd.DataFrame)->pd.DataFrame:
 x=raw.copy();x["date"]=pd.to_datetime(x.date,utc=True);x=x.sort_values("date").set_index("date");grid=pd.date_range(START-pd.Timedelta(minutes=5),END,freq="5min",inclusive="left");x=x.reindex(grid);x["open"]=pd.to_numeric(x.open,errors="coerce");x["close"]=pd.to_numeric(x.close,errors="coerce");x["quote_turnover"]=pd.to_numeric(x.quote_turnover,errors="coerce");x["valid"]=x.source_rows.eq(5)&np.isfinite(x[["open","close","quote_turnover"]]).all(axis=1)&x[["open","close"]].gt(0).all(axis=1)&x.quote_turnover.gt(0);x["r"]=np.log(x.close/x.close.shift(1));x=x[x.index>=START];rows=[]
 for start,w in x.groupby(x.index.floor("8h"),sort=True):
  expected=pd.date_range(start,start+pd.Timedelta(hours=8),freq="5min",inclusive="left");w=w.reindex(expected);pos=w.r.gt(0);neg=w.r.lt(0);valid=len(w)==96 and w.index.equals(expected) and w.valid.eq(True).all() and np.isfinite(w.r).all() and int(pos.sum())>=16 and int(neg.sum())>=16;up=float(w.loc[pos,"r"].sum()/w.loc[pos,"quote_turnover"].sum()) if valid else np.nan;down=float((-w.loc[neg,"r"].sum())/w.loc[neg,"quote_turnover"].sum()) if valid else np.nan;asym=float(np.log(down/up)) if valid and up>0 and down>0 else np.nan;var=float(np.sqrt(np.square(w.r).sum())) if valid else np.nan;rows.append({"block_start":start,"decision_time":start+pd.Timedelta(hours=8),"source_valid":bool(valid and np.isfinite([up,down,asym,var]).all() and asym!=0),"positive_bars":int(pos.sum()),"negative_bars":int(neg.sum()),"upside_impact":up,"downside_impact":down,"impact_asymmetry":asym,"variation":var})
 e=pd.DataFrame(rows);e["asymmetry_rank"]=rank(e.impact_asymmetry.abs().where(e.source_valid));e["variation_rank"]=rank(e.variation.where(e.source_valid));return e
def build_clock(states:pd.DataFrame,control:str="primary")->pd.DataFrame:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 used=states.shift(1) if control=="one_block_stale_asymmetry" else states;valid=used.source_valid.eq(True)&used.impact_asymmetry.ne(0)&np.isfinite(used[["impact_asymmetry","asymmetry_rank","variation","variation_rank"]]).all(axis=1)
 if control=="one_block_stale_asymmetry":valid&=states.block_start.sub(used.block_start).eq(pd.Timedelta(hours=8))
 ag=pd.Series(True,index=states.index) if control=="no_asymmetry_tail" else used.asymmetry_rank.ge(.8);vg=pd.Series(True,index=states.index) if control=="no_variation_gate" else (states.variation_rank if control=="one_block_stale_asymmetry" else used.variation_rank).ge(.65);active=valid&ag&vg;side=-np.sign(used.impact_asymmetry).fillna(0).astype(int)
 if control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=states.index)
 rows=[]
 for i in states.index[active]:
  decision=pd.Timestamp(states.at[i,"decision_time"]);entry=decision+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8);split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  u=used.loc[i];rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"impact_asymmetry":float(u.impact_asymmetry),"asymmetry_rank":float(u.asymmetry_rank),"variation":float(states.at[i,"variation"]),"variation_rank":float(states.at[i,"variation_rank"]),"positive_bars":int(u.positive_bars),"negative_bars":int(u.negative_bars),"upside_impact":float(u.upside_impact),"downside_impact":float(u.downside_impact)})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict[str,Any]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=pd.to_datetime(x.entry_time,utc=True).dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 bindings={prereg.DEFAULT_OUTPUT:PREREG_SHA,WRITER:WRITER_SHA,prereg.MARKET:prereg.MARKET_SHA}
 for p,h in bindings.items():
  if sha(p)!=h:raise RuntimeError(f"HVPIAR binding drift: {p}")
 raw=load_source();states=score_states(raw);primary=build_clock(states);controls={n:build_clock(states,n) for n in CONTROLS};STATE.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(states,STATE);_write_gzip_csv(primary,CLOCK)
 for n,x in controls.items():_write_gzip_csv(x,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,x in support.items() for k,v in ((f"{n}_minimum_events",x["events"]>=MINIMUM[n]),(f"{n}_side_balance",x["minority_side_share"]>=.2),(f"{n}_month_concentration",x["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvpiar_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"bindings":{str(p):h for p,h in bindings.items()},"source_query":{"sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"five_minute_rows":len(raw)},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"source_state":{"path":str(STATE),"sha256":sha(STATE),"rows":len(states)},"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(x),"promotion_authorized":False} for n,x in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":prereg.canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
