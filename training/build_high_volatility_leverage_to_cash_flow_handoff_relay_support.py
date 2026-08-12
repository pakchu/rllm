"""Build source-only HVLCFH-8 clocks before novelty or economics."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_high_volatility_leverage_to_cash_flow_handoff_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="b3fb5032b75f40c3fa18d5f8c8973e3a16ca1d894f09675d196b992c4de5fcb9";START=pd.Timestamp("2023-03-01T03:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR=Path("data/high_volatility_leverage_to_cash_flow_handoff_relay_sources_2023_2026");PANEL=SOURCE_DIR/"eight_hour_flow_handoff_panel.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/high_volatility_leverage_to_cash_flow_handoff_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_leverage_to_cash_flow_handoff_relay_controls_2023_2026");RESULT=Path("results/high_volatility_leverage_to_cash_flow_handoff_relay_support_2026-08-13.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_handoff_tail","no_variation_gate","no_onset","one_block_stale_handoff","direction_flip","same_clock_forced_long")
COLUMNS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","perp_first_flow","spot_first_flow","perp_second_flow","spot_second_flow","handoff_strength","handoff_rank","variation","variation_rank")
QUERY="""SELECT ts,open,high,low,close,quote_asset_volume,taker_buy_quote FROM {table} WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(v:pd.Series,minimum:int=180)->pd.Series:
 x=pd.to_numeric(v,errors="coerce");o=pd.Series(np.nan,index=x.index,dtype=float);h=[]
 for i,c in x.items():
  q=np.asarray(h[-270:],float)
  if math.isfinite(c) and len(q)>=minimum:o.at[i]=((q<c).sum()+.5*(q==c).sum())/len(q)
  if math.isfinite(c):h.append(float(c))
 return o
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def prepare(x:pd.DataFrame)->pd.DataFrame:
 x=x.copy();x["ts"]=pd.to_datetime(x.ts,utc=True)
 for c in ("open","high","low","close","quote_asset_volume","taker_buy_quote"):x[c]=pd.to_numeric(x[c],errors="coerce")
 return x.drop_duplicates("ts",keep=False).set_index("ts").sort_index()
def valid(x:pd.DataFrame)->bool:
 f=np.isfinite(x[["open","high","low","close","quote_asset_volume","taker_buy_quote"]]).all(axis=1);p=x[["open","high","low","close"]].gt(0).all(axis=1);shape=x.high.ge(x[["open","close"]].max(axis=1))&x.low.le(x[["open","close"]].min(axis=1))&x.high.ge(x.low);vol=x[["quote_asset_volume","taker_buy_quote"]].ge(0).all(axis=1)&x.taker_buy_quote.le(x.quote_asset_volume);return len(x)==480 and bool((f&p&shape&vol).all())
def half_flow(x:pd.DataFrame,a:int,b:int)->float:
 q=float(x.quote_asset_volume.iloc[a:b].sum());buy=float(x.taker_buy_quote.iloc[a:b].sum());return (2*buy-q)/q if q>0 else np.nan
def build_panel(perp:pd.DataFrame,spot:pd.DataFrame)->pd.DataFrame:
 perp,spot=prepare(perp),prepare(spot);rows=[]
 for d in pd.date_range(START,END,freq="8h",inclusive="left"):
  g=pd.date_range(d-pd.Timedelta(hours=8),d,freq="1min",inclusive="left");p=perp.reindex(g);s=spot.reindex(g);ok=valid(p) and valid(s)
  if ok:
   pf,sf,ps,ss=half_flow(p,0,240),half_flow(s,0,240),half_flow(p,240,480),half_flow(s,240,480);variation=float(np.sqrt(np.square(np.log(p.close.to_numpy(float)/p.open.to_numpy(float))).sum()));flows=np.array([pf,sf,ps,ss]);signs=np.sign(flows);persistent=bool(np.isfinite(flows).all() and (signs!=0).all() and (signs==signs[0]).all());first_gap=abs(pf)-abs(sf);second_gap=abs(ss)-abs(ps);ordered=persistent and first_gap>0 and second_gap>0;strength=min(first_gap,second_gap) if ordered else np.nan;ok=bool(variation>0 and np.isfinite(variation))
  else:pf=sf=ps=ss=variation=strength=np.nan;persistent=ordered=False
  rows.append({"decision_time":d,"source_valid":ok,"ordered_handoff":ordered,"side":int(np.sign(pf)) if persistent else 0,"perp_first_flow":pf,"spot_first_flow":sf,"perp_second_flow":ps,"spot_second_flow":ss,"handoff_strength":strength,"variation":variation,"perp_source_rows":int(p.notna().all(axis=1).sum()),"spot_source_rows":int(s.notna().all(axis=1).sum())})
 x=pd.DataFrame(rows);x["handoff_rank"]=rank(x.handoff_strength.where(x.ordered_handoff));x["variation_rank"]=rank(x.variation.where(x.source_valid));return x
def materialize()->dict[str,Any]:
 from sqlalchemy import text
 e=postgres_engine();params={"start":(START-pd.Timedelta(hours=8)).to_pydatetime(),"end":END.to_pydatetime()}
 with e.connect() as c:perp=pd.read_sql_query(text(QUERY.format(table="bars_binance")),c,params=params);spot=pd.read_sql_query(text(QUERY.format(table="bars_binance_spot")),c,params=params)
 e.dispose();x=build_panel(perp,spot);SOURCE_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(x,PANEL);core={"protocol_version":"hvlcfh_8_sources_v1","queries":{"perpetual":QUERY.format(table="bars_binance"),"spot":QUERY.format(table="bars_binance_spot")},"tables":["bars_binance","bars_binance_spot"],"symbol":"BTCUSDT","interval":"1m","window":[START.isoformat(),END.isoformat()],"candidate_incidence_opened":False,"postentry_outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True,"output":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(x),"valid_rows":int(x.source_valid.sum())}};r={**core,"manifest_hash":prereg.canonical_hash(core)};MANIFEST.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
def features()->pd.DataFrame:
 x=pd.read_csv(PANEL);x["decision_time"]=pd.to_datetime(x.decision_time,utc=True,format="mixed");x["source_valid"]=x.source_valid.astype(str).str.lower().eq("true");x["ordered_handoff"]=x.ordered_handoff.astype(str).str.lower().eq("true")
 for c in ("side","perp_first_flow","spot_first_flow","perp_second_flow","spot_second_flow","handoff_strength","handoff_rank","variation","variation_rank"):x[c]=pd.to_numeric(x[c],errors="coerce")
 return x
def conditions(x:pd.DataFrame,control:str)->tuple[pd.Series,pd.Series]:
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 ordered=x.ordered_handoff;strength=x.handoff_strength;hr=x.handoff_rank;side=x.side
 if control=="one_block_stale_handoff":ordered=ordered.shift(1,fill_value=False);strength=strength.shift(1);hr=hr.shift(1);side=side.shift(1)
 tail=pd.Series(True,index=x.index) if control=="no_handoff_tail" else hr.ge(.70);vg=pd.Series(True,index=x.index) if control=="no_variation_gate" else x.variation_rank.ge(.65);eligible=x.source_valid&ordered&np.isfinite(strength)&tail&vg&side.ne(0)
 active=eligible if control=="no_onset" else eligible&x.source_valid.shift(1,fill_value=False)&~eligible.shift(1,fill_value=False)
 if control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=x.index)
 return active,side
def clock(x:pd.DataFrame,control:str="primary")->pd.DataFrame:
 active,side=conditions(x,control);rows=[]
 for i in x.index[active]:
  d=pd.Timestamp(x.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8);split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),**{c:float(x.at[i,c]) for c in COLUMNS[8:]}})
 return pd.DataFrame(rows,columns=COLUMNS)
def stats(c:pd.DataFrame,n:str)->dict[str,Any]:
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVLCFH prereg drift")
 sm=materialize();x=features();primary=clock(x);controls={n:clock(x,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,c in controls.items():_write_gzip_csv(c,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,s in support.items() for k,v in ((f"{n}_minimum_events",s["events"]>=MINIMUM[n]),(f"{n}_side_balance",s["minority_side_share"]>=.2),(f"{n}_month_concentration",s["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvlcfh_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(c),"promotion_authorized":False} for n,c in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":prereg.canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
