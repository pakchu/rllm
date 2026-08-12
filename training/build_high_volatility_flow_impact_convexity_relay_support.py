"""Build source-only HVFIC-8 clocks."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_high_volatility_flow_impact_convexity_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="d4f9ad566c8897237953751348fbbbeb7a1ea92b7dc933f795a9108f0156329a";START=pd.Timestamp("2023-03-01T01:00:00Z");END=pd.Timestamp("2026-08-01T00:00:00Z");SOURCE_DIR=Path("data/high_volatility_flow_impact_convexity_relay_sources_2023_2026");PANEL=SOURCE_DIR/"eight_hour_convexity_panel.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/high_volatility_flow_impact_convexity_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_flow_impact_convexity_relay_controls_2023_2026");RESULT=Path("results/high_volatility_flow_impact_convexity_relay_support_2026-08-13.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};CONTROLS=("no_convexity_tail","no_variation_gate","linear_impact","one_block_stale_convexity","direction_flip","same_clock_forced_long");COLS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","low_beta","high_beta","impact_convexity","convexity_rank","aggregate_flow","block_return","variation","variation_rank")
QUERY="""SELECT ts,open,high,low,close,quote_asset_volume,taker_buy_quote FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(v:pd.Series)->pd.Series:
 x=pd.to_numeric(v,errors="coerce");o=pd.Series(np.nan,index=x.index,dtype=float);h=[]
 for i,c in x.items():
  q=np.asarray(h[-270:],float)
  if math.isfinite(c) and len(q)>=180:o.at[i]=((q<c).sum()+.5*(q==c).sum())/len(q)
  if math.isfinite(c):h.append(float(c))
 return o
def engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def prepare(x):
 x=x.copy();x.ts=pd.to_datetime(x.ts,utc=True)
 for c in ("open","high","low","close","quote_asset_volume","taker_buy_quote"):x[c]=pd.to_numeric(x[c],errors="coerce")
 return x.drop_duplicates("ts",keep=False).set_index("ts").sort_index()
def block_metrics(w:pd.DataFrame)->dict[str,Any]:
 f=np.isfinite(w[["open","high","low","close","quote_asset_volume","taker_buy_quote"]]).all(axis=1);p=w[["open","high","low","close"]].gt(0).all(axis=1);shape=w.high.ge(w[["open","close"]].max(axis=1))&w.low.le(w[["open","close"]].min(axis=1))&w.high.ge(w.low);vol=w[["quote_asset_volume","taker_buy_quote"]].ge(0).all(axis=1)&w.taker_buy_quote.le(w.quote_asset_volume)
 if len(w)!=480 or not bool((f&p&shape&vol).all()):return {"source_valid":False}
 g=np.arange(480)//5;five=w.groupby(g).agg(open=("open","first"),close=("close","last"),quote=("quote_asset_volume","sum"),buy=("taker_buy_quote","sum"));flow=(2*five.buy-five.quote)/five.quote;ret=np.log(five.close/five.open);med=float(flow.abs().median());hi=flow.abs().gt(med);lo=flow.abs().lt(med);hd=float(np.square(flow[hi]).sum());ld=float(np.square(flow[lo]).sum());hb=float(np.dot(flow[hi],ret[hi])/hd) if hd>0 else np.nan;lb=float(np.dot(flow[lo],ret[lo])/ld) if ld>0 else np.nan;conv=float(np.log(hb/lb)) if hb>0 and lb>0 else np.nan;agg=float(2*w.taker_buy_quote.sum()-w.quote_asset_volume.sum());br=float(np.log(w.close.iloc[-1]/w.open.iloc[0]));var=float(np.sqrt(np.square(ret).sum()));valid=bool(hi.sum()>=32 and lo.sum()>=32 and np.isfinite([hb,lb,conv,agg,br,var]).all() and conv>0 and agg!=0 and br!=0 and var>0)
 return {"source_valid":valid,"low_beta":lb,"high_beta":hb,"impact_convexity":conv,"aggregate_flow":agg,"block_return":br,"variation":var,"high_bars":int(hi.sum()),"low_bars":int(lo.sum())}
def build_panel(bars):
 x=prepare(bars);rows=[]
 for d in pd.date_range(START,END,freq="8h",inclusive="left"):
  g=pd.date_range(d-pd.Timedelta(hours=8),d,freq="1min",inclusive="left");w=x.reindex(g);rows.append({"decision_time":d,"source_rows":int(w.notna().all(axis=1).sum()),**block_metrics(w)})
 p=pd.DataFrame(rows);p["convexity_rank"]=rank(p.impact_convexity.where(p.source_valid));p["variation_rank"]=rank(p.variation.where(p.source_valid));return p
def materialize():
 from sqlalchemy import text
 e=engine()
 with e.connect() as c:b=pd.read_sql_query(text(QUERY),c,params={"start":(START-pd.Timedelta(hours=8)).to_pydatetime(),"end":END.to_pydatetime()})
 e.dispose();x=build_panel(b);SOURCE_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(x,PANEL);core={"protocol_version":"hvfic_8_sources_v1","query":QUERY,"table":"bars_binance","symbol":"BTCUSDT","interval":"1m","window":[START.isoformat(),END.isoformat()],"candidate_incidence_opened":False,"postentry_outcomes_opened":False,"gross9_rows_opened":False,"no_imputation":True,"output":{"path":str(PANEL),"sha256":sha(PANEL),"rows":len(x),"valid_rows":int(x.source_valid.sum())}};r={**core,"manifest_hash":prereg.canonical_hash(core)};MANIFEST.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
def features():
 x=pd.read_csv(PANEL);x.decision_time=pd.to_datetime(x.decision_time,utc=True,format="mixed");x.source_valid=x.source_valid.astype(str).str.lower().eq("true")
 for c in ("low_beta","high_beta","impact_convexity","convexity_rank","aggregate_flow","block_return","variation","variation_rank"):x[c]=pd.to_numeric(x[c],errors="coerce")
 return x
def conditions(x,control):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 conv=x.impact_convexity;cr=x.convexity_rank
 if control=="one_block_stale_convexity":conv=conv.shift(1);cr=cr.shift(1)
 if control=="linear_impact":conv=-conv;cr=rank(conv.where(x.source_valid))
 tail=pd.Series(True,index=x.index) if control=="no_convexity_tail" else cr.ge(.75);vg=pd.Series(True,index=x.index) if control=="no_variation_gate" else x.variation_rank.ge(.65);side=np.sign(x.aggregate_flow).fillna(0).astype(int);agree=np.sign(x.block_return).eq(side);eligible=x.source_valid&np.isfinite(conv)&conv.gt(0)&tail&vg&side.ne(0)&agree;active=eligible&x.source_valid.shift(1,fill_value=False)&~eligible.shift(1,fill_value=False)
 if control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=x.index)
 return active,side
def clock(x,control="primary"):
 active,side=conditions(x,control);rows=[]
 for i in x.index[active]:
  d=pd.Timestamp(x.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=8);split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),**{c:float(x.at[i,c]) for c in COLS[8:]}})
 return pd.DataFrame(rows,columns=COLS)
def stats(c,n):
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVFIC prereg drift")
 sm=materialize();x=features();primary=clock(x);controls={n:clock(x,n) for n in CONTROLS};CLOCK.parent.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(primary,CLOCK)
 for n,c in controls.items():_write_gzip_csv(c,CONTROL_DIR/f"{n}.csv.gz")
 support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,s in support.items() for k,v in ((f"{n}_minimum_events",s["events"]>=MINIMUM[n]),(f"{n}_side_balance",s["minority_side_share"]>=.2),(f"{n}_month_concentration",s["max_month_share"]<=.45))};passed=all(checks.values());reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvfic_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(c),"promotion_authorized":False} for n,c in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":prereg.canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
