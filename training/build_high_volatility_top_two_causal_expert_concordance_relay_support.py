"""Build causal HVTCEC-24 top-two concordance clocks before novelty/economics."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from training import preregister_high_volatility_top_two_causal_expert_concordance_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_scheduled_trend_concordance_relay_support import load_market
PREREG_SHA="26f869135ea943147a98449c581a4480373346d3cef5898040a76a0eff6362cc";LOADER=Path("training/build_scheduled_trend_concordance_relay_support.py");LOADER_SHA="8ca554d88506df277434f73e5eb8850426614a880110088eb91aaae3b23c154f";END=pd.Timestamp("2026-08-01T00:00:00Z");SOURCE_DIR=Path("data/high_volatility_top_two_causal_expert_concordance_relay_sources_2020_2026");STATES=SOURCE_DIR/"causal_top_two_expert_states.csv.gz";MANIFEST=SOURCE_DIR/"manifest.json";CLOCK=Path("data/high_volatility_top_two_causal_expert_concordance_relay_clocks_2023_2026.csv.gz");CONTROL_DIR=Path("data/high_volatility_top_two_causal_expert_concordance_relay_controls_2023_2026");RESULT=Path("results/high_volatility_top_two_causal_expert_concordance_relay_support_2026-08-13.json")
SPLITS={"train":(pd.Timestamp("2023-07-01T00:00:00Z"),pd.Timestamp("2024-01-01T00:00:00Z")),"test":(pd.Timestamp("2024-01-01T00:00:00Z"),pd.Timestamp("2025-01-01T00:00:00Z")),"eval":(pd.Timestamp("2025-01-01T00:00:00Z"),pd.Timestamp("2026-01-01T00:00:00Z")),"final":(pd.Timestamp("2026-01-01T00:00:00Z"),END)};MINIMUM={"train":8,"test":12,"eval":12,"final":8};EXPERTS=("momentum_4h","reversal_4h","momentum_24h","reversal_24h");CONTROLS=("no_variation_gate","no_top_two_concordance","one_decision_stale_committee","direction_flip","same_clock_forced_long");COLS=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","return_4h","return_24h","top_expert","second_expert","top_score","second_score","variation","variation_rank")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(v):
 x=pd.to_numeric(v,errors="coerce").to_numpy(float);o=np.full(len(x),np.nan);h=[]
 for i,c in enumerate(x):
  q=np.asarray(h[-270:],float)
  if np.isfinite(c) and len(q)>=180:o[i]=((q<c).sum()+.5*(q==c).sum())/len(q)
  if np.isfinite(c):h.append(float(c))
 return pd.Series(o,index=v.index)
def sides(r4,r24):return np.asarray([np.sign(r4),-np.sign(r4),np.sign(r24),-np.sign(r24)],dtype=int)
def online_states(market,memory=60,minimum=30):
 f=market.copy().sort_values("date").drop_duplicates("date",keep="last");f.date=pd.to_datetime(f.date,utc=True);f=f.set_index("date")
 for c in ("open","close"):f[c]=pd.to_numeric(f[c],errors="coerce")
 first=f.index.min().ceil("D")+pd.Timedelta(hours=2);decisions=pd.date_range(first,END,freq="24h",inclusive="left");hist=[[] for _ in EXPERTS];pending=[];rows=[]
 for d in decisions:
  matured=[item for item in pending if item["exit"]<=d];pending=[item for item in pending if item["exit"]>d]
  for item in matured:
   if item["entry"] not in f.index or item["exit"] not in f.index:continue
   opens=f.loc[[item["entry"],item["exit"]],"open"].to_numpy(float)
   if not np.isfinite(opens).all() or not (opens>0).all():continue
   forward=float(np.log(opens[1]/opens[0]))
   for j,h in enumerate(hist):h.append(float(item["sides"][j]*forward))
  times=pd.date_range(d-pd.Timedelta(hours=24,minutes=5),d-pd.Timedelta(minutes=5),freq="5min")
  if len(times)!=289 or not times.isin(f.index).all():continue
  block=f.loc[times,["open","close"]].to_numpy(float)
  if not np.isfinite(block).all() or not (block>0).all():continue
  closes=block[:,1];r4=float(np.log(closes[-1]/closes[-49]));r24=float(np.log(closes[-1]/closes[0]));intrabar=np.log(block[-96:,1]/block[-96:,0]);var=float(np.sqrt(np.square(intrabar).sum()));current=sides(r4,r24)
  scores=np.asarray([np.mean(h[-memory:]) if len(h)>=minimum else np.nan for h in hist]);order=sorted(range(len(EXPERTS)),key=lambda j:(-scores[j],j)) if np.isfinite(scores).all() else []
  first_i=order[0] if order else -1;second_i=order[1] if order else -1;common=int(current[first_i]) if order and current[first_i]==current[second_i] and current[first_i]!=0 else 0;valid=bool(r4!=0 and r24!=0 and var>0)
  rows.append({"decision_time":d,"source_valid":valid,"return_4h":r4,"return_24h":r24,"variation":var,**{f"{n}_side":int(current[j]) for j,n in enumerate(EXPERTS)},**{f"{n}_score":float(scores[j]) for j,n in enumerate(EXPERTS)},"top_expert":EXPERTS[first_i] if order else "","second_expert":EXPERTS[second_i] if order else "","top_score":float(scores[first_i]) if order else np.nan,"second_score":float(scores[second_i]) if order else np.nan,"top_side":int(current[first_i]) if order else 0,"common_side":common,"mature_labels_before_decision":min((len(h) for h in hist),default=0)})
  pending.append({"entry":d+pd.Timedelta(minutes=5),"exit":d+pd.Timedelta(hours=24,minutes=5),"sides":current.copy()})
 x=pd.DataFrame(rows);x["variation_rank"]=rank(x.variation.where(x.source_valid));x["ranking_valid"]=x.source_valid&x.top_expert.ne("")&x.top_side.ne(0)&np.isfinite(x[["top_score","second_score","variation","variation_rank"]]).all(axis=1);x["concordance"]=x.ranking_valid&x.common_side.ne(0);return x
def conditions(x,control):
 if control not in ("primary",*CONTROLS):raise ValueError(control)
 active=x.ranking_valid.copy() if control=="no_top_two_concordance" else x.concordance.copy();side=(x.top_side if control=="no_top_two_concordance" else x.common_side).astype(int).copy()
 if control=="one_decision_stale_committee":side=x.common_side.shift(1).fillna(0).astype(int);active=x.concordance&side.ne(0)
 elif control=="direction_flip":side=-side
 elif control=="same_clock_forced_long":side=pd.Series(1,index=x.index)
 if control!="no_variation_gate":active&=x.variation_rank.ge(.65)
 return active,side
def clock(x,control="primary"):
 active,side=conditions(x,control);rows=[];next_allowed=None
 for i in x.index[active]:
  d=pd.Timestamp(x.at[i,"decision_time"]);entry=d+pd.Timedelta(minutes=5);exit_=entry+pd.Timedelta(hours=24)
  if next_allowed is not None and entry<next_allowed:continue
  split=next((n for n,(a,b) in SPLITS.items() if entry>=a and exit_<=b),None)
  if split is None:continue
  next_allowed=exit_;rows.append({"candidate":prereg.POLICY_ID,"control":control,"split":split,"decision_time":d,"feature_available_time":d,"entry_time":entry,"exit_time":exit_,"side":int(side.at[i]),"return_4h":float(x.at[i,"return_4h"]),"return_24h":float(x.at[i,"return_24h"]),"top_expert":x.at[i,"top_expert"],"second_expert":x.at[i,"second_expert"],"top_score":float(x.at[i,"top_score"]),"second_score":float(x.at[i,"second_score"]),"variation":float(x.at[i,"variation"]),"variation_rank":float(x.at[i,"variation_rank"])})
 return pd.DataFrame(rows,columns=COLS)
def stats(c,n):
 x=c[c.split.eq(n)]
 if x.empty:return {"events":0,"longs":0,"shorts":0,"minority_side_share":0.,"max_month_share":0.}
 lo=int(x.side.eq(1).sum());sh=int(x.side.eq(-1).sum());m=x.entry_time.dt.strftime("%Y-%m").value_counts();return {"events":len(x),"longs":lo,"shorts":sh,"minority_side_share":min(lo,sh)/len(x),"max_month_share":int(m.max())/len(x)}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA or sha(prereg.MARKET)!=prereg.MARKET_SHA or sha(LOADER)!=LOADER_SHA:raise RuntimeError("HVTCEC binding drift")
 market,source=load_market();x=online_states(market);primary=clock(x);controls={n:clock(x,n) for n in CONTROLS};SOURCE_DIR.mkdir(parents=True,exist_ok=True);CONTROL_DIR.mkdir(parents=True,exist_ok=True);_write_gzip_csv(x,STATES);_write_gzip_csv(primary,CLOCK)
 for n,c in controls.items():_write_gzip_csv(c,CONTROL_DIR/f"{n}.csv.gz")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());source_core={"protocol_version":"hvtcec_24_causal_source_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"market":source,"loader":{"path":str(LOADER),"sha256":LOADER_SHA},"states":{"path":str(STATES),"sha256":sha(STATES),"rows":len(x)},"causally_mature_counterfactual_labels_opened":True,"same_decision_label_used":False,"unmatured_label_used":False,"funding_pnl_cagr_mdd_opened":False,"gross9_rows_opened":False};sm={**source_core,"manifest_hash":prereg.canonical_hash(source_core)};MANIFEST.write_text(json.dumps(sm,indent=2,ensure_ascii=False,allow_nan=False)+"\n");support={n:stats(primary,n) for n in SPLITS};checks={k:v for n,s in support.items() for k,v in ((f"{n}_minimum_events",s["events"]>=MINIMUM[n]),(f"{n}_side_balance",s["minority_side_share"]>=.2),(f"{n}_month_concentration",s["max_month_share"]<=.45))};passed=all(checks.values());core={"protocol_version":"hvtcec_24_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":source_core["preregistration"],"source_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"manifest_hash":sm["manifest_hash"]},"causal_label_audit":{"mature_counterfactual_labels_opened":True,"same_decision_label_used":False,"unmatured_label_used":False},"current_decision_future_used":False,"funding_pnl_cagr_mdd_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(primary)},"controls":{n:{"path":str(CONTROL_DIR/f"{n}.csv.gz"),"sha256":sha(CONTROL_DIR/f"{n}.csv.gz"),"rows":len(c),"promotion_authorized":False} for n,c in controls.items()},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};r={**core,"manifest_hash":prereg.canonical_hash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":argparse.ArgumentParser().parse_args();r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
