"""Prehistory-seeded causal source support for HVCQCA-24."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_credit_transition_causal_adapter as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="a949d2c3f447ad676ff33d28829e32de159b0226a8898b5f6de157025cd11252";LABELS=Path("data/high_volatility_credit_transition_causal_adapter_labels_2022_2026.csv.gz");CLOCK=Path("data/high_volatility_credit_transition_causal_adapter_clocks_2023_2026.csv.gz");RESULT=Path("results/high_volatility_credit_transition_causal_adapter_support_2026-08-18.json");POLICY=prereg.build()["policy"];STAGES={k:tuple(pd.Timestamp(v) for v in values) for k,values in prereg.build()["stages"].items()};GATES=prereg.build()["source_support_gates"]
QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,(array_agg(open ORDER BY ts))[1] AS open,count(*) AS source_rows FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""
def db_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_opens()->pd.DataFrame:
 from sqlalchemy import text
 db=db_engine()
 try:
  with db.connect() as c:x=pd.read_sql_query(text(QUERY),c,params={"start":pd.Timestamp("2022-09-01T00:00Z").to_pydatetime(),"end":pd.Timestamp("2026-08-02T00:10Z").to_pydatetime()})
 finally:db.dispose()
 x["date"]=pd.to_datetime(x["date"],utc=True,errors="raise");x["open"]=pd.to_numeric(x["open"],errors="raise");x["source_rows"]=pd.to_numeric(x["source_rows"],errors="raise")
 if x["date"].duplicated().any() or not x["source_rows"].eq(5).all() or not np.isfinite(x["open"]).all() or not x["open"].gt(0).all():raise RuntimeError("HVCQCA open source drift")
 return x
def transitions(states:pd.DataFrame)->pd.DataFrame:
 x=states.copy();x["decision_time"]=pd.to_datetime(x["decision_time"],utc=True,errors="raise");x["relative_credit_return"]=pd.to_numeric(x["relative_credit_return"],errors="coerce");previous=x["relative_credit_return"].shift(1);previous_valid=x["source_valid"].shift(1,fill_value=False);active=x["source_valid"].eq(True)&previous_valid&np.isfinite(x["relative_credit_return"])&np.isfinite(previous)&x["relative_credit_return"].ne(0)&previous.ne(0)&(np.sign(x["relative_credit_return"])==-np.sign(previous));out=x[active].copy();out["transition_side"]=np.sign(out["relative_credit_return"]).astype(int);out["entry_time"]=out["decision_time"]+pd.Timedelta("5m");out["exit_time"]=out["entry_time"]+pd.Timedelta("24h");return out
def label(events:pd.DataFrame,opens:pd.DataFrame)->pd.DataFrame:
 x=events.copy();lookup=opens.set_index("date")["open"];x["entry_open"]=x["entry_time"].map(lookup);x["exit_open"]=x["exit_time"].map(lookup);x["label_valid"]=np.isfinite(x[["entry_open","exit_open"]]).all(axis=1)&x["entry_open"].gt(0)&x["exit_open"].gt(0);x["gross_directional_label"]=np.nan;x.loc[x["label_valid"],"gross_directional_label"]=x.loc[x["label_valid"],"transition_side"]*np.log(x.loc[x["label_valid"],"exit_open"]/x.loc[x["label_valid"],"entry_open"]);return x.sort_values("decision_time",kind="stable").reset_index(drop=True)
def adapt(labels:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for cur in labels.itertuples(index=False):
  if not np.isfinite(cur.btc_variation_rank) or cur.btc_variation_rank<POLICY["variation_rank_min"]:continue
  entry=cur.entry_time;exit_=cur.exit_time;split=next((k for k,(a,b) in STAGES.items() if a<=entry and exit_<=b),None)
  if split is None:continue
  prior=labels[(labels["exit_time"]<=cur.decision_time)&labels["label_valid"]].tail(POLICY["memory_labels"])
  if len(prior)<POLICY["minimum_mature_labels"]:continue
  score=float(prior["gross_directional_label"].mean())
  if not np.isfinite(score) or score==0:continue
  rows.append({"candidate":prereg.POLICY_ID,"control":"primary","split":split,"session_date":cur.session_date,"decision_time":cur.decision_time,"feature_available_time":cur.decision_time,"entry_time":entry,"exit_time":exit_,"side":int(cur.transition_side*np.sign(score)),"transition_side":int(cur.transition_side),"memory_count":len(prior),"memory_score":score,"memory_latest_exit":prior["exit_time"].max(),"btc_variation_rank":float(cur.btc_variation_rank)})
 out=pd.DataFrame(rows,columns=("candidate","control","split","session_date","decision_time","feature_available_time","entry_time","exit_time","side","transition_side","memory_count","memory_score","memory_latest_exit","btc_variation_rank"))
 if not out.empty and not out["memory_latest_exit"].le(out["decision_time"]).all():raise RuntimeError("HVCQCA immature leakage")
 return out
def stats(clock:pd.DataFrame,k:str)->dict[str,Any]:
 x=clock[clock["split"].eq(k)];n=len(x);l=int(x["side"].eq(1).sum());s=int(x["side"].eq(-1).sum());m=float(x["entry_time"].dt.strftime("%Y-%m").value_counts().max()/n) if n else 0.;return {"events":n,"longs":l,"shorts":s,"minority_side_share":min(l,s)/n if n else 0.,"max_month_share":m}
def run(labels_path:Path=LABELS,clock_path:Path=CLOCK,result_path:Path=RESULT)->dict[str,Any]:
 if prereg.sha256(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCQCA prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);source=pd.read_csv(prereg.SOURCE);events=transitions(source);labels=label(events,load_opens());_write_gzip_csv(labels,labels_path);clock=adapt(labels);_write_gzip_csv(clock,clock_path);support={k:stats(clock,k) for k in STAGES};checks={}
 for k,v in support.items():checks[f"{k}_minimum_events"]=v["events"]>=GATES["minimum_events"][k];checks[f"{k}_side_balance"]=v["minority_side_share"]>=GATES["minority_side_share_min"];checks[f"{k}_month_concentration"]=v["max_month_share"]<=GATES["max_month_share"]
 passed=all(checks.values());core={"protocol_version":"hvcqca_24_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"mature_counterfactual_labels_opened":True,"current_candidate_postentry_return_or_pnl_opened":False,"funding_opened":False,"gross9_rows_opened":False,"labels":{"path":str(labels_path),"sha256":prereg.sha256(labels_path),"rows":len(labels),"valid_rows":int(labels["label_valid"].sum())},"clock":{"path":str(clock_path),"sha256":prereg.sha256(clock_path),"rows":len(clock)},"causality_checks":{"all_memory_latest_exit_le_decision":bool(clock.empty or clock["memory_latest_exit"].le(clock["decision_time"]).all()),"immature_labels_used":False,"stage_resets":False},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":prereg.canonical_hash(core)};result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"],"causality":r["causality_checks"]},indent=2))
