"""Causal mature-label source support for HVSAUDCA-8."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_spot_underwater_causal_sign_adapter as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="4e36303f972e27d3f02db670e0324d439688a782fa5dbf1111c502915139d7be";BASE=Path(prereg.BASE["clock"]["path"]);LABELS=Path("data/high_volatility_spot_underwater_causal_sign_adapter_mature_labels_2023_2026.csv.gz");CLOCK=Path("data/high_volatility_spot_underwater_causal_sign_adapter_clocks_2023_2026.csv.gz");RESULT=Path("results/high_volatility_spot_underwater_causal_sign_adapter_support_2026-08-18.json");POLICY=prereg.build()["policy"];GATES=prereg.build()["source_support_gates"]
QUERY="""SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,(array_agg(open ORDER BY ts))[1] AS open,count(*) AS source_rows FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY 1 ORDER BY 1"""
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_opens()->pd.DataFrame:
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c:x=pd.read_sql_query(text(QUERY),c,params={"start":pd.Timestamp("2023-07-01T00:00Z").to_pydatetime(),"end":pd.Timestamp("2026-08-01T00:10Z").to_pydatetime()})
 finally:db.dispose()
 x["date"]=pd.to_datetime(x["date"],utc=True,errors="raise");x["open"]=pd.to_numeric(x["open"],errors="raise");x["source_rows"]=pd.to_numeric(x["source_rows"],errors="raise");
 if x["date"].duplicated().any() or not x["source_rows"].eq(5).all() or not np.isfinite(x["open"]).all() or not x["open"].gt(0).all():raise RuntimeError("HVSAUDCA execution-open source drift")
 return x
def label_base(base:pd.DataFrame,opens:pd.DataFrame)->pd.DataFrame:
 x=base.copy()
 for c in ("decision_time","feature_available_time","entry_time","exit_time"):x[c]=pd.to_datetime(x[c],utc=True,errors="raise")
 lookup=opens.set_index("date")["open"];x["entry_open"]=x["entry_time"].map(lookup);x["exit_open"]=x["exit_time"].map(lookup);x["label_valid"]=np.isfinite(x[["entry_open","exit_open"]]).all(axis=1)&x["entry_open"].gt(0)&x["exit_open"].gt(0);x["gross_directional_label"]=np.nan;x.loc[x["label_valid"],"gross_directional_label"]=x.loc[x["label_valid"],"side"]*np.log(x.loc[x["label_valid"],"exit_open"]/x.loc[x["label_valid"],"entry_open"]);return x.sort_values("decision_time",kind="stable").reset_index(drop=True)
def adapt(labels:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for current in labels.itertuples(index=False):
  prior=labels[(labels["exit_time"]<=current.decision_time)&labels["label_valid"]].tail(POLICY["memory_labels"]);count=len(prior)
  if count<POLICY["minimum_mature_labels"]:continue
  score=float(prior["gross_directional_label"].mean())
  if not np.isfinite(score) or score==0:continue
  side=int(current.side)*int(np.sign(score));rows.append({"candidate":prereg.POLICY_ID,"control":"primary","split":current.split,"decision_time":current.decision_time,"feature_available_time":current.feature_available_time,"entry_time":current.entry_time,"exit_time":current.exit_time,"side":side,"base_side":int(current.side),"mature_label_count":count,"memory_score":score,"memory_latest_exit":prior["exit_time"].max()})
 out=pd.DataFrame(rows,columns=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","base_side","mature_label_count","memory_score","memory_latest_exit"))
 if not out.empty and not out["memory_latest_exit"].le(out["decision_time"]).all():raise RuntimeError("HVSAUDCA immature label leakage")
 return out
def stats(clock:pd.DataFrame,k:str)->dict[str,Any]:
 x=clock[clock["split"].eq(k)];n=len(x);l=int(x["side"].eq(1).sum());s=int(x["side"].eq(-1).sum());m=float(x["entry_time"].dt.strftime("%Y-%m").value_counts().max()/n) if n else 0.;return {"events":n,"longs":l,"shorts":s,"minority_side_share":min(l,s)/n if n else 0.,"max_month_share":m}
def run(labels_path:Path=LABELS,clock_path:Path=CLOCK,result_path:Path=RESULT)->dict[str,Any]:
 if prereg.sha256(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVSAUDCA prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);opens=load_opens();labels=label_base(pd.read_csv(BASE),opens);_write_gzip_csv(labels,labels_path);clock=adapt(labels);_write_gzip_csv(clock,clock_path);support={k:stats(clock,k) for k in prereg.build()["stages"]};checks={}
 for k,v in support.items():checks[f"{k}_minimum_events"]=v["events"]>=GATES["minimum_events"][k];checks[f"{k}_side_balance"]=v["minority_side_share"]>=GATES["minority_side_share_min"];checks[f"{k}_month_concentration"]=v["max_month_share"]<=GATES["max_month_share"]
 passed=all(checks.values());core={"protocol_version":"hvsaudca_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"mature_counterfactual_label_values_opened":True,"current_candidate_postentry_return_or_pnl_opened":False,"funding_opened":False,"gross9_rows_opened":False,"label_source":{"query_sha256":__import__('hashlib').sha256(QUERY.encode()).hexdigest(),"physical_5m_rows":len(opens)},"labels":{"path":str(labels_path),"sha256":prereg.sha256(labels_path),"rows":len(labels),"valid_rows":int(labels["label_valid"].sum())},"clock":{"path":str(clock_path),"sha256":prereg.sha256(clock_path),"rows":len(clock)},"causality_checks":{"all_memory_latest_exit_le_decision":bool(clock.empty or clock["memory_latest_exit"].le(clock["decision_time"]).all()),"current_or_immature_labels_used":False,"stage_resets":False},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":prereg.canonical_hash(core)};result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"],"causality":r["causality_checks"]},indent=2))
