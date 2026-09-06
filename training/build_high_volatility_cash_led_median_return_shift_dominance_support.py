"""Deterministic source support for HVCVMSD-8."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from training import preregister_high_volatility_cash_led_median_return_shift_dominance as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="b7d744933de8029d65c22d4f05f883328a13fbecf6e6d040186ad04e554af97b"
PRIMARY=Path(prereg.PRIMARY["clock"]["path"]);SPOT=Path("data/high_volatility_cash_led_median_return_shift_dominance_spot_shifts_2023_2026.csv.gz");CLOCK=Path("data/high_volatility_cash_led_median_return_shift_dominance_clocks_2023_2026.csv.gz");RESULT=Path("results/high_volatility_cash_led_median_return_shift_dominance_support_2026-08-18.json")
STAGES={k:tuple(pd.Timestamp(v) for v in values) for k,values in prereg.build()["stages"].items()};GATES=prereg.build()["source_support_gates"]
QUERY="""WITH r AS (SELECT ts,open,high,low,close,date_bin('8 hours',ts-INTERVAL '1 hour',TIMESTAMPTZ '1970-01-01 00:00:00+00')+INTERVAL '1 hour' AS block_start FROM bars_binance_spot WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end) SELECT block_start+INTERVAL '8 hours' AS decision_time,percentile_cont(0.5) WITHIN GROUP (ORDER BY ln(close/open)) FILTER (WHERE ts<block_start+INTERVAL '4 hours') AS first_median,percentile_cont(0.5) WITHIN GROUP (ORDER BY ln(close/open)) FILTER (WHERE ts>=block_start+INTERVAL '4 hours') AS second_median,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>0 AND low>0 AND close>0 AND high>=greatest(open,close,low) AND low<=least(open,close,high)) AS coherent FROM r GROUP BY block_start ORDER BY block_start"""
def postgres_engine():
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);return create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
def load_spot()->pd.DataFrame:
 from sqlalchemy import text
 db=postgres_engine()
 try:
  with db.connect() as c:frame=pd.read_sql_query(text(QUERY),c,params={"start":pd.Timestamp("2023-04-01T00:00Z").to_pydatetime(),"end":pd.Timestamp("2026-08-01T00:00Z").to_pydatetime()})
 finally:db.dispose()
 return frame
def prepare_spot(frame:pd.DataFrame)->pd.DataFrame:
 expected=["decision_time","first_median","second_median","source_rows","distinct_rows","first_ts","last_ts","coherent"]
 if frame.columns.tolist()!=expected:raise RuntimeError("HVCVMSD-8 spot schema drift")
 x=frame.copy()
 for c in ("decision_time","first_ts","last_ts"):x[c]=pd.to_datetime(x[c],utc=True,errors="raise")
 for c in ("first_median","second_median","source_rows","distinct_rows"):x[c]=pd.to_numeric(x[c],errors="raise")
 start=x["decision_time"]-pd.Timedelta("8h");x["source_valid"]=(np.isfinite(x[["first_median","second_median"]]).all(axis=1)&x["source_rows"].eq(480)&x["distinct_rows"].eq(480)&x["first_ts"].eq(start)&x["last_ts"].eq(x["decision_time"]-pd.Timedelta("1m"))&x["coherent"].eq(True));x["spot_shift"]=x["second_median"]-x["first_median"];return x
def confirm(primary:pd.DataFrame,spot:pd.DataFrame)->pd.DataFrame:
 p=primary.copy()
 for c in ("decision_time","feature_available_time","entry_time","exit_time"):p[c]=pd.to_datetime(p[c],utc=True,errors="raise")
 merged=p.merge(spot[["decision_time","source_valid","spot_shift"]],on="decision_time",how="left",validate="one_to_one")
 keep=(merged["source_valid"].eq(True)&np.isfinite(merged["spot_shift"])&merged["spot_shift"].ne(0)&(np.sign(merged["spot_shift"])==merged["side"])&merged["spot_shift"].abs().gt(merged["median_shift"].abs()))
 out=merged.loc[keep].copy();out["candidate"]=prereg.POLICY_ID;out["perpetual_shift"]=out["median_shift"];return out[["candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","perpetual_shift","spot_shift","shift_rank","variation_rank"]].sort_values("decision_time",kind="stable").reset_index(drop=True)
def stats(clock:pd.DataFrame,split:str)->dict[str,Any]:
 x=clock[clock["split"].eq(split)];n=len(x);l=int(x["side"].eq(1).sum());s=int(x["side"].eq(-1).sum());m=float(x["entry_time"].dt.strftime("%Y-%m").value_counts().max()/n) if n else 0.;return {"events":n,"longs":l,"shorts":s,"minority_side_share":min(l,s)/n if n else 0.,"max_month_share":m}
def run(spot_path:Path=SPOT,clock_path:Path=CLOCK,result_path:Path=RESULT)->dict[str,Any]:
 if prereg.sha256(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCVMSD-8 prereg drift")
 reg=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(reg);raw=load_spot();spot=prepare_spot(raw);_write_gzip_csv(spot,spot_path);primary=pd.read_csv(PRIMARY);clock=confirm(primary,spot);_write_gzip_csv(clock,clock_path)
 support={k:stats(clock,k) for k in STAGES};checks={}
 for k,v in support.items():checks[f"{k}_minimum_events"]=v["events"]>=GATES["minimum_events"][k];checks[f"{k}_side_balance"]=v["minority_side_share"]>=GATES["minority_side_share_min"];checks[f"{k}_month_concentration"]=v["max_month_share"]<=GATES["max_month_share"]
 passed=all(checks.values());core={"protocol_version":"hvcvmsd_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":reg["manifest_hash"]},"query":{"sha256":__import__('hashlib').sha256(QUERY.encode()).hexdigest(),"table":"bars_binance_spot","rows":len(raw)},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_opened":False,"gross9_rows_opened":False,"spot_shifts":{"path":str(spot_path),"sha256":prereg.sha256(spot_path),"rows":len(spot),"valid_rows":int(spot["source_valid"].sum())},"clock":{"path":str(clock_path),"sha256":prereg.sha256(clock_path),"rows":len(clock)},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"};result={**core,"manifest_hash":prereg.canonical_hash(core)};result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
