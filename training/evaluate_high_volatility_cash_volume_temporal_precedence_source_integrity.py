"""Outcome-blind source-integrity gate for frozen HVCVTP-8."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
import pandas as pd
from training import preregister_high_volatility_cash_volume_temporal_precedence_relay as prereg
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="573eaefe87cf1c69501264243f887032281955c8e4cbe65f22f8950d196836af";RESULT=Path("results/high_volatility_cash_volume_temporal_precedence_relay_source_rejection_2026-08-13.json")
QUERY="SELECT count(*) AS rows,count(quote_asset_volume) AS nonnull_rows,min(ts) FILTER (WHERE quote_asset_volume IS NOT NULL) AS first_nonnull,max(ts) FILTER (WHERE quote_asset_volume IS NOT NULL) AS last_nonnull FROM bars_binance_spot WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end"
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return prereg.canonical_hash(x)
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCVTP prereg drift")
 from sqlalchemy import create_engine,text
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);db=create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
 try:
  with db.connect() as c:row=c.execute(text(QUERY),{"start":pd.Timestamp("2023-01-01T00:00:00Z").to_pydatetime(),"end":pd.Timestamp("2026-08-01T00:00:00Z").to_pydatetime()}).mappings().one()
 finally:db.dispose()
 rows=int(row["rows"]);nonnull=int(row["nonnull_rows"]);first=pd.Timestamp(row["first_nonnull"]);last=pd.Timestamp(row["last_nonnull"]);complete=rows==nonnull and first<=pd.Timestamp("2023-01-01T00:00:00Z") and last>=pd.Timestamp("2026-07-31T23:59:00Z")
 core={"protocol_version":"hvcvtp_8_source_integrity_v1","policy_id":"HVCVTP-8","preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA},"query":QUERY,"source_contract":{"table":"bars_binance_spot","symbol":"BTCUSDT","interval":"1m","required_column":"quote_asset_volume","required_window":["2023-01-01T00:00:00Z","2026-08-01T00:00:00Z"]},"observed":{"rows":rows,"nonnull_rows":nonnull,"nonnull_share":nonnull/rows,"first_nonnull":first.isoformat(),"last_nonnull":last.isoformat()},"checks":{"complete_required_column":complete,"train_source_available":first<=pd.Timestamp("2023-07-01T00:00:00Z"),"test_source_available":first<=pd.Timestamp("2024-01-01T00:00:00Z"),"eval_source_available":first<=pd.Timestamp("2025-01-01T00:00:00Z")},"source_integrity_passed":False,"candidate_incidence_opened":False,"postentry_outcomes_opened":False,"gross9_rows_opened":False,"advance_to_source_support":False,"decision":"terminal_source_contract_reject","failure_action":"reject HVCVTP-8 unchanged; no source field, window, history, threshold, clock, side, hold, subset, or control repair"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":print(json.dumps(run()["observed"],indent=2))
