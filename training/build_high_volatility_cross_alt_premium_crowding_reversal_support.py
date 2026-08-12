"""Metadata-only source support for frozen HVCAPCR-6."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_cross_alt_premium_crowding_reversal as prereg
ENV_FILE="/home/pakchu/rllm/.env";PREREG_SHA="0f0dfdf57ca23c9f93f0f02d6b40c5ea1aa26a7bc223575c1f784e12eca504ae";OUTPUT=Path("results/high_volatility_cross_alt_premium_crowding_reversal_support_2026-08-12.json")
QUERY="SELECT symbol,interval,min(ts) AS first_ts,max(ts) AS last_ts,count(*) AS rows FROM bars_binance_premium WHERE ts>=:start AND ts<:end GROUP BY symbol,interval ORDER BY symbol,interval"
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def evaluate(rows:list[dict[str,Any]])->dict[str,Any]:
 required=set(prereg.build()["features"]["alts"]);covered={str(x["symbol"]) for x in rows if x["interval"]=="1m" and int(x["rows"])>0};missing=sorted(required-covered);return {"required_symbols":sorted(required),"covered_symbols":sorted(covered),"missing_symbols":missing,"source_support_passed":not missing}
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("HVCAPCR prereg drift")
 from sqlalchemy import create_engine,text
 from preprocessing.live_db_features import load_env_file,postgres_url_from_env
 load_env_file(ENV_FILE);db=create_engine(postgres_url_from_env(ENV_FILE),connect_args={"connect_timeout":10})
 try:
  with db.connect() as c: raw=[dict(x._mapping) for x in c.execute(text(QUERY),{"start":"2023-01-01T00:00:00Z","end":"2026-08-01T00:00:00Z"})]
 finally:db.dispose()
 rows=[{"symbol":x["symbol"],"interval":x["interval"],"first_ts":x["first_ts"].isoformat(),"last_ts":x["last_ts"].isoformat(),"rows":int(x["rows"])} for x in raw];coverage=evaluate(rows);registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());core={"protocol_version":"hvcapcr_6_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"query_sha256":hashlib.sha256(QUERY.encode()).hexdigest(),"metadata_rows":rows,"coverage":coverage,"premium_values_opened":0,"candidate_incidence_opened":False,"postentry_return_pnl_execution_price_opened":False,"gross9_rows_opened":False,"decision":"pass_to_incidence" if coverage["source_support_passed"] else "terminal_source_axis_absent","advance_to_gross9_novelty":False};result={**core,"manifest_hash":chash(core)};OUTPUT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return result
if __name__=="__main__":print(json.dumps({"passed":run()["coverage"]["source_support_passed"],"result":str(OUTPUT)}))
