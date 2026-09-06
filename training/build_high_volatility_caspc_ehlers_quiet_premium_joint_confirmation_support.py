"""Source-only exact-clock intersection for frozen HVCELVPQA-8."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
import pandas as pd
from training import preregister_high_volatility_caspc_ehlers_quiet_premium_joint_confirmation as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PREREG_SHA="5aa77cc9fafca7aa7aba098641af010d0d697b8565ce48e2f7c5596cf52a40f1"
LEFT=Path("data/high_volatility_caspc_ehlers_active_veto_clocks_2023_2026.csv.gz")
RIGHT=Path("data/high_volatility_caspc_quiet_premium_activity_clocks_2023_2026.csv.gz")
CLOCK=Path("data/high_volatility_caspc_ehlers_quiet_premium_joint_confirmation_clocks_2023_2026.csv.gz")
RESULT=Path("results/high_volatility_caspc_ehlers_quiet_premium_joint_confirmation_support_2026-08-16.json")
MINIMUM_EVENTS={"train":8,"test":12,"eval":12,"final":8}
KEYS=["split","decision_time","feature_available_time","entry_time","exit_time","side"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_hash(v:Any):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def load(path:Path,label:str)->pd.DataFrame:
 d=pd.read_csv(path)
 if "control" in d:d=d[d.control.eq("primary")].copy()
 for c in ("decision_time","feature_available_time","entry_time","exit_time"):d[c]=pd.to_datetime(d[c],utc=True)
 if d[KEYS[:-1]].duplicated().any():raise RuntimeError(f"duplicate {label} clock key")
 if not d.side.isin([-1,1]).all():raise RuntimeError(f"invalid {label} side")
 return d
def intersect(left:pd.DataFrame,right:pd.DataFrame)->pd.DataFrame:
 joined=left[KEYS].merge(right[KEYS],on=KEYS,how="inner",validate="one_to_one")
 joined=joined.sort_values("entry_time").reset_index(drop=True)
 if (joined.entry_time<joined.exit_time.shift(fill_value=pd.Timestamp.min.tz_localize('UTC'))).any():
  raise RuntimeError("overlapping joint clock")
 out=pd.DataFrame({"candidate":prereg.POLICY_ID,"control":"primary",**{c:joined[c] for c in KEYS}})
 return out[["candidate","control",*KEYS]]
def stats(clock:pd.DataFrame,split:str):
 d=clock[clock.split.eq(split)];n=len(d);longs=int(d.side.eq(1).sum());shorts=int(d.side.eq(-1).sum())
 return {"events":n,"longs":longs,"shorts":shorts,"minority_side_share":min(longs,shorts)/n if n else 0.0,"max_month_share":float(d.entry_time.dt.strftime('%Y-%m').value_counts().max()/n) if n else 0.0}
def run():
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA:raise RuntimeError("preregistration drift")
 registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(registration)
 left,right=load(LEFT,"HVCELV"),load(RIGHT,"HVCASPCPQA")
 clock=intersect(left,right);CLOCK.parent.mkdir(parents=True,exist_ok=True);_write_gzip_csv(clock,CLOCK)
 support={s:stats(clock,s) for s in MINIMUM_EVENTS};checks={}
 for s,v in support.items():
  checks[f"{s}_minimum_events"]=v["events"]>=MINIMUM_EVENTS[s]
  checks[f"{s}_side_balance"]=v["minority_side_share"]>=.20
  checks[f"{s}_month_concentration"]=v["max_month_share"]<=.45
 passed=all(checks.values())
 core={"protocol_version":"hvcelvpqa_8_source_support_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"component_clock_rows_opened":{"HVCELV-8":len(left),"HVCASPCPQA-8":len(right)},"component_clock_hashes":{"HVCELV-8":sha(LEFT),"HVCASPCPQA-8":sha(RIGHT)},"completed_preentry_sources_opened":True,"postentry_return_pnl_execution_price_opened":False,"held_interval_funding_values_opened":False,"gross9_rows_opened":False,"clock":{"path":str(CLOCK),"sha256":sha(CLOCK),"rows":len(clock)},"controls":{},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"}
 result={**core,"manifest_hash":canonical_hash(core)};RESULT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result
if __name__=="__main__":
 r=run();print(json.dumps({"passed":r["support_passed"],"support":r["support"]},indent=2))
