"""Evaluate pre-2024 OI support for PPOSM residual actions."""
from __future__ import annotations
import argparse, hashlib, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np, pandas as pd
from training import preregister_pposm_pre2024_oi_action_source as prereg
from training import build_pposm_pre2024_intrahour_premium_action_source as base
from training import build_pposm_state_router_data as numeric
DEFAULT_OUTPUT=Path("results/pposm_pre2024_oi_action_source_2026-09-05.json"); DEFAULT_ENV_FILE=Path("/home/pakchu/rllm/.env"); CANDIDATES=("SKIP","TP12"); EPS=1e-12
@dataclass(frozen=True)
class Config:
 preregistration:Path=prereg.DEFAULT_OUTPUT; output:Path=DEFAULT_OUTPUT; env_file:Path=DEFAULT_ENV_FILE; manifest:Path=numeric.DEFAULT_MANIFEST
def canonical_json(v:Any)->str:return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def sha256_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def sha256_path(p:Path)->str:
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def load_preregistration(path:Path)->dict[str,Any]:
 d=json.loads(path.read_text()); obs=sha256_bytes(canonical_json({k:v for k,v in d.items() if k!="preregistration_hash"}).encode())
 if d.get("policy_id")!=prereg.POLICY_ID or d.get("preregistration_hash")!=obs: raise RuntimeError("OI preregistration mismatch")
 return d
def immutable_write_json(path:Path,obj:dict[str,Any])->None:
 payload=json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n"; path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and json.loads(path.read_text()).get("result_hash")!=obj.get("result_hash"): raise RuntimeError(f"refusing to overwrite different result: {path}")
 path.write_text(payload)
def db_engine(env_file:Path):
 from sqlalchemy import create_engine
 from preprocessing.live_db_features import load_env_file, postgres_url_from_env
 load_env_file(env_file); return create_engine(postgres_url_from_env(env_file),connect_args={"connect_timeout":10})
def query_oi(env_file:Path)->pd.DataFrame:
 from sqlalchemy import text
 e=db_engine(env_file)
 try:
  with e.connect() as c: f=pd.read_sql_query(text(prereg.OI_QUERY),c,params={"symbol":"BTCUSDT","start":pd.Timestamp(prereg.QUERY_START),"end":pd.Timestamp(prereg.QUERY_END_EXCLUSIVE)})
 finally:e.dispose()
 if f.columns.tolist()!=["date","sum_open_interest"]: raise RuntimeError(f"OI schema drift: {f.columns.tolist()}")
 f["date"]=pd.to_datetime(f["date"],utc=True,errors="raise"); f["sum_open_interest"]=pd.to_numeric(f["sum_open_interest"],errors="coerce")
 return f.sort_values("date").reset_index(drop=True)
def _finite(v:Iterable[float])->bool:return all(math.isfinite(float(x)) for x in v)
def feature_row(decision_time:pd.Timestamp, oi:pd.DataFrame)->dict[str,Any]:
 row={"decision_time":decision_time,"source_valid":True,"invalid_reason":"pass"}
 for periods in prereg.FEATURE_WINDOWS_PERIODS:
  minutes=periods*5; start=decision_time-pd.Timedelta(minutes=minutes); end=decision_time-pd.Timedelta(minutes=5); block=oi[(oi.date>=start)&(oi.date<decision_time)]; prefix=f"oi{minutes}m"
  if len(block)!=periods or block.date.nunique()!=periods or block.date.min()!=start or block.date.max()!=end:
   row["source_valid"]=False; row["invalid_reason"]=f"{prefix}_grid"; continue
  vals=block.sum_open_interest.to_numpy(dtype=float)
  if (not np.isfinite(vals).all()) or np.any(vals<=0): row["source_valid"]=False; row["invalid_reason"]=f"{prefix}_finite_positive"; continue
  logs=np.log(vals); diffs=np.diff(logs); std=float(np.std(logs,ddof=0)); half=periods//2
  feats={f"{prefix}_level_mean":float(np.mean(logs)),f"{prefix}_level_std":std,f"{prefix}_level_min":float(np.min(logs)),f"{prefix}_level_max":float(np.max(logs)),f"{prefix}_level_range":float(np.max(logs)-np.min(logs)),f"{prefix}_last_over_first_log":float(logs[-1]-logs[0]),f"{prefix}_second_half_minus_first_half_log":float(np.mean(logs[half:])-np.mean(logs[:half])),f"{prefix}_abs_logdiff_sum":float(np.sum(np.abs(diffs))) if len(diffs) else 0.0,f"{prefix}_positive_logdiff_share":float(np.mean(diffs>0.0)) if len(diffs) else 0.0,f"{prefix}_negative_logdiff_share":float(np.mean(diffs<0.0)) if len(diffs) else 0.0,f"{prefix}_end_zscore":float((logs[-1]-np.mean(logs))/(std+EPS))}
  if not _finite(feats.values()): row["source_valid"]=False; row["invalid_reason"]=f"{prefix}_nonfinite_feature"
  row.update(feats)
 return row
def build_feature_frame(labels:pd.DataFrame, oi:pd.DataFrame)->pd.DataFrame:
 feats=pd.DataFrame([feature_row(pd.Timestamp(t).tz_convert('UTC'),oi) for t in labels.decision_time]); m=labels.merge(feats,on="decision_time",validate="one_to_one"); cols=[c for c in m.columns if c.startswith('oi')]; m["source_valid"]=m.source_valid.astype(bool)&np.isfinite(m[cols].to_numpy(dtype=float)).all(axis=1); return m
def source_hash_frame(frame:pd.DataFrame, cols:list[str])->str:return sha256_bytes(frame[cols].to_csv(index=False,lineterminator="\n").encode())
def build(cfg:Config)->dict[str,Any]:
 reg=load_preregistration(cfg.preregistration); labels=base.pre2024_label_frame(cfg.manifest); oi=query_oi(cfg.env_file); frame=build_feature_frame(labels,oi); feature_cols=[c for c in frame.columns if c.startswith('oi')]
 evals={c:base.evaluate_candidate(frame,c,feature_cols) for c in CANDIDATES}; checks={f"{c}_{k}":bool(v) for c,e in evals.items() for k,v in e["checks"].items()}; valid=frame.source_valid.astype(bool)
 core={"policy_id":prereg.POLICY_ID,"preregistration":{"path":str(cfg.preregistration),"sha256":sha256_path(cfg.preregistration),"preregistration_hash":reg["preregistration_hash"]},"query_hashes":{"oi":sha256_bytes(prereg.OI_QUERY.encode())},"source_rows":{"oi_5m":int(len(oi)),"labels_pre2024_signals":int(len(labels)),"feature_rows":int(len(frame)),"source_valid_feature_rows":int(valid.sum())},"source_window":{"first":oi.date.min().isoformat(),"last":oi.date.max().isoformat(),"query_start":prereg.QUERY_START,"query_end_exclusive":prereg.QUERY_END_EXCLUSIVE},"source_hashes":{"oi_5m":source_hash_frame(oi,["date","sum_open_interest"]),"labels_pre2024":source_hash_frame(labels,["signal_position","decision_time","SKIP","TP12","tp4_utility","skip_advantage","tp12_advantage"]),"features":source_hash_frame(frame.loc[:,["decision_time","source_valid",*feature_cols]],["decision_time","source_valid",*feature_cols])},"label_distribution":{c:{"switch":int(frame[c].sum()),"keep":int(len(frame)-frame[c].sum())} for c in CANDIDATES},"feature_columns":feature_cols,"evaluations":evals,"acceptance_checks":checks,"support_passed":all(checks.values()),"decision":"pass_to_separate_oos_preregistration" if all(checks.values()) else "terminal_pre2024_source_support_reject","evidence_boundary":{"oi_pre2024_opened":True,"pre2024_labels_opened":True,"oos_labels_opened":False,"fresh_outcomes_opened":False,"trained_adapter_scores_opened":False}}
 core["result_hash"]=sha256_bytes(canonical_json(core).encode()); immutable_write_json(cfg.output,core); return core
def parse_args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--preregistration",type=Path,default=prereg.DEFAULT_OUTPUT); p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); p.add_argument("--env-file",type=Path,default=DEFAULT_ENV_FILE); p.add_argument("--manifest",type=Path,default=numeric.DEFAULT_MANIFEST); return p.parse_args()
def main():
 r=build(Config(**vars(parse_args()))); print(json.dumps({"support_passed":r["support_passed"],"decision":r["decision"],"evaluations":{k:v["pooled"] for k,v in r["evaluations"].items()}},indent=2,sort_keys=True))
if __name__=='__main__':main()
