"""Audit immutable rank-history drift in two terminal eight-hour candidates."""
from __future__ import annotations
import hashlib,inspect,json
from pathlib import Path
from typing import Any
from training import preregister_high_volatility_settlement_absorption_relay as hvsar_reg
from training import build_high_volatility_settlement_absorption_relay_support as hvsar_build
from training import preregister_high_volatility_aggressive_flow_confirmation_relay as hvafc_reg
from training import build_high_volatility_aggressive_flow_confirmation_relay_support as hvafc_build
OUTPUT=Path("results/eight_hour_rank_contract_drift_audit_2026-08-09.json")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_hash(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def defaults(fn)->dict[str,int]:
 p=inspect.signature(fn).parameters;return {"history_observations":p["lookback"].default,"minimum_history_observations":p["minimum"].default}
def record(policy:str,reg_module,build_module,downstream:list[Path])->dict[str,Any]:
 reg=reg_module.build()["policy"];registered={k:reg[k] for k in ("history_observations","minimum_history_observations")};implemented=defaults(build_module.strict_prior_midrank)
 return {"policy_id":policy,"preregistration":{"path":str(reg_module.DEFAULT_OUTPUT),"sha256":sha(reg_module.DEFAULT_OUTPUT),"registered":registered},"support_builder":{"path":str(Path(build_module.__file__).relative_to(Path.cwd())),"sha256":sha(Path(build_module.__file__)),"implemented_defaults":implemented},"rank_contract_drift":registered!=implemented,"downstream_artifacts":[{"path":str(p),"sha256":sha(p)} for p in downstream if p.exists()],"exact_preregistration_reproduction_evidence_valid":False,"terminal_artifacts_mutated":False,"retry_or_repair_authorized":False}
def build()->dict[str,Any]:
 core={"protocol_version":"eight_hour_rank_contract_drift_audit_v1","as_of_date":"2026-08-09","outcomes_reopened":False,"candidates":{"HVSAR-6":record("HVSAR-6",hvsar_reg,hvsar_build,[hvsar_build.RESULT]),"HVAFC-6":record("HVAFC-6",hvafc_reg,hvafc_build,[hvafc_build.RESULT,Path("results/high_volatility_aggressive_flow_confirmation_relay_gross9_novelty_2026-08-09.json"),Path("results/high_volatility_aggressive_flow_confirmation_relay_train_economics_2026-08-09.json")])},"conclusion":"Both immutable terminal runs used 180/120 instead of registered 270/180. They remain terminal and cannot be retried or repaired, but their downstream outputs are not exact-preregistration reproduction evidence.","followup":"All later eight-hour builders must bind and test 270/180 before source incidence is opened."}
 return {**core,"manifest_hash":canonical_hash(core)}
if __name__=="__main__":OUTPUT.write_text(json.dumps(build(),indent=2)+"\n");print(OUTPUT)
