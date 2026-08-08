from training import audit_eight_hour_rank_contract_drift as audit
def test_audit_detects_both_immutable_rank_drifts():
 r=audit.build();assert r["outcomes_reopened"] is False
 for x in r["candidates"].values():
  assert x["preregistration"]["registered"]=={"history_observations":270,"minimum_history_observations":180};assert x["support_builder"]["implemented_defaults"]=={"history_observations":180,"minimum_history_observations":120};assert x["rank_contract_drift"] is True;assert x["exact_preregistration_reproduction_evidence_valid"] is False;assert x["terminal_artifacts_mutated"] is False;assert x["retry_or_repair_authorized"] is False
def test_audit_hash_binds_core():
 r=audit.build();assert r["manifest_hash"]==audit.canonical_hash({k:v for k,v in r.items() if k!="manifest_hash"})
