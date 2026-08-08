import hashlib,json
from pathlib import Path
from training import build_funding_acceleration_reversal_relay_support as support
ARTIFACT=Path("results/funding_acceleration_reversal_relay_source_contract_failure_2026-08-09.json")
def test_farr_source_contract_failure_is_terminal_and_sealed():
 assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()=="5d95b5a99e4c61c4be236d863f3c2628a6c9e1441a8ace818c37d388bdc85d4f";r=json.loads(ARTIFACT.read_text());assert r["policy_id"]=="FARR-6";assert r["duplicate_boundaries"]==184;assert r["decision"]=="terminal_source_contract_reject_no_repair";assert r["support_passed"] is False;assert r["advance_to_gross9_novelty"] is False;assert r["candidate_incidence_opened"] is False;assert r["postentry_return_pnl_execution_price_opened"] is False;assert r["gross9_rows_opened"] is False;assert not support.RESULT.exists();assert not support.CLOCK.exists()
def test_farr_source_failure_manifest_is_hash_bound():
 r=json.loads(ARTIFACT.read_text());assert r["manifest_hash"]==support.chash({k:v for k,v in r.items() if k!="manifest_hash"});assert r["source_evaluator"]["sha256"]==support.sha(Path(r["source_evaluator"]["path"]))
