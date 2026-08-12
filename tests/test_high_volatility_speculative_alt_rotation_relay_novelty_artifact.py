import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_speculative_alt_rotation_relay_gross9_novelty_2026-08-12.json")
def test_hvsarr_novelty_artifact_passes_and_is_hash_bound():
 r=json.loads(RESULT.read_text());assert r["policy_id"]=="HVSARR-8";assert r["gross9_novelty_status"]=="passed";assert r["every_gross9_sleeve_passed"] is True;assert r["advance_to_economic_outcomes"] is True;assert r["evidence_boundary"]["outcomes_opened"] is False;assert all(x["passed"] for x in r["gross9_sleeves"].values());assert hashlib.sha256(RESULT.read_bytes()).hexdigest()=="0fd22a847e0fd81a2de294f241571e22abe4864f720b3acd0b6b98410d1ee27a"
