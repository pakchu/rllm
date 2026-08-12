import json
from pathlib import Path
RESULT=Path("results/high_volatility_speculative_alt_rotation_relay_train_economics_2026-08-12.json")
def test_hvsarr_train_artifact_is_terminal():
 r=json.loads(RESULT.read_text());assert r["policy_id"]=="HVSARR-8";assert r["stage"]=="train";assert r["passed"] is False;assert r["decision"]=="terminal_reject_no_repair";assert r["later_stage_outcomes_opened"] is False;assert r["primary"]["base"]["absolute_return_pct"]>0;assert r["primary"]["base"]["mean_gross_underlying_bp"]>=20;assert r["checks"]["cagr_to_strict_mdd_min_3"] is False;assert r["checks"]["cluster_signflip_p_max_0_1"] is False
