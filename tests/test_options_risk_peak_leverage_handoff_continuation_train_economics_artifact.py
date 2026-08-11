import json
from pathlib import Path


def test_train_rejection_is_terminal():
 v=json.loads(Path("results/options_risk_peak_leverage_handoff_continuation_train_economics_2026-08-11.json").read_text())
 assert v["policy_id"]=="ORPLHC-6" and v["stage"]=="train"
 assert v["passed"] is False and v["decision"]=="terminal_reject_no_repair"
 assert v["advance_to_next_stage"] is False and v["later_stage_outcomes_opened"] is False
 assert v["primary"]["base"]["absolute_return_pct"]==-5.767418957001236
 assert v["primary"]["base"]["mean_gross_underlying_bp"]==-28.720724837236716
 assert v["primary"]["cluster_signflip"]["pvalue"]==0.976520234797652
