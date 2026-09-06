import json
from training import evaluate_mexican_peso_risk_barometer_relay_gross9_novelty as novelty
def test_mxrbr_novelty_pass_is_outcome_sealed():
 r=json.loads(novelty.OUTPUT.read_text());assert r["policy_id"]=="MXRBR-12";assert r["every_gross9_sleeve_passed"] is True;assert r["gross9_novelty_status"]=="passed";assert r["advance_to_economic_outcomes"] is True;b=r["evidence_boundary"];assert b["btc_execution_rows_opened"]==0;assert b["funding_rows_opened"]==0;assert b["economic_outcome_rows_opened"]==0;assert b["outcomes_opened"] is False
def test_mxrbr_passes_every_frozen_metric():
 r=json.loads(novelty.OUTPUT.read_text())
 for x in r["gross9_sleeves"].values():assert x["passed"] is True and all(x["checks"].values()) and x["metrics"]["exact_entry_jaccard"]==0.
