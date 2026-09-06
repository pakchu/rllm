import json
from training import evaluate_high_volatility_epu_hedge_relay_gross9_novelty as novelty
def test_novelty_artifact_passes_and_keeps_economics_closed():
 p=json.loads(novelty.OUTPUT.read_text());assert p["policy_id"]=="HVEPUH-24";assert p["source_support_passed"];assert p["every_gross9_sleeve_passed"];assert p["advance_to_economic_outcomes"];assert p["evidence_boundary"]["economic_outcome_rows_opened"]==0;assert not p["evidence_boundary"]["outcomes_opened"];h=p.pop("manifest_hash");assert novelty.chash(p)==h
