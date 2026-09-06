import hashlib,json
from training import evaluate_high_volatility_ordinal_pattern_classifier_relay_gross9_novelty as n
def test_hvocpr_novelty_failure_is_terminal_and_outcome_blind():
 assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=="5514eadaa45ab370f3e976b75bd99b91fa8d1ebf04ed4ace32eab19a1de73006";p=json.loads(n.OUTPUT.read_text());assert p["manifest_hash"]==n.chash({k:v for k,v in p.items() if k!="manifest_hash"});assert p["gross9_novelty_status"]=="failed" and p["every_gross9_sleeve_passed"] is False and p["advance_to_economic_outcomes"] is False;assert p["gross9_sleeves"]["cand_rex_veto_7"]["metrics"]["one_to_one_6h_max_matched_share"]==.36;assert p["evidence_boundary"]["outcomes_opened"] is False and p["evidence_boundary"]["economic_outcome_rows_opened"]==0
