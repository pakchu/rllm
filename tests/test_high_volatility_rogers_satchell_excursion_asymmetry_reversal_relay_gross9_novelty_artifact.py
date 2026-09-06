import hashlib,json
from training import evaluate_high_volatility_rogers_satchell_excursion_asymmetry_reversal_relay_gross9_novelty as n
def test_hvrsar_novelty_pass_reproduces_blindly():
 x=json.loads(n.OUTPUT.read_text());core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==n.canonical_hash(core)
 assert x["policy_id"]=="HVRSAR-8" and x["every_gross9_sleeve_passed"] is True and x["gross9_novelty_status"]=="passed" and x["advance_to_economic_outcomes"] is True
 assert x["evidence_boundary"]["outcomes_opened"] is False and x["evidence_boundary"]["btc_execution_rows_opened"]==0 and x["evidence_boundary"]["funding_rows_opened"]==0
 assert all(v["passed"] for v in x["gross9_sleeves"].values()) and hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest()=="45b443409b08c3631aab67bd792acf2c75703572decd00b55a1626c135754e1f"
