import json
from training import evaluate_spot_participation_volatility_ignition_relay_gross9_novelty as novelty
def test_spvir_novelty_terminal_without_outcomes():
 p=json.loads(novelty.OUTPUT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==novelty.canonical_hash(core);assert p["every_gross9_sleeve_passed"] is False and p["advance_to_economic_outcomes"] is False and p["evidence_boundary"]["outcomes_opened"] is False;assert max(x["metrics"]["one_to_one_6h_max_matched_share"] for x in p["gross9_sleeves"].values())==0.6470588235294118
