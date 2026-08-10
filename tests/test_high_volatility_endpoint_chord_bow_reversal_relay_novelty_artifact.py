import hashlib,json
from pathlib import Path
RESULT=Path("results/high_volatility_endpoint_chord_bow_reversal_relay_gross9_novelty_2026-08-10.json")
def test_hvecbr_novelty_failure_is_frozen_and_outcomes_stay_sealed():
 assert hashlib.sha256(RESULT.read_bytes()).hexdigest()=="bb922fa37a0872e78825ac6392fa2178d0046aa60f9fcbe9ab7855a6f400d8da";x=json.loads(RESULT.read_text());assert x["every_gross9_sleeve_passed"] is False and x["advance_to_economic_outcomes"] is False and x["evidence_boundary"]["outcomes_opened"] is False;failed=x["gross9_sleeves"]["frozen_annual_rank7"];assert failed["metrics"]["one_to_one_6h_max_matched_share"]>.35 and failed["checks"]["one_to_one_6h_max_matched_share"] is False
