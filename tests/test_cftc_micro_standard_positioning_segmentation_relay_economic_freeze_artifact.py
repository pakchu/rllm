import hashlib,json
from pathlib import Path
from training import evaluate_cftc_micro_standard_positioning_segmentation_relay_economics as e
def test_evaluator_freeze_is_outcome_blind():
 v=json.loads(e.FREEZE.read_text());core={k:x for k,x in v.items() if k!='manifest_hash'};assert v['manifest_hash']==e.canonical_hash(core) and v['outcomes_opened'] is False;assert v['empty_clock_policy'].startswith('outcome-blind load_clock_allow_empty');assert v['evaluator']['sha256']==hashlib.sha256(Path(v['evaluator']['path']).read_bytes()).hexdigest()
