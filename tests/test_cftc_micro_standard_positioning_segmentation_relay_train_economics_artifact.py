import hashlib,json
from pathlib import Path
R=Path("results/cftc_micro_standard_positioning_segmentation_relay_train_economics_2026-08-11.json")
def test_train_terminal_rejection_is_frozen():
 assert hashlib.sha256(R.read_bytes()).hexdigest()=="8a175e667090a5aee8259d036b33b32603b2a626e0e26e54cc36573d6c6c2bbc";x=json.loads(R.read_text());assert x["stage"]=="train" and x["passed"] is False and x["decision"]=="terminal_reject_no_repair";assert x["later_stage_outcomes_opened"] is False;assert x["primary"]["base"]["absolute_return_pct"]<0 and x["primary"]["stress"]["absolute_return_pct"]<0;assert x["primary"]["base"]["mean_gross_underlying_bp"]<20;assert x["advance_to_next_stage"] is False and x["advance_to_post_stage_volatility_audit"] is False
