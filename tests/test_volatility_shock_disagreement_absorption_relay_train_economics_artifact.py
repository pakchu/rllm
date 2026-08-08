import hashlib,json
from training import evaluate_volatility_shock_disagreement_absorption_relay_economics as e
def test_vsdar_train_economics_frozen_terminal():
 p=e.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="4f9b0539729a0980cd035bf06fa5e061d2821c91427cecd2256ef5c0116a8b64";d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!="manifest_hash"};assert d["manifest_hash"]==e.canonical_hash(core) and d["passed"] is False and d["decision"]=="terminal_reject_no_repair";assert d["advance_to_next_stage"] is False and d["later_stage_outcomes_opened"] is False;assert d["primary"]["base"]["absolute_return_pct"]<0 and d["primary"]["base"]["mean_gross_underlying_bp"]<0
