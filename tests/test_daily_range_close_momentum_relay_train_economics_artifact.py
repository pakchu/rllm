import hashlib,json
from training import evaluate_daily_range_close_momentum_relay_economics as economics

def test_drcmr_train_economics_is_terminal_and_later_stages_closed():
 assert hashlib.sha256(economics.OUTPUTS["train"].read_bytes()).hexdigest()=="a4844493976297e64c5b8de2bc73493e5a805dd67e60ec1a4da80faa1617b014";p=json.loads(economics.OUTPUTS["train"].read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==economics.canonical_hash(core);assert p["stage"]=="train" and p["passed"] is False and p["decision"]=="terminal_reject_no_repair" and p["advance_to_next_stage"] is False and p["later_stage_outcomes_opened"] is False;assert p["primary"]["base"]["absolute_return_pct"]<0 and p["primary"]["stress"]["absolute_return_pct"]<0;assert not economics.OUTPUTS["test"].exists() and not economics.OUTPUTS["eval"].exists() and not economics.OUTPUTS["final"].exists()
