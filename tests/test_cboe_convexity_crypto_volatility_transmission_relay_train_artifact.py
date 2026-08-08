import hashlib,json
from training import evaluate_cboe_convexity_crypto_volatility_transmission_relay_economics as economics
def test_ccxtr_train_is_terminal_and_later_stages_sealed():
 p=economics.OUTPUTS["train"];assert hashlib.sha256(p.read_bytes()).hexdigest()=="897270fe587c4552a8e1bdffe0bce7e87c603dac842456071f8d4119d4068fb6";r=json.loads(p.read_text());core={k:v for k,v in r.items() if k!="manifest_hash"};assert r["manifest_hash"]==economics.canonical_hash(core) and r["passed"] is False and r["decision"]=="terminal_reject_no_repair" and r["later_stage_outcomes_opened"] is False;assert not economics.OUTPUTS["test"].exists() and not economics.OUTPUTS["eval"].exists() and not economics.OUTPUTS["final"].exists()
