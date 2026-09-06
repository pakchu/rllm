import hashlib,json
from training import build_volatility_shock_disagreement_absorption_relay_support as s
def test_vsdar_support_frozen_pass():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=="e8d3efe78556be7762c010b495a363db0d45a62d90a5cc24a2e9c83796730751";assert hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()=="fbdbd8a5eba77e715880f0b2db1572df524074c7dca3ad3b3f30ef4cb4773027";p=json.loads(s.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==s.chash(core) and p["support_passed"] is True and p["advance_to_gross9_novelty"] is True;assert p["advance_to_economic_outcomes"] is False and p["postentry_return_pnl_execution_price_opened"] is False;assert [p["support"][n]["events"] for n in s.SPLITS]==[25,57,55,34]
