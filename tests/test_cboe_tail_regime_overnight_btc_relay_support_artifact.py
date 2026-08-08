import hashlib,json
from training import build_cboe_tail_regime_overnight_btc_relay_support as s
def test_cvtlr_support_frozen_terminal():
 assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest()=="8556f5bf71d61cc7d52db2f04af692c312fef3a58dbca9cb627ebc7a84092a0c";assert hashlib.sha256(s.CLOCK.read_bytes()).hexdigest()=="d4bdddb7e66ee7cbe0cda64538e50a037ff296cb41eb092803f20f2836b73bf8";p=json.loads(s.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==s.canonical_hash(core) and p["support_passed"] is False and p["decision"]=="terminal_source_support_reject";assert p["advance_to_gross9_novelty"] is False and p["advance_to_economic_outcomes"] is False;assert p["support"]["train"]["events"]==49 and p["support"]["train"]["longs"]==0
