import hashlib,json
from training import build_kimchi_premium_acceleration_relay_support as support

def test_kpar_support_artifact_is_terminal_before_outcomes():
 assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest()=="a17696c92b43793c6daf47760a23085e62d8c8a6c6ea25d67bd0aec1b5f30c4e";p=json.loads(support.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==support.chash(core);assert p["decision"]=="terminal_source_support_reject" and p["support_passed"] is False and p["advance_to_gross9_novelty"] is False and p["advance_to_economic_outcomes"] is False;assert p["postentry_return_pnl_execution_price_opened"] is False and p["gross9_rows_opened"] is False;assert [p["support"][s]["events"] for s in support.SPLITS]==[0,0,0,0]
