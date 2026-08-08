import hashlib,json
from training import build_joint_volatility_intrahour_acceleration_relay_support as support

def test_jviar_source_support_is_frozen_terminal_without_outcomes():
 assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest()=="0f20d809e561229d14826e045524930703e78563edb767aa25989362978ceb02"
 assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest()=="52ba81e866a352d659efac8ed582953bb0b81cdfbdb7b5552eccc896229a776c"
 p=json.loads(support.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"}
 assert p["manifest_hash"]==support.canonical_hash(core)
 assert p["support_passed"] is False and p["advance_to_gross9_novelty"] is False and p["advance_to_economic_outcomes"] is False
 assert p["decision"]=="terminal_source_support_reject" and p["postentry_return_pnl_execution_price_opened"] is False and p["gross9_rows_opened"] is False
 assert [p["support"][n]["events"] for n in ("train","test","eval","final")]==[79,58,34,50]
 assert p["support"]["final"]["minority_side_share"]==.12 and p["support_checks"]["final_side_balance"] is False
 assert all(not x["promotion_authorized"] for x in p["controls"].values())
