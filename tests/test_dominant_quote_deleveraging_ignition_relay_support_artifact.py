import hashlib,json
from training import build_dominant_quote_deleveraging_ignition_relay_support as support
def test_dqdir_source_support_is_frozen_pass_without_outcomes():
 assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest()=="918c66e59f620f7191158a66e9129e87a8a0726506d4956c27a16028bae4090e"
 assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest()=="304f4c27ccafb60e396b39a8cf054409ccc5ca7fdc382d4fbd11f7b429dd87e7"
 p=json.loads(support.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==support.canonical_hash(core)
 assert p["support_passed"] is True and p["advance_to_gross9_novelty"] is True and p["advance_to_economic_outcomes"] is False and p["decision"]=="pass_to_novelty"
 assert p["postentry_return_pnl_execution_price_opened"] is False and p["gross9_rows_opened"] is False
 assert [p["support"][n]["events"] for n in ("train","test","eval","final")]==[42,69,59,52]
 assert all(not x["promotion_authorized"] for x in p["controls"].values())
