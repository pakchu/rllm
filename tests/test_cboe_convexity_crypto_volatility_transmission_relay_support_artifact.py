import json
from training import build_cboe_convexity_crypto_volatility_transmission_relay_support as support
def test_ccxtr_support_pass_is_hash_bound_and_outcome_blind():
 p=json.loads(support.RESULT.read_text());core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==support.canonical_hash(core);assert p["support_passed"] is True and p["advance_to_gross9_novelty"] is True and p["advance_to_economic_outcomes"] is False;assert p["btc_price_postentry_return_pnl_opened"] is False and p["gross9_rows_opened"] is False and all(p["support_checks"].values());assert [p["support"][n]["events"] for n in ("train","test","eval","final")]==[48,77,75,55]
