import json
from training import build_high_volatility_epu_hedge_relay_support as support
def test_source_support_passes_and_keeps_later_outcomes_closed():
 p=json.loads(support.RESULT.read_text());assert p["policy_id"]=="HVEPUH-24";assert p["support_passed"];assert p["advance_to_gross9_novelty"];assert not p["advance_to_economic_outcomes"];assert not p["postentry_return_pnl_execution_price_opened"];assert not p["gross9_rows_opened"];assert [p["support"][n]["events"] for n in support.SPLITS]==[19,35,27,21];h=p.pop("manifest_hash");assert support.prereg.canonical_hash(p)==h
