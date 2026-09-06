import json
from training import build_high_volatility_turn_of_candle_momentum_support as support
def test_support_passes_and_keeps_later_gates_closed():
 p=json.loads(support.RESULT.read_text());assert p["policy_id"]=="HVTOCM-30M";assert p["support_passed"];assert p["advance_to_gross9_novelty"];assert not p["advance_to_economic_outcomes"];assert not p["gross9_rows_opened"];assert [p["support"][n]["events"] for n in support.SPLITS]==[67,137,130,63];assert not p["information_embargo_audit"]["eligibility_inputs_after_00_30_opened"];h=p.pop("manifest_hash");assert support.prereg.canonical_hash(p)==h
