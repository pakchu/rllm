import json
from training import build_high_volatility_dvol_variation_risk_relay_support as support
def test_support_artifact_passes_without_opening_later_data():
 p=json.loads(support.RESULT.read_text());assert p["policy_id"]=="HVDVVR-12";assert p["support_passed"];assert p["advance_to_gross9_novelty"];assert not p["advance_to_economic_outcomes"];assert not p["postentry_return_pnl_execution_price_opened"];assert not p["gross9_rows_opened"];assert [p["support"][n]["events"] for n in support.SPLITS]==[26,65,75,37];h=p.pop("manifest_hash");assert support.prereg.canonical_hash(p)==h
