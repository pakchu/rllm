import json
from training import build_high_volatility_equity_adjusted_btc_residual_reversal_support as support
def test_support_is_terminal_before_later_gates():
 p=json.loads(support.RESULT.read_text());assert p["policy_id"]=="HVEABRR-12";assert not p["support_passed"];assert not p["advance_to_gross9_novelty"];assert not p["advance_to_economic_outcomes"];assert not p["gross9_rows_opened"];assert [p["support"][n]["events"] for n in support.SPLITS]==[22,34,41,11];assert p["support"]["final"]["max_month_share"]>.45;h=p.pop("manifest_hash");assert support.prereg.canonical_hash(p)==h
