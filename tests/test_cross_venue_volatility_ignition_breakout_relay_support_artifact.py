import hashlib, json
from training import build_cross_venue_volatility_ignition_breakout_relay_support as support

def test_cvvib_source_support_is_frozen_terminal():
    assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest()=="e851d5df16db139c327267bdef4d5137f77744a6f6cee93eb53caeca19c54dd8"
    assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest()=="baec90d05757e8502f5072f36c70dce76689935b6a4e070801dc30e0cdb0bfc7"
    payload=json.loads(support.RESULT.read_text());core={key:value for key,value in payload.items() if key!="manifest_hash"}
    assert payload["manifest_hash"]==support.canonical_hash(core)
    assert payload["support_passed"] is False and payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["decision"]=="terminal_source_support_reject"
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["support"]["test"]["events"]<12
    assert payload["support"]["eval"]["events"]<12
    assert payload["support"]["final"]["events"]==23
    assert payload["support"]["final"]["minority_side_share"]<.20
    assert payload["support"]["final"]["max_month_share"]>.45
