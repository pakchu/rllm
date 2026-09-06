import hashlib
import json

from training import build_cboe_volatility_surface_regime_crossing_relay_support as support


def test_cvsrc_source_support_is_frozen_terminal():
    assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest() == "fcabe0b1ebbfd996be6a5b735ead1804025a14d9c2f20e44693bb80fbcd79e68"
    assert hashlib.sha256(support.CLOCK.read_bytes()).hexdigest() == "6e4a70eac0f9423e9729a2fc6af900bd0fbb26b0beddd629e270e2e1a5f11eaf"
    payload = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == support.canonical_hash(core)
    assert payload["support_passed"] is False
    assert payload["decision"] == "terminal_source_support_reject"
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert [payload["support"][name]["events"] for name in support.SPLITS] == [0, 0, 0, 0]
