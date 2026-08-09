import hashlib
import json

from training import preregister_high_volatility_dydx_auto_deleveraging_handoff_relay as prereg


def test_manifest_is_canonical_and_outcome_blind() -> None:
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    expected = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert payload["manifest_hash"] == expected
    assert payload["policy_id"] == "HVDADH-8"
    assert payload["oos_outcomes_opened"] is False
    assert payload["oos_source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["grid"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False


def test_frozen_source_clock_and_gates() -> None:
    payload = prereg.build()
    assert payload["source_contract"]["endpoint"] == "/trades/perpetualMarket/BTC-USD"
    assert "DELEVERAGED" in payload["source_contract"]["accepted_types"]
    assert payload["policy"]["forced_notional_rank_min"] == 0.80
    assert payload["policy"]["btc_variation_rank_min"] == 0.65
    assert payload["policy"]["hold_hours"] == 8
    assert payload["oos_clock"]["entry"] == "D+5m BTCUSDT perpetual open"
    assert payload["source_support_gates"]["complete_cursor_replay_required"] is True
    assert payload["economic_gates"]["stop_on_first_failure"] is True
    assert "no source cursor" in payload["stopping_rule"]
