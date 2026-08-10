import hashlib
import json

from training import preregister_high_volatility_mstr_relative_short_volume_pressure_relay as prereg


def test_manifest_is_outcome_blind_and_hash_bound() -> None:
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    expected = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert payload["manifest_hash"] == expected
    assert payload["oos_outcomes_opened"] is False
    assert payload["oos_source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["source_contract"]["symbols"] == ["MSTR", "QQQ"]


def test_policy_and_first_failure_contract_are_frozen() -> None:
    payload = prereg.build()
    assert payload["policy"] == {
        "history_source_days": 252,
        "minimum_history_source_days": 126,
        "absolute_pressure_change_rank_min": 0.80,
        "btc_variation_rank_min": 0.65,
        "entry_delay_minutes": 5,
        "hold_hours": 24,
        "leverage": 0.5,
        "base_cost_per_notional_side": 0.0006,
        "stress_cost_per_notional_side": 0.001,
    }
    assert payload["economic_gates"]["stop_on_first_failure"] is True
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False
