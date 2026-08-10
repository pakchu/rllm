import hashlib
import json

from training import preregister_high_volatility_cross_asset_tail_dependence_continuation_relay as prereg


def test_preregistration_is_hash_bound_and_outcome_blind() -> None:
    payload = prereg.build(); core = {k: v for k, v in payload.items() if k != "manifest_hash"}
    assert payload["manifest_hash"] == hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert payload["oos_outcomes_opened"] is False
    assert payload["oos_source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False


def test_singleton_policy_and_no_repair_are_frozen() -> None:
    payload = prereg.build()
    assert payload["policy"]["history_blocks"] == 270
    assert payload["policy"]["minimum_history_blocks"] == 180
    assert payload["policy"]["tail_probability"] == 0.20
    assert payload["policy"]["hold_hours"] == 8
    assert payload["research_boundary"]["grid"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False
    assert payload["economic_gates"]["stop_on_first_failure"] is True
