import hashlib
import json

from training import preregister_high_volatility_signed_path_hysteresis_onset as prereg


def test_manifest_is_canonical_and_outcome_blind() -> None:
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert payload["manifest_hash"] == digest
    assert payload["policy_id"] == "HVSPH-8"
    assert payload["oos_outcomes_opened"] is False
    assert payload["oos_source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["grid"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False


def test_frozen_clock_and_terminal_contract() -> None:
    payload = prereg.build()
    assert payload["policy"]["realized_variation_rank_min"] == 0.80
    assert payload["policy"]["absolute_area_rank_min"] == 0.80
    assert payload["policy"]["hold_hours"] == 8
    assert payload["oos_clock"]["entry"] == "D+5m BTCUSDT perpetual open"
    assert payload["economic_gates"]["stop_on_first_failure"] is True
    assert "no threshold" in payload["stopping_rule"]
