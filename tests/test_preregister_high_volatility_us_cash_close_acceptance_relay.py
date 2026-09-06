import hashlib
import json

from training import preregister_high_volatility_us_cash_close_acceptance_relay as prereg


def test_manifest_is_hash_bound_and_outcome_blind() -> None:
    payload = prereg.build(); core = {k: v for k, v in payload.items() if k != "manifest_hash"}
    assert payload["manifest_hash"] == hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert payload["oos_outcomes_opened"] is False and payload["oos_source_incidence_opened"] is False and payload["gross9_rows_opened"] is False


def test_dst_session_policy_and_calendar_are_frozen() -> None:
    payload = prereg.build()
    assert payload["source_contract"]["timezone"] == "America/New_York"
    assert "2025-01-09" in payload["source_contract"]["excluded_local_dates"]
    assert "2026-07-02" in payload["source_contract"]["excluded_local_dates"]
    assert payload["policy"]["hold_hours"] == 8
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False
