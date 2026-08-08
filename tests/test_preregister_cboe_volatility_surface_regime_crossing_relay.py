import json

import pytest

from training import preregister_cboe_volatility_surface_regime_crossing_relay as prereg


def test_cvsrc_is_outcome_blind_singleton_with_causal_next_session_entry():
    payload = prereg.build()
    prereg.validate(payload)
    assert payload["policy_id"] == "CVSRC-24" and payload["singleton"] is True
    assert payload["outcomes_opened"] is False and payload["source_incidence_opened"] is False
    assert payload["clock"]["entry"].startswith("first later exact common Cboe source date")
    assert payload["features"]["current_observation_appended_only_after_all_ranks_are_fixed"] is True
    assert payload["research_boundary"]["future_can_rank_repair_or_reselect"] is False


def test_cvsrc_hash_rejects_tampering():
    payload = prereg.build()
    payload["thresholds"]["vix_rank_min"] = 0.50
    with pytest.raises(RuntimeError, match="canonical hash mismatch"):
        prereg.validate(payload)


def test_cvsrc_write_is_byte_stable(tmp_path):
    output = tmp_path / "prereg.json"
    first = prereg.write(output)
    second = prereg.write(output)
    assert first == second == json.loads(output.read_text())
