import hashlib
import json

from training import preregister_high_volatility_premium_change_autocorrelation_relay as prereg


def test_frozen_singleton_blind():
    value = prereg.build()
    prereg.validate(value)
    assert value["policy_id"] == "HVPACR-12"
    assert value["policy"]["five_minute_bars"] == 48
    assert value["policy"]["persistence_rank_min"] == 0.8
    assert value["clock"]["hold"] == "12 elapsed hours"
    boundary = value["research_boundary"]
    assert boundary["candidate_count"] == 1 and boundary["grid"] is False
    assert boundary["candidate_incidence_opened"] is False
    assert boundary["postentry_return_or_pnl_opened"] is False
    assert boundary["gross9_rows_opened"] is False


def test_written_matches_builder_and_utf8_hash():
    value = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    assert value == prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
    expected = hashlib.sha256(json.dumps({"한글": "premium"}, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert prereg.canonical_hash({"한글": "premium"}) == expected
