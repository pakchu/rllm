import hashlib
import json

from training import (
    preregister_high_volatility_relative_daily_volume_continuation_relay as prereg,
)


def test_preregistration_is_singleton_outcome_blind_and_hash_bound():
    value = prereg.build()
    prereg.validate(value)
    assert value["policy_id"] == "HVRDV-8"
    assert value["research_boundary"]["candidate_count"] == 1
    assert value["research_boundary"]["grid"] is False
    assert value["research_boundary"]["candidate_incidence_opened"] is False
    assert value["research_boundary"]["postentry_return_or_pnl_opened"] is False
    assert value["research_boundary"]["gross9_rows_opened"] is False
    assert value["policy"]["relative_daily_volume_days"] == 20
    assert value["policy"]["relative_daily_volume_average_level"] == 1.0
    assert value["clock"]["hold"] == "8 elapsed hours"


def test_written_preregistration_matches_builder_and_manifest():
    value = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    assert value == prereg.build()
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == prereg.canonical_hash(core)
    assert hashlib.sha256(prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
