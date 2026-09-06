import json

from training import preregister_high_volatility_usd_index_forecast_threshold_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == p.canonical_hash(core)
    assert value["policy_id"] == "HVDXYFT-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["policy"]["var_lags"] == 1
    assert value["policy"]["var_trailing_rows"] == 252
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert value["source_plan"]["uup"]["symbol"] == "UUP"
    assert value["policy"]["forecast_magnitude_midrank_min"] == 0.60
    assert len(value["diagnostic_controls"]["names"]) == 6
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_committed_artifact_matches_builder():
    value = json.loads(p.DEFAULT_OUTPUT.read_text())
    p.validate(value)
    assert value == p.build()
