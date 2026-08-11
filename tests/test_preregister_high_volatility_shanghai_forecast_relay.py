from training import preregister_high_volatility_shanghai_forecast_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == p.canonical_hash(core)
    assert value["policy_id"] == "HVSFR-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["policy"]["var_lags"] == 1
    assert value["policy"]["var_trailing_rows"] == 252
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
