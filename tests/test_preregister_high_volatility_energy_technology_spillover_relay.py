from training import preregister_high_volatility_energy_technology_spillover_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == p.canonical_hash(core)
    assert value["policy_id"] == "HVETSR-12"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["policy"]["factor_weights"] == {"XLE": 0.5, "XLK": 0.5}
    assert value["policy"]["factor_abs_z_min"] == 1.0
    assert value["clock"]["hold"] == "12 elapsed hours"
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
