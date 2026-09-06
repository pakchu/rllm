from training import preregister_high_volatility_exchange_deposit_pressure_relay as p


def test_frozen_preregistration_hash_and_boundary():
    value = p.build()
    core = dict(value)
    manifest = core.pop("manifest_hash")
    assert manifest == p.canonical_hash(core)
    assert value["policy_id"] == "HVEXDP-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["research_boundary"]["source_value_rows_opened"] is False


def test_deposit_direction_variation_and_source_are_fixed():
    value = p.build()
    assert value["source_plan"]["coin_metrics"]["metrics"] == list(p.METRICS)
    assert value["policy"]["deposit_change_lag_days"] == 1
    assert value["policy"]["variation_prior_minimum"] == 120
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert "change>0 maps short" in value["mechanism"]["side"]
    assert "change<0 maps long" in value["mechanism"]["side"]


def test_irregular_causal_clock_and_terminal_no_repair_are_fixed():
    value = p.build()
    assert "AssetEODCompletionTime" in value["clock"]["decision"]
    assert value["policy"]["entry_delay_minutes"] == 5
    assert value["policy"]["hold_hours"] == 24
    assert value["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert "no metric, vintage, change transform" in value["stopping_rule"]
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
