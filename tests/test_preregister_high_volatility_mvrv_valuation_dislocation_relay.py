from training import preregister_high_volatility_mvrv_valuation_dislocation_relay as p


def test_frozen_preregistration_hash_and_boundary():
    value = p.build()
    core = dict(value)
    manifest = core.pop("manifest_hash")
    assert manifest == p.canonical_hash(core)
    assert value["policy_id"] == "HVMVRV-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["research_boundary"]["source_value_rows_opened"] is False


def test_metric_local_value_rule_and_volatility_are_fixed():
    value = p.build()
    assert value["source_plan"]["coin_metrics"]["metrics"] == list(p.METRICS)
    assert value["policy"]["mvrv_reference_days"] == 30
    assert value["policy"]["mvrv_absolute_z_min"] == 0.5
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert "z-score>=+0.5 maps short" in value["mechanism"]["side"]
    assert "z-score<=-0.5 maps long" in value["mechanism"]["side"]


def test_clock_controls_and_terminal_no_repair_are_fixed():
    value = p.build()
    assert value["clock"]["entry"] == "exact BTCUSDT five-minute open at 12:05 UTC"
    assert value["policy"]["hold_hours"] == 24
    assert value["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert "no metric, vintage, reference" in value["stopping_rule"]
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
