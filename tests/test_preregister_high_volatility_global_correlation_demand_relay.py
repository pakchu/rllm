from training import preregister_high_volatility_global_correlation_demand_relay as p


def test_frozen_preregistration():
    value=p.build(); core=dict(value); manifest=core.pop("manifest_hash")
    assert manifest==p.canonical_hash(core)
    assert value["policy_id"]=="HVGCDR-24" and value["outcomes_opened"] is False
    assert value["policy"]["equity_symbols"]==["SPY","EFA","EEM"]
    assert value["policy"]["correlation_sessions"]==20
    assert value["policy"]["magnitude_midrank_min"]==0.80
    assert value["policy"]["variation_midrank_min"]==0.65
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_direction_global_aggregation_and_nonrepair_are_explicit():
    value=p.build()
    assert "negative strict sign" in value["mechanism"]["side"]
    assert "equal arithmetic mean" in value["features"]["global_correlation"]
    assert value["research_boundary"]["prior_single_spy_formula_event_set_or_control_reused"] is False
    assert value["research_boundary"]["prior_bscbr_failure_used_to_set_global_formula_or_threshold"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
