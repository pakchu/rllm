from training import preregister_high_volatility_ethereum_staking_deposit_pressure_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == p.canonical_hash(core)
    assert value["policy_id"] == "HVESDP-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["policy"]["same_weekday_lag_days"] == 7
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_execution_log_source_and_availability_are_fixed():
    value = p.build()
    source = value["source_plan"]["ethereum_execution_logs"]
    assert source["chain_id"] == "0x1"
    assert source["contract"] == p.DEPOSIT_CONTRACT
    assert source["topic0"] == p.DEPOSIT_EVENT_TOPIC
    assert source["confirmation_blocks"] == 64
    assert source["two_host_exact_replay_required"] is True
    assert value["policy"]["deposit_event_signature"] == p.DEPOSIT_EVENT_SIGNATURE
    assert value["research_boundary"]["historical_execution_logs_opened"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
