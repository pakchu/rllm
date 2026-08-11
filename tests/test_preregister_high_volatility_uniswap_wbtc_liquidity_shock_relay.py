from training import preregister_high_volatility_uniswap_wbtc_liquidity_shock_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == p.canonical_hash(core)
    assert value["policy_id"] == "HVUWLS-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["policy"]["shock_midrank_min"] == 0.80
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_pool_source_and_causal_day_boundary_are_fixed():
    value = p.build()
    assert value["policy"]["pool"].lower() == "0x99ac8ca7087fa4a2a1fb6357269965a2014abc35"
    assert value["policy"]["swap_topic0"] == p.SWAP_TOPIC0
    assert value["policy"]["confirmation_blocks"] == 64
    assert value["source_plan"]["ethereum_uniswap_v3_logs"]["two_host_exact_replay_required"] is True
    assert value["source_plan"]["ethereum_uniswap_v3_logs"]["full_day_boundary_header_replay_required"] is True
    assert "zero-log days" in value["features"]["day_completeness"]
    assert value["research_boundary"]["historical_uniswap_swap_logs_opened"] is False


def test_direction_and_no_repair_are_explicit():
    value = p.build()
    assert "amount0<0 maps long" in value["features"]["side"]
    assert "no pool, provider, event, finality" in value["stopping_rule"]
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
