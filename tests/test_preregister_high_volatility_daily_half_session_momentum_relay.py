from training import preregister_high_volatility_daily_half_session_momentum_relay as prereg


def test_frozen_outcome_blind_singleton():
    value=prereg.build();prereg.validate(value)
    assert value["policy_id"]=="HVDHSM-12" and value["singleton"] is True
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["research_boundary"]["candidate_source_values_opened"] is False
    assert value["research_boundary"]["grid"] is False and value["research_boundary"]["repair_of_prior_candidate"] is False


def test_half_session_and_gates_are_frozen():
    value=prereg.build()
    assert "[D 00:00,D 12:00) UTC" in value["features"]["first_half"]
    assert value["clock"]["entry"].startswith("exact BTCUSDT D 12:05 UTC")
    assert value["clock"]["hold"]=="12 elapsed hours"
    assert value["policy"]["variation_rank_min"]==0.65
    assert value["economic_gates"]["stop_on_first_failure"] is True


def test_canonical_hash():
    value=prereg.build();core={k:v for k,v in value.items() if k!="manifest_hash"}
    assert value["manifest_hash"]==prereg.canonical_hash(core)
